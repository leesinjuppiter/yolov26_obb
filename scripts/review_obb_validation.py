#!/usr/bin/env python3
"""将 V 中已有的双头预测与 OBB 标注做几何对照，生成复核材料。"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from ultralytics.utils.ops import xywhr2xyxyxyxy, xyxyxyxy2xywhr

from inspect_obb_head_disagreement import match_boxes

NAMES = ("coal", "rockbolt", "mesh", "stick", "board", "cap", "bottle")
COLORS = dict(correct=(85, 238, 112), wrong_class=(60, 80, 255),
              poor_localization=(0, 209, 255), unmatched_prediction=(245, 105, 231))


def load_ground_truth(path, image_shape):
    entries = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        cls, *coords = line.split()
        if len(coords) != 8 or not 0 <= int(cls) < len(NAMES):
            raise ValueError(f"旋转框标签格式错误: {path.name}")
        vertices = np.asarray(coords, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(vertices).all() or not ((0 <= vertices) & (vertices <= 1)).all():
            raise ValueError(f"标签坐标越界: {path.name}")
        vertices *= (image_shape[1], image_shape[0])
        entries.append((int(cls), vertices))
    if not entries:
        return [], torch.empty((0, 5))
    return entries, torch.from_numpy(xyxyxyxy2xywhr(np.stack([p for _, p in entries])))


def compare_head(gt, gt_boxes, preds, association, hit):
    p_boxes = torch.as_tensor([p["xywhr"] for p in preds], dtype=torch.float32).reshape(-1, 5)
    matches, missing, extra = match_boxes(gt_boxes, p_boxes, association)
    rows, pred_states = [], {}
    for g, p, similarity in matches:
        gt_cls, pt = gt[g][0], preds[p]
        class_equal = gt_cls == pt["class_id"]
        state = ("poor_localization" if similarity < hit else
                 "correct" if class_equal else "wrong_class")
        pred_states[p] = state
        rows.append(dict(gt_id=f"G{g+1}", gt_class=NAMES[gt_cls], pred_id=f"P{p+1}",
                         pred_class=NAMES[pt["class_id"]], score=pt["score"],
                         probiou=similarity, class_equal=class_equal, status=state))
    for g in missing:
        rows.append(dict(gt_id=f"G{g+1}", gt_class=NAMES[gt[g][0]], status="missed_gt"))
    for p in extra:
        pt = preds[p]
        pred_states[p] = "unmatched_prediction"
        rows.append(dict(pred_id=f"P{p+1}", pred_class=NAMES[pt["class_id"]],
                         score=pt["score"], status="unmatched_prediction"))
    assert len(pred_states) == len(preds)
    assert sum(bool(r.get("gt_id")) for r in rows) == len(gt)
    return rows, pred_states


def paint(image, items, heading, footer=""):
    width, header = 665, 56
    gain = width / image.shape[1]
    view = cv2.resize(image, (width, round(image.shape[0] * gain)), interpolation=cv2.INTER_AREA)
    for vertices, tag, color in items:
        p = np.round(vertices * gain).astype(np.int32)
        cv2.polylines(view, [p], True, color, 1, cv2.LINE_AA)
        x, y = p.min(axis=0)
        x = int(np.clip(x, 2, view.shape[1]-86))
        y = int(np.clip(y, 14, view.shape[0]-2))
        cv2.putText(view, tag, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .35, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(view, tag, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    canvas = cv2.copyMakeBorder(view, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(34,38,37))
    cv2.putText(canvas, heading, (14,24), cv2.FONT_HERSHEY_SIMPLEX, .65, (247,250,246), 1, cv2.LINE_AA)
    if footer:
        cv2.putText(canvas, footer, (14,44), cv2.FONT_HERSHEY_SIMPLEX, .40, (192,211,197), 1, cv2.LINE_AA)
    return canvas


def preview(path, image, gt, record, by_head):
    gt_panel = paint(image, [(p, f"G{i+1}:{NAMES[c]}", (235,230,95)) for i,(c,p) in enumerate(gt)],
                     "OBB GROUND TRUTH", f"{len(gt)} annotated objects")
    panels = []
    for head in ("one2one", "one2many"):
        predictions = record[head]
        boxes = np.asarray([p["xywhr"] for p in predictions], dtype=np.float32).reshape(-1,5)
        vertices = xywhr2xyxyxyxy(boxes)
        items = [(v, f"P{i+1}:{NAMES[p['class_id']]}", COLORS[by_head[head][i]])
                 for i,(v,p) in enumerate(zip(vertices,predictions))]
        panels.append(paint(image, items, head.upper(),
                            "green correct | red class | yellow box | purple unmatched"))
    grid = cv2.vconcat([
        cv2.hconcat([paint(image, [], "INPUT FRAME", record["image"]), gt_panel]),
        cv2.hconcat(panels),
    ])
    if not cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 84]):
        raise OSError(f"无法保存对照图 {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--association", type=float, default=.30)
    parser.add_argument("--hit", type=float, default=.50)
    args = parser.parse_args()
    if args.v_root.name != "V_validation" or not 0 < args.association < args.hit <= 1:
        parser.error("只支持 V_validation，且须满足 0 < 关联门槛 < 合格门槛 ≤ 1")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    records = [json.loads(row) for row in args.predictions.read_text(encoding="utf-8").splitlines() if row.strip()]
    if not records:
        raise ValueError("预测文件没有图片记录")
    seen, detail, by_image, candidates = set(), [], [], []
    overall = {h: Counter() for h in ("one2one", "one2many")}
    for rec in records:
        name = rec["image"]
        if not re.fullmatch(r"d4911_v_\d{6}\.jpg", name) or name in seen:
            raise ValueError("预测应来自 V 中的不同图片")
        seen.add(name)
        stem = Path(name).stem
        image = cv2.imread(str(args.v_root / "images" / name))
        if image is None:
            raise OSError(f"无法读取 {name}")
        gt, gt_boxes = load_ground_truth(args.v_root / "labels" / (stem+".txt"), image.shape)
        by_head, counts, gt_status = {}, {}, {}
        for head in ("one2one", "one2many"):
            rows, by_head[head] = compare_head(gt, gt_boxes, rec[head], args.association, args.hit)
            one = Counter(r["status"] for r in rows)
            overall[head].update(one)
            counts[head] = dict(one)
            gt_status[head] = {r["gt_id"]:r for r in rows if "gt_id" in r}
            detail.extend(dict(image=name, head=head, **row) for row in rows)
            for row in rows:
                if row["status"] != "correct" and row.get("pred_class") != "coal" and "pred_id" in row:
                    candidates.append(dict(image=name, head=head, **row))
        for label, a in gt_status["one2one"].items():
            b = gt_status["one2many"][label]
            if a["gt_class"] != "coal" and (a["status"] != "correct" or b["status"] != "correct"):
                candidates.append(dict(image=name, gt_id=label, gt_class=a["gt_class"],
                                       one2one=a["status"], one2many=b["status"]))
        by_image.append(dict(image=name, ground_truth=len(gt), **counts))
        preview(args.out/(stem+"_review.jpg"), image, gt, rec, by_head)

    fields = ["image","head","gt_id","gt_class","pred_id","pred_class","score","probiou","class_equal","status"]
    with (args.out/"frame_box_review.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail)
    sort_key = lambda r: ("one2one" not in r, -r.get("score", 0), r["image"])
    review = sorted(candidates,key=sort_key)
    report = {"images":len(records),"gt_boxes":sum(r["ground_truth"] for r in by_image),
              "association_probiou":args.association,"hit_probiou":args.hit,
              "counts":{h:dict(c) for h,c in overall.items()},"per_image":by_image,
              "review_candidates":review[:12],
              "note":"Heuristic geometric review under pilot scores. Not official AP or confirmed errors."}
    (args.out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="review_candidates"},ensure_ascii=False))


if __name__ == "__main__":
    main()
