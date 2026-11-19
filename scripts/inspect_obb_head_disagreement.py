#!/usr/bin/env python3
"""在 V 上检查未融合 YOLO26 OBB 双头，记录匹配及分歧，供人工校准。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from ultralytics import __version__
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.head import OBB26
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import xywhr2xyxyxyxy


def validation_examples(root, count, seed):
    groups = defaultdict(list)
    validation = root / "V_validation" / "images"
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "V_validation":
                image = (root / row["image"]).resolve(strict=True)
                if image.parent != validation.resolve():
                    raise ValueError("验证集清单指向了 V 之外")
                groups[row["capture_date"]].append(image)
    if sum(map(len, groups.values())) < count:
        raise ValueError("V 中图片数量不足")
    rng = random.Random(seed)
    for images in groups.values():
        rng.shuffle(images)
    selected = []
    while len(selected) < count:
        for day in sorted(groups):
            if groups[day]:
                selected.append((groups[day].pop(), day))
                if len(selected) == count:
                    break
    return selected


def input_tensor(image, size, device):
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    pad_x = round((size - round(w * ratio)) / 2 - 0.1)
    pad_y = round((size - round(h * ratio)) / 2 - 0.1)
    padded = LetterBox(new_shape=(size, size), auto=False)(image=image)
    rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).to(device, dtype=torch.float32)[None] / 255.0, (ratio, pad_x, pad_y)


def decoded_candidates(head, raw, transform, args, dedup):
    for key in ("boxes", "scores", "angle"):
        if key not in raw:
            raise ValueError(f"预测分支没有 {key}，请检查模型是否融合")
    decoded = head._inference(raw)[0].T.float()
    if decoded.shape[1] != 4 + head.nc + 1 or not torch.isfinite(decoded).all():
        raise ValueError("OBB 解码尺寸或数值异常")
    scores, classes = decoded[:, 4:4 + head.nc].max(dim=1)
    indices = scores.argsort(descending=True)
    indices = indices[scores[indices] >= args.conf][:args.pre_k]
    boxes = decoded[indices][:, [0, 1, 2, 3, 4 + head.nc]].clone()
    scores, classes = scores[indices], classes[indices]
    valid = (boxes[:, 2:4] > 0).all(dim=1)
    boxes, scores, classes, indices = boxes[valid], scores[valid], classes[valid], indices[valid]
    ratio, left, top = transform
    boxes[:, 0] = (boxes[:, 0] - left) / ratio
    boxes[:, 1] = (boxes[:, 1] - top) / ratio
    boxes[:, 2:4] /= ratio
    if dedup:
        keep = TorchNMS.fast_nms(boxes, scores, args.nms, iou_func=batch_probiou)[:args.max_det]
    else:
        keep = torch.arange(len(boxes), device=boxes.device)[:args.max_det]
    return boxes[keep].cpu(), scores[keep].cpu(), classes[keep].cpu(), indices[keep].cpu()


def match_boxes(left, right, threshold):
    """优先对齐重叠最高的旋转框，类别不参与匹配。"""
    if not len(left) or not len(right):
        return [], list(range(len(left))), list(range(len(right)))
    similarity = batch_probiou(left, right)
    pairs = torch.nonzero(similarity >= threshold, as_tuple=False)
    values = similarity[pairs[:, 0], pairs[:, 1]]
    used_left, used_right, matches = set(), set(), []
    for pos in torch.argsort(values, descending=True, stable=True).tolist():
        i, j = pairs[pos].tolist()
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            matches.append((i, j, float(values[pos])))
    return matches, sorted(set(range(len(left))) - used_left), sorted(set(range(len(right))) - used_right)


def as_rows(candidates):
    boxes, scores, classes, anchors = candidates
    return [{"xywhr": box, "score": score, "class_id": cls, "position": loc}
            for box, score, cls, loc in zip(boxes.tolist(), scores.tolist(), classes.tolist(), anchors.tolist())]


def preview(image, candidates, out):
    for name, values in candidates.items():
        canvas = image.copy()
        boxes, scores, classes, _ = values
        points = xywhr2xyxyxyxy(boxes).round().to(torch.int32).numpy()
        for poly, score, cls in zip(points, scores, classes):
            cv2.polylines(canvas, [poly], True, (80, 245, 180), 2)
            pos = tuple(map(int, poly[0]))
            cv2.putText(canvas, f"{int(cls)}:{float(score):.2f}", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 245, 180), 1)
        if not cv2.imwrite(str(out.with_name(f"{out.stem}_{name}.jpg")), canvas):
            raise OSError("无法保存预览图")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--match", type=float, default=.30)
    parser.add_argument("--localization", type=float, default=.65)
    parser.add_argument("--nms", type=float, default=.65)
    parser.add_argument("--pre-k", type=int, default=600)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()
    if (__version__ != "8.4.115" or args.imgsz < 32 or args.imgsz % 32 or args.count < 1
            or not 0 < args.conf < 1 or not 0 < args.match <= args.localization <= 1
            or not 0 < args.nms < 1 or not 1 <= args.max_det <= args.pre_k):
        parser.error("版本、尺寸或阈值不符合本次校准约束")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    device = torch.device(args.device)
    paths = validation_examples(args.data_root.resolve(strict=True), args.count, args.seed)
    weights = args.weights.resolve(strict=True)
    model, _ = load_checkpoint(str(weights), device=device, fuse=False)
    head = model.model[-1]
    if not isinstance(head, OBB26) or not head.end2end or head.cv2 is None or head.cv4 is None or head.nc != 7:
        raise ValueError("要求七类且未融合的 YOLO26 OBB 双头权重")
    head.export = False
    args.out.mkdir(parents=True, exist_ok=False)
    totals = Counter({k: 0 for k in ("class_conflict", "localization", "one2one_only", "one2many_only")})
    counts = Counter()
    rank = []
    with torch.inference_mode(), (args.out / "details.jsonl").open("w", encoding="utf-8") as fp:
        for i, (path, day) in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"不能读取 {path.name}")
            x, transform = input_tensor(image, args.imgsz, device)
            _, raw = model(x)
            one = decoded_candidates(head, raw["one2one"], transform, args, dedup=False)
            many = decoded_candidates(head, raw["one2many"], transform, args, dedup=True)
            matches, left, right = match_boxes(one[0], many[0], args.match)
            events = Counter(one2one_only=len(left), one2many_only=len(right))
            joined = []
            for a, b, similarity in matches:
                conflict = bool(one[2][a] != many[2][b])
                events["class_conflict"] += conflict
                events["localization"] += similarity < args.localization
                joined.append({"one2one_index": a, "one2many_index": b, "probiou": similarity,
                               "class_conflict": conflict})
            details = dict(image=path.name, capture_date=day, events=dict(events),
                           one2one=as_rows(one), one2many=as_rows(many), matched=joined,
                           one2one_only=left, one2many_only=right)
            fp.write(json.dumps(details, allow_nan=False) + "\n")
            totals.update(events)
            counts.update(one2one=len(one[0]), one2many=len(many[0]), matched=len(joined))
            rank.append({"image": path.name, "capture_date": day, **events})
            if i < 3:
                preview(image, {"one2one": one, "one2many": many}, args.out / path.name)
    fields = ["image", "capture_date", "class_conflict", "localization", "one2one_only", "one2many_only"]
    with (args.out / "frame_events.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval=0)
        writer.writeheader()
        writer.writerows(rank)
    summary = dict(images=len(paths), config=vars(args), candidate_counts=dict(counts),
                   provisional_events=dict(totals),
                   weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                   experiment_split="V", labels_read=False, ultralytics=__version__)
    result = json.dumps(summary, indent=2, default=str, ensure_ascii=False)
    (args.out / "summary.json").write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
