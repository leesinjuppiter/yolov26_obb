"""Join W0 V-set divergence events with the existing geometric label associations."""
import argparse
from collections import Counter
import csv
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    events = read_csv(args.review_folder / "review_events.csv")
    if not events:
        parser.error("review_events.csv contains no divergence events")
    rows = read_csv(args.review_folder / "frame_box_review.csv")
    predictions = {(row["image"], row["head"], row["pred_id"]): row for row in rows if row["pred_id"]}
    targets = {(row["image"], row["head"], row["gt_id"]): row for row in rows if row["gt_id"]}
    result = []
    for event in events:
        image, head, box = (event[key] for key in ("原图文件名", "预测头", "框编号"))
        other = "one2many" if head == "one2one" else "one2one"
        row = predictions[image, head, box]
        counterpart = event["另一头框编号"]
        competing = predictions[image, other, counterpart] if counterpart else {}
        if row["gt_id"] and not competing:
            competing = targets.get((image, other, row["gt_id"]), {})
        class_name = event["类别"]
        categories = {class_name, row["gt_class"], competing.get("gt_class"), event["另一头类别"]} - {"coal", "", None}
        association = row["status"]
        opposite = competing.get("status", "")
        if association == "correct" and opposite == "missed_gt":
            hint = f"核对 {other} 是否漏检，并检查另一头是否有同目标候选"
        elif association == "unmatched_prediction":
            hint = "查看附近有无同一目标，区分重复框、误检或原标签漏标"
        elif association == "correct" and opposite == "correct":
            hint = "两个头均关联到已有标注，检查匹配分歧或重复框"
        else:
            hint = "核对原图、原框与两路预测的类别和位置"
        result.append(dict(原图文件名=image, 审核顺序=int(event["顺序"]), 分歧类型=event["分歧类型"],
                           预测头=head, 框编号=box, 类别=class_name, 分数=event["分数"],
                           参考标注=row["gt_id"], 当前头几何关联=association, 另一头几何关联=opposite,
                           异物类别="、".join(sorted(categories)), 复核提示=hint, 人工判定="", 备注=""))
    result.sort(key=lambda row: (row["审核顺序"], not row["异物类别"], -float(row["分数"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0])
        writer.writeheader()
        writer.writerows(result)
    focus = [row for row in result if row["异物类别"]]
    print(f"{len(result)} events; {len(focus)} foreign-object events across {len({r['原图文件名'] for r in focus})} images")
    print("Foreign-object geometric associations:", dict(Counter(row["当前头几何关联"] for row in focus)))
    print("First six images:", ", ".join(dict.fromkeys(row["原图文件名"] for row in result if row["审核顺序"] <= 6)))


if __name__ == "__main__":
    main()
