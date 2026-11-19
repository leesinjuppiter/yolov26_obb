"""Build and verify a capture-group-disjoint full-data YOLO OBB split."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

SPLITS = ("train", "val", "test")
RATIOS = np.array([0.6, 0.2, 0.2])


def label_counts(data, class_count):
    counts = np.zeros(class_count, dtype=int)
    for line in data.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        cls, *coordinates = line.split()
        cls = int(cls)
        if not 0 <= cls < class_count or len(coordinates) != 8:
            raise ValueError("invalid YOLO OBB class or column count")
        if any(not (0 <= float(value) <= 1) for value in coordinates):
            raise ValueError("invalid normalized OBB coordinate")
        counts[cls] += 1
    return counts


def get_plan(groups, class_count):
    if not 0 < len(groups) < 63:
        raise ValueError("unexpected number of capture groups")
    sizes = [len(rows) for rows in groups.values()]
    image_targets = (sum(sizes) * RATIOS).round().astype(int)
    if image_targets.sum() != sum(sizes) or image_targets[1] != image_targets[2]:
        raise ValueError("unexpected source count for the exact 6:2:2 split")
    heldout = int(image_targets[1])
    masks = []

    def collect(start, remaining, mask):
        if remaining == 0:
            masks.append(mask)
            return
        for index in range(start, len(sizes)):
            if sizes[index] <= remaining:
                collect(index + 1, remaining - sizes[index], mask | (1 << index))

    collect(0, heldout, 0)
    if not masks:
        raise ValueError("no group assignment matches the held-out image count")
    masks = np.array(masks, dtype=np.int64)
    bits = 1 << np.arange(len(sizes), dtype=np.int64)
    class_images = np.array([np.count_nonzero([row[1] for row in rows], axis=0)
                             for rows in groups.values()]).reshape(len(groups), class_count)
    total = class_images.sum(axis=0)
    balance = (np.bitwise_and(masks[:, None], bits) != 0) @ class_images

    def loss(counts, ratio):
        deviation = (np.abs(counts - ratio * total) / np.maximum(total, 1)).sum(axis=-1)
        return deviation + 3 * np.count_nonzero((counts == 0) & (total > 0), axis=-1)

    evaluation_loss = loss(balance, RATIOS[1])
    best, chosen = np.inf, None
    for index, val in enumerate(masks):
        disjoint = np.flatnonzero((masks & val) == 0)
        if not len(disjoint):
            continue
        train = total - balance[index] - balance[disjoint]
        costs = evaluation_loss[index] + evaluation_loss[disjoint] + loss(train, RATIOS[0])
        best_index = int(np.argmin(costs))
        if costs[best_index] < best:
            best, chosen = costs[best_index], (int(val), int(masks[disjoint[best_index]]))
    if chosen is None:
        raise ValueError("no disjoint capture groups achieve exact 6:2:2")
    val, test = chosen
    assignment = {key: (1 if val & int(bit) else 2 if test & int(bit) else 0)
                  for key, bit in zip(groups, bits)}
    return assignment, image_targets.tolist(), "Exact exhaustive search, with class coverage and balance"


def prepare(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    manifest = source / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sample_ids = {row["sample_id"] for row in rows}
    if len(rows) != 4911 or len(sample_ids) != len(rows):
        raise ValueError("expected 4,911 unique source samples")
    names = yaml.safe_load((source / "dataset4911_yolo26obb_al.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    groups, seen_hashes = defaultdict(list), {}
    for row in rows:
        group = row["capture_group"]
        digest = row["image_sha256"]
        if not group or not digest or seen_hashes.setdefault(digest, group) != group:
            raise ValueError("duplicate images span multiple source capture groups")
        labels = (source / row["label"]).read_bytes()
        counts = label_counts(labels, len(names))
        if counts.sum() != int(row["objects"]):
            raise ValueError(f"label count mismatch: {row['sample_id']}")
        groups[group].append((row, counts))
    groups = dict(sorted(groups.items()))
    assignment, expected, solver = get_plan(groups, len(names))
    print(f"Split assignment ready: {solver}", flush=True)
    renamed, summary = [], {}
    with TemporaryDirectory(prefix=".dataset-full-", dir=destination.parent) as directory:
        stage = Path(directory) / "dataset"
        for split_code, split in enumerate(SPLITS):
            keys = [key for key in groups if assignment[key] == split_code]
            samples = sorted((item for key in keys for item in groups[key]),
                             key=lambda item: (item[0]["capture_date"], item[0]["source_image"]))
            for kind in ("images", "labels"):
                (stage / kind / split).mkdir(parents=True)
            for index, (row, _) in enumerate(samples, 1):
                name = f"d4911_full_{split}_{index:06d}"
                image = f"images/{split}/{name}.jpg"
                label = f"labels/{split}/{name}.txt"
                for field, target in (("image", image), ("label", label)):
                    data = (source / row[field]).read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if field == "image" and digest != row["image_sha256"]:
                        raise ValueError(f"image checksum mismatch: {row['sample_id']}")
                    (stage / target).write_bytes(data)
                    if hashlib.sha256((stage / target).read_bytes()).hexdigest() != digest:
                        raise IOError(f"copy checksum mismatch: {target}")
                    if field == "label":
                        label_digest = digest
                renamed.append(dict(new_split=split, new_image=image, new_label=label,
                                    label_sha256=label_digest, **row))
            matrix = np.array([counts for _, counts in samples])
            summary[split] = dict(images=len(samples), groups=len(keys), negative_images=int(np.sum(matrix.sum(axis=1) == 0)),
                                  boxes=int(matrix.sum()), class_boxes=dict(zip(names, matrix.sum(axis=0).tolist())),
                                  class_images=dict(zip(names, np.count_nonzero(matrix, axis=0).tolist())))
        if len(renamed) != len(rows):
            raise RuntimeError("not all rows were assigned")
        with (stage / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=renamed[0])
            writer.writeheader()
            writer.writerows(renamed)
        config = dict(path=str(destination), **{kind: f"images/{kind}" for kind in SPLITS}, names=names)
        (stage / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report = dict(source=str(source), source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      targets=dict(zip(SPLITS, expected)), groups=len(groups), duplicate_cross_split=0,
                      checksums_verified=len(renamed) * 2, splits=summary, solver=solver)
        (stage / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.destination.resolve())
