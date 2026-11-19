"""Check a review export against the submitted YOLO ZIP before server-side use."""

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def paths(archive, prefix):
    return {name for name in archive.namelist() if name.startswith(prefix + "/") and not name.endswith("/")}


def inspect_labels(archive, labels, class_count):
    total = 0
    for name in sorted(labels):
        for number, row in enumerate(archive.read(name).decode("utf-8-sig").splitlines(), 1):
            if not row.strip():
                continue
            tokens = row.split()
            try:
                if len(tokens) != 9 or not 0 <= int(tokens[0]) < class_count:
                    raise ValueError("invalid class or coordinate count")
                coords = [float(token) for token in tokens[1:]]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
                    raise ValueError("coordinates out of range")
            except ValueError as error:
                raise ValueError(f"invalid OBB label at {name}:{number}") from error
            total += 1
    return total


def check(original, result, expected_split):
    entries = result.namelist()
    if len(entries) != len({name.rstrip("/").casefold() for name in entries}) or any(
        "\\" in name or ":" in name or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts or name.rstrip("/") != PurePosixPath(name).as_posix()
        for name in entries
    ):
        raise ValueError("unsafe or duplicate export archive paths")
    yaml = json.loads(result.read("data.yaml"))
    classes = yaml["names"]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("invalid class order")
    from archive_io import dataset_index
    source_classes, source_entries, _ = dataset_index(original)
    if classes != source_classes:
        raise ValueError("class order changed")
    mapping = {f"images/{split}/{PurePosixPath(image).name}": (image, label)
               for image, label, split in source_entries}
    images = paths(result, "images")
    if not images or images != set(mapping):
        raise ValueError("image names or split paths differ from the original task")
    for name in images:
        if result.read(name) != original.read(mapping[name][0]):
            raise ValueError(f"image bytes differ: {name}")
    labels = paths(result, "labels")
    expected_labels = {
        "labels/" + str(PurePosixPath(name).relative_to("images").with_suffix(".txt")): mapping[name][1]
        for name in images
    }
    if not set(expected_labels).issuperset(labels):
        raise ValueError("labels without an associated image")
    boxes = inspect_labels(result, labels, len(classes))

    frames = list(csv.DictReader(io.StringIO(result.read("review/frames.csv").decode("utf-8-sig"))))
    events = list(csv.DictReader(io.StringIO(result.read("review/events.csv").decode("utf-8-sig"))))
    changes = [json.loads(line) for line in result.read("review/changes.jsonl").splitlines() if line.strip()]
    identities = {(row["split"], row["image"]) for row in frames}
    expected_frames = {(PurePosixPath(image).parts[1], PurePosixPath(image).name) for image in images}
    if len(frames) != len(images) or identities != expected_frames:
        raise ValueError("review frame manifest does not cover the exported images")
    statuses = {"pending", "in_progress", "done", "negative", "uncertain"}
    if any(row["status"] not in statuses for row in frames):
        raise ValueError("unexpected frame review status")
    source = json.loads(result.read("review/source.json")) if "review/source.json" in entries else {}
    if "review_source.json" in original.namelist() and source != json.loads(original.read("review_source.json")):
        raise ValueError("GPU source provenance changed")
    if expected_split and source.get("experiment_split") != expected_split:
        raise ValueError("required GPU source split missing or different")
    changed_labels = sorted(name for name, old in expected_labels.items()
                            if (result.read(name) if name in labels else None)
                            != (original.read(old) if old else None))
    return dict(images=len(images), boxes=boxes, events=len(events), audit_entries=len(changes),
                statuses={status: sum(row["status"] == status for row in frames) for status in sorted(statuses)},
                changed_label_files=changed_labels, source=source)


def save_snapshot(result, destination):
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".review-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        staged.mkdir()
        for name in result.namelist():
            path = PurePosixPath(name)
            if name != "data.yaml" and path.parts[0] not in {"images", "labels", "review"}:
                raise ValueError(f"unexpected result entry: {name}")
            if name.endswith("/"):
                continue
            target = staged.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result.read(name))
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="ZIP originally sent to the reviewer")
    parser.add_argument("result", type=Path, help="ZIP exported by the local workbench")
    parser.add_argument("--expected-split", choices=("V", "U", "T"), help="require source split provenance")
    parser.add_argument("--unpack", type=Path, help="create a new directory from the verified export")
    args = parser.parse_args()
    with ZipFile(args.original) as original, ZipFile(args.result) as result:
        summary = check(original, result, args.expected_split)
        if args.unpack:
            save_snapshot(result, args.unpack)
            summary["snapshot"] = str(args.unpack)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
