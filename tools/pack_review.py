"""Combine a GPU-produced YOLO ZIP and optional review artifacts."""

import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


def copy_entry(source, destination, name):
    with source.open(name) as incoming, destination.open(name, "w") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="YOLO OBB ZIP")
    parser.add_argument("output", type=Path, help="new review task ZIP")
    parser.add_argument("--review", type=Path, help="legacy review ZIP or extracted folder")
    parser.add_argument("--predictions", type=Path, help="GPU details.jsonl or details.jsonl.gz")
    parser.add_argument("--summary", type=Path, help="GPU inference summary.json")
    parser.add_argument("--run-id", help="GPU run identifier")
    parser.add_argument("--split", choices=("V", "U", "T"), help="source experiment split")
    args = parser.parse_args()

    source_info = {}
    if args.summary:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        for key in ("weights_sha256", "experiment_split", "ultralytics"):
            if key in summary:
                source_info[key] = summary[key]
    if args.split:
        if source_info.get("experiment_split", args.split) != args.split:
            parser.error("provided split differs from GPU summary")
        source_info["experiment_split"] = args.split
    if args.run_id:
        source_info["run_id"] = args.run_id

    with ExitStack() as stack:
        dataset = stack.enter_context(ZipFile(args.dataset))
        names = set(dataset.namelist())
        if "data.yaml" not in names:
            parser.error("dataset ZIP must contain data.yaml at its root")
        review = args.review
        sidecar = stack.enter_context(ZipFile(review)) if review and review.is_file() else None
        if review and not (sidecar or review.is_dir()):
            parser.error("review path must be a ZIP or a directory")
        review_names = sidecar.namelist() if sidecar else (
            [path.relative_to(review).as_posix() for path in review.rglob("*") if path.is_file()]
            if review else []
        )
        companions = [
            name for name in review_names
            if name in {"review_frames.csv", "review_events.csv", "frame_box_review.csv", "details.jsonl"}
            or (name.startswith("previews/") and len(PurePosixPath(name).parts) == 2
                and PurePosixPath(name).suffix.lower() in {".jpg", ".png", ".jpeg"})
        ]
        if any(name in names or ".." in PurePosixPath(name).parts for name in companions):
            parser.error("duplicate or invalid review archive entry")
        if args.predictions and ("details.jsonl" in names or "details.jsonl" in companions):
            parser.error("predictions are already present in an input archive")
        if source_info and "review_source.json" in names:
            parser.error("provenance is already present in the dataset archive")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as output:
            for name in dataset.namelist():
                copy_entry(dataset, output, name)
            for name in companions:
                if sidecar:
                    copy_entry(sidecar, output, name)
                else:
                    output.write(review / name, name)
            if args.predictions:
                reader = gzip.open if args.predictions.suffix == ".gz" else open
                with reader(args.predictions, "rb") as incoming, output.open("details.jsonl", "w") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            if source_info:
                output.writestr("review_source.json", json.dumps(source_info, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
