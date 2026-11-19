"""Read portable YOLO OBB review packages without extracting arbitrary paths."""

import ast
import csv
import io
import json
import math
import re
import struct
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

MAX_UPLOAD = 256 * 1024 * 1024
MAX_ARCHIVE = 512 * 1024 * 1024
SPLITS = ("train", "val", "test")
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".bmp": "image/bmp"}
HEADS = ("one2one", "one2many")


def open_zip(source):
    try:
        archive = ZipFile(source)
    except BadZipFile as exc:
        raise ValueError("无法读取 ZIP 文件") from exc
    try:
        members = archive.infolist()
        names = set()
        expanded = 0
        for item in members:
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or not path.parts or ".." in path.parts
                    or "\\" in item.filename or ":" in item.filename or item.flag_bits & 1):
                raise ValueError("ZIP 中存在不支持的路径或加密文件")
            if item.is_dir():
                continue
            key = path.as_posix()
            if key.casefold() in names or key != item.filename:
                raise ValueError("ZIP 中存在重复或不规范的文件路径")
            names.add(key.casefold())
            expanded += item.file_size
        if len(members) > 20_000 or expanded > MAX_ARCHIVE:
            raise ValueError("ZIP 解压后的大小或文件数量超出限制")
        return archive
    except Exception:
        archive.close()
        raise


def read(archive, path, limit=32 * 1024 * 1024):
    entry = archive.getinfo(path)
    if entry.file_size > limit:
        raise ValueError(f"文件过大：{path}")
    return archive.read(entry)


def load_text(archive, path, limit=32 * 1024 * 1024):
    return read(archive, path, limit).decode("utf-8-sig")


def load_config(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Standard library only. Support conventional scalar paths and names as an
    # inline/block list or integer-keyed mapping, as emitted by YOLO tooling.
    result, section = {}, ""
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0]
        if section in (*SPLITS, "names") and line.startswith("- "):
            result.setdefault(section, []).append(unquote(line[2:]))
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError("data.yaml 包含不支持的写法")
        key, value = key.strip(), value.strip()
        if section == "names" and key.strip("'\"").isdigit():
            result.setdefault("names", {})[int(key.strip("'\""))] = unquote(value)
            continue
        section = key
        if key not in (*SPLITS, "names"):
            continue
        if value.startswith("[") and value.endswith("]"):
            result[key] = [unquote(part) for part in next(csv.reader([value[1:-1]], skipinitialspace=True))]
        elif value.startswith("{") and value.endswith("}") and key == "names":
            result[key] = {int(k.strip().strip("'\"")): unquote(v)
                           for item in value[1:-1].split(",") for k, v in [item.split(":", 1)]}
        elif value:
            result[key] = unquote(value)
    return result


def unquote(value):
    value = value.strip()
    if value.startswith(("'", '"')):
        return str(ast.literal_eval(value))
    return value


def image_size(data):
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"BM" and len(data) >= 26:
        width, height = struct.unpack_from("<ii", data, 18)
        return width, abs(height)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        if data[12:16] == b"VP8X" and len(data) >= 30:
            return tuple(int.from_bytes(data[offset:offset + 3], "little") + 1 for offset in (24, 27))
        if data[12:16] == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
            value = int.from_bytes(data[21:25], "little")
            return (value & 0x3FFF) + 1, ((value >> 14) & 0x3FFF) + 1
        if data[12:16] == b"VP8 " and data[23:26] == b"\x9d\x01\x2a" and len(data) >= 30:
            return tuple(value & 0x3FFF for value in struct.unpack_from("<HH", data, 26))
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 4 <= len(data) and data[offset] == 0xFF:
            code = data[offset + 1]
            if code == 0xFF:
                offset += 1
                continue
            if code in (0xDA, 0xD9):
                break
            size = int.from_bytes(data[offset + 2:offset + 4], "big")
            if code in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF) and offset + 9 <= len(data):
                height, width = struct.unpack_from(">HH", data, offset + 5)
                return width, height
            if size < 2:
                break
            offset += 2 + size
    raise ValueError("图片格式或尺寸无法识别")


def validate_box(box, count):
    class_id, points = box.get("class_id"), box.get("points")
    if type(class_id) is not int or not 0 <= class_id < count:
        raise ValueError("旋转框类别超出范围")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("旋转框需要四个顶点")
    if any(not isinstance(point, list) or len(point) != 2
           or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in point)
           for point in points):
        raise ValueError("旋转框顶点必须在原图范围内")
    area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
    if len(set(map(tuple, points))) < 3 or abs(area) <= 1e-14:
        raise ValueError("旋转框面积须大于零")


def parse_labels(raw, count, path):
    boxes = []
    for index, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tokens = line.split()
            if len(tokens) != 9:
                raise ValueError("OBB 标签要求类别编号加八列坐标")
            xy = list(map(float, tokens[1:]))
            box = {"id": f"g{index}", "class_id": int(tokens[0]),
                   "points": [[x, y] for x, y in zip(xy[::2], xy[1::2])]}
            validate_box(box, count)
        except ValueError as exc:
            raise ValueError(f"{path} 第 {index} 行：{exc}") from exc
        boxes.append(box)
    return boxes


def dataset_index(archive):
    files = set(archive.namelist())
    configs = [n for n in files if PurePosixPath(n).name.lower() in ("data.yaml", "dataset.yaml", "data.yml")]
    if len(configs) != 1:
        raise ValueError("数据集 ZIP 需包含一份 data.yaml 或 dataset.yaml")
    config_path = configs[0]
    base = PurePosixPath(config_path).parent
    config = load_config(load_text(archive, config_path, 1024 * 1024))
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
        if set(names) != set(range(len(names))):
            raise ValueError("YOLO 类别编号须从零开始连续排列")
        names = [names[i] for i in range(len(names))]
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name.strip()
            for name in names) or len(names) != len(set(names))):
        raise ValueError("data.yaml 需定义不重复的类别")
    selected = {}
    candidate_images = sorted(f for f in files if PurePosixPath(f).suffix.lower() in IMAGE_TYPES)
    for split in SPLITS:
        sources = config.get(split) or [f"images/{split}", f"{split}/images"]
        if isinstance(sources, str):
            sources = [sources]
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise ValueError(f"{split} 须定义图片目录或清单")
        for source in sources:
            relative = PurePosixPath(source)
            if relative.is_absolute() or ".." in relative.parts or "\\" in source or ":" in source:
                raise ValueError(f"分集路径必须位于压缩包内：{source}")
            path = (base / relative).as_posix()
            if path.lower().endswith(".txt"):
                if path not in files:
                    raise ValueError(f"分集清单不存在：{path}")
                candidates = []
                for line in load_text(archive, path).splitlines():
                    if not line.strip():
                        continue
                    ref = PurePosixPath(line.strip())
                    options = [(PurePosixPath(path).parent / ref).as_posix(), (base / ref).as_posix()]
                    match = next((p for p in options if p in candidate_images), None)
                    if match is None:
                        raise ValueError(f"清单中的图片不存在：{line}")
                    candidates.append(match)
            else:
                candidates = [f for f in candidate_images if f.startswith(path.rstrip("/") + "/")]
            for image in candidates:
                if image in selected and selected[image] != split:
                    raise ValueError(f"图片同时出现在多个分集：{image}")
                selected[image] = split
    if not selected:
        raise ValueError("没有找到 train、val 或 test 图片")
    entries = []
    seen = set()
    for image, split in selected.items():
        path = PurePosixPath(image)
        output_key = (split, path.stem.casefold())
        if output_key in seen:
            raise ValueError(f"同一个分集出现重名图片：{path.name}")
        seen.add(output_key)
        parts = list(path.parts)
        if "images" not in parts:
            raise ValueError(f"YOLO 图片须位于 images 目录：{image}")
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        label = PurePosixPath(*parts).with_suffix(".txt").as_posix()
        entries.append((image, label if label in files else None, split))
    return names, entries, base


def optional_file(archive, base, name):
    for path in dict.fromkeys(((base / name).as_posix(), name)):
        if path in archive.NameToInfo:
            return path
    return None


def review_metadata(archive, base):
    def csv_rows(name):
        path = optional_file(archive, base, name)
        return list(csv.DictReader(io.StringIO(load_text(archive, path)))) if path else []

    frames = {row["原图文件名"]: index for index, row in enumerate(csv_rows("review_frames.csv"), 1)}
    events = {}
    for row in csv_rows("review_events.csv"):
        image = row["原图文件名"]
        bucket = events.setdefault(image, [])
        bucket.append(event(len(bucket) + 1, row["分歧类型"], row["预测头"], row["框编号"],
                            row.get("类别"), row.get("分数"), row.get("另一头框编号"),
                            row.get("另一头类别"), row.get("另一头分数"), row.get("ProbIoU")))
    details_path = optional_file(archive, base, "details.jsonl")
    details = {}
    if details_path:
        for line in load_text(archive, details_path).splitlines():
            if line.strip():
                record = json.loads(line)
                image = record["image"]
                if image in details:
                    raise ValueError(f"双头预测中出现重复图片：{image}")
                details[image] = record
    source_path = optional_file(archive, base, "review_source.json")
    source = json.loads(load_text(archive, source_path, 1024 * 1024)) if source_path else None
    if source is not None and not isinstance(source, dict):
        raise ValueError("review_source.json 应为 JSON 对象")
    return frames, events, details, source


def event(index, kind, head, box_id, cls, score, other=None, other_cls=None, other_score=None, iou=None):
    def number(value):
        value = float(value) if value not in ("", None) else None
        if value is not None and not math.isfinite(value):
            raise ValueError("审核事件分数无效")
        return value
    return {"id": f"e{index}", "kind": kind, "head": head, "box_id": box_id,
            "class_name": cls or "", "score": number(score), "other_box_id": other or "",
            "other_class_name": other_cls or "", "other_score": number(other_score),
            "probiou": number(iou), "decision": "", "note": ""}


def predictions(record, width, height, classes):
    result = {}
    for head in HEADS:
        boxes = []
        for index, raw in enumerate(record.get(head, []), 1):
            cx, cy, w, h, angle = raw["xywhr"]
            score, class_id = raw["score"], raw["class_id"]
            if not all(type(n) in (int, float) and math.isfinite(n) for n in (cx, cy, w, h, angle, score)):
                raise ValueError("预测框包含无效坐标")
            if type(class_id) is not int or not 0 <= class_id < len(classes) or w <= 0 or h <= 0:
                raise ValueError("预测框尺寸或类别无效")
            co, si = math.cos(angle), math.sin(angle)
            points = [[(cx+x*co-y*si)/width, (cy+x*si+y*co)/height]
                      for x, y in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]]
            boxes.append({"id": f"P{index}", "class_id": class_id, "points": points, "score": score})
        result[head] = boxes
    return result


def derived_events(record, preds, classes):
    rows = []
    for head in HEADS:
        for index in record.get(head + "_only", []):
            box = preds[head][index]
            rows.append(event(len(rows)+1, head + "_only", head, box["id"], classes[box["class_id"]], box["score"]))
    for pair in record.get("matched", []):
        iou = pair["probiou"]
        if not pair["class_conflict"] and iou >= 0.65:
            continue
        left = preds["one2one"][pair["one2one_index"]]
        right = preds["one2many"][pair["one2many_index"]]
        kind = "class_conflict" if pair["class_conflict"] else "localization"
        rows.append(event(len(rows)+1, kind, "one2one", left["id"], classes[left["class_id"]], left["score"],
                          right["id"], classes[right["class_id"]], right["score"], iou))
    return rows
