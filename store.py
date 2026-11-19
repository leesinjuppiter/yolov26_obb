"""Task and frame persistence with a versioned, transactional edit log."""

from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import sqlite3
from tempfile import SpooledTemporaryFile
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from archive_io import (
    IMAGE_TYPES, SPLITS, dataset_index, derived_events, image_size, open_zip,
    optional_file, parse_labels, predictions, read, review_metadata, validate_box,
)

STATUSES = ("pending", "in_progress", "done", "negative", "uncertain")
DONE = ("done", "negative")
FRAME_FIELDS = "id, name, split, status, width, height, version, note"


class Conflict(ValueError):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "review.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, fingerprint TEXT UNIQUE NOT NULL,
                    classes TEXT NOT NULL, source TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
                    name TEXT NOT NULL, split TEXT NOT NULL, width INTEGER NOT NULL,
                    height INTEGER NOT NULL, content BLOB NOT NULL, original_label BLOB,
                    original_boxes TEXT NOT NULL, boxes TEXT NOT NULL, preview BLOB,
                    review_order INTEGER, predictions TEXT NOT NULL DEFAULT '{}',
                    events TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'pending',
                    note TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (task_id, split, name)
                );
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY, frame_id TEXT NOT NULL REFERENCES frames(id),
                    version INTEGER NOT NULL, created_at TEXT NOT NULL, changes TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS frames_task ON frames(task_id);
                CREATE INDEX IF NOT EXISTS history_frame ON history(frame_id, id);
            """)

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.database, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        finally:
            db.close()

    def import_task(self, stream, name):
        with open_zip(stream) as archive:
            classes, entries, base = dataset_index(archive)
            fingerprint = hashlib.sha256(dumps(classes).encode())
            for image_path, label_path, split in sorted(entries, key=lambda row: (row[2], PurePosixPath(row[0]).name)):
                fingerprint.update(dumps((split, PurePosixPath(image_path).name)).encode())
                fingerprint.update(hashlib.sha256(read(archive, image_path)).digest())
                fingerprint.update(hashlib.sha256(read(archive, label_path, 2 * 1024**2)).digest()
                                   if label_path else b"\0")
            digest = fingerprint.hexdigest()
            with self.connect(write=True) as db:
                found = db.execute("SELECT id FROM tasks WHERE fingerprint=?", (digest,)).fetchone()
                task_id = found["id"] if found else uuid4().hex
                if not found:
                    db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                               (task_id, name, digest, dumps(classes), None, now()))
                    for image_path, label_path, split in entries:
                        image = read(archive, image_path)
                        width, height = image_size(image)
                        if width <= 0 or height <= 0 or width * height > 40_000_000:
                            raise ValueError(f"无效的图片尺寸：{image_path}")
                        raw = read(archive, label_path, 2 * 1024**2) if label_path else None
                        boxes = parse_labels(raw or b"", len(classes), label_path or image_path)
                        db.execute("""INSERT INTO frames
                                    (id,task_id,name,split,width,height,content,original_label,original_boxes,boxes)
                                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                                   (uuid4().hex, task_id, PurePosixPath(image_path).name, split, width,
                                    height, image, raw, dumps(boxes), dumps(boxes)))
                self.merge_review(db, task_id, archive, base, classes)
        return {"id": task_id, "existing": found is not None}

    def import_review(self, task_id, stream):
        with open_zip(stream) as archive, self.connect(write=True) as db:
            task = self.find_task(db, task_id)
            merged = self.merge_review(db, task_id, archive, PurePosixPath("."), json.loads(task["classes"]))
            if not merged:
                raise ValueError("审核 ZIP 中没有可识别的事件、预测或对照图")
        return {"id": task_id, "images": merged}

    def merge_review(self, db, task_id, archive, base, classes):
        order, events, details, source = review_metadata(archive, base)
        if source is not None:
            existing = self.find_task(db, task_id)["source"]
            if existing and json.loads(existing) != source:
                raise Conflict("此任务已关联不同的服务器运行记录")
            db.execute("UPDATE tasks SET source=? WHERE id=?", (dumps(source), task_id))
        frames = db.execute("""SELECT id,name,width,height,events,predictions,review_order,
                              preview IS NOT NULL AS has_preview FROM frames WHERE task_id=?""",
                            (task_id,)).fetchall()
        names = {frame["name"] for frame in frames}
        referenced = set(order) | set(events) | set(details)
        if referenced - names:
            raise ValueError("审核包引用了当前数据集没有的图片")
        updated = 0
        for frame in frames:
            name = frame["name"]
            view = optional_file(archive, base, f"previews/{PurePosixPath(name).stem}_review.jpg")
            if name not in referenced and not view:
                continue
            if sum(other["name"] == name for other in frames) > 1:
                raise ValueError(f"多个分集存在同名图片，无法绑定审核记录：{name}")
            old_predictions = json.loads(frame["predictions"])
            preds = (predictions(details[name], frame["width"], frame["height"], classes)
                     if name in details else old_predictions)
            if old_predictions and preds != old_predictions:
                raise Conflict(f"图片已关联不同的预测记录：{name}")
            incoming = events.get(name)
            old_events = json.loads(frame["events"])
            if incoming is None and not old_events and name in details:
                incoming = derived_events(details[name], preds, classes)
            merged = merge_events(old_events, incoming, name)
            new_preview = read(archive, view) if view else None
            if new_preview and not new_preview.startswith(b"\xff\xd8"):
                raise ValueError(f"审核对照图不是 JPEG：{view}")
            has_changes = (merged != old_events or preds != old_predictions or
                           (name in order and frame["review_order"] != order[name]))
            db.execute("""UPDATE frames SET events=?,predictions=?,review_order=?,
                          preview=COALESCE(?,preview),version=version+? WHERE id=?""",
                       (dumps(merged), dumps(preds), order.get(name, frame["review_order"]),
                        new_preview, int(has_changes), frame["id"]))
            updated += 1
        return updated

    @staticmethod
    def find_task(db, task_id):
        task = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            raise KeyError("审核任务不存在")
        return task

    @staticmethod
    def find_frame(db, task_id, frame_id, fields="*"):
        frame = db.execute(f"SELECT {fields} FROM frames WHERE task_id=? AND id=?",
                           (task_id, frame_id)).fetchone()
        if frame is None:
            raise KeyError("图片不存在")
        return frame

    def list_tasks(self):
        with self.connect() as db:
            tasks = db.execute("""SELECT t.id,t.name,COUNT(f.id) total,
                                 SUM(f.status IN ('done','negative')) done FROM tasks t
                                 JOIN frames f ON f.task_id=t.id GROUP BY t.id
                                 ORDER BY t.created_at DESC""")
            return {"tasks": [dict(row) for row in tasks]}

    def task(self, task_id):
        with self.connect() as db:
            task = self.find_task(db, task_id)
            frames = db.execute(f"""SELECT {FRAME_FIELDS},json_array_length(events) event_count,
                                preview IS NOT NULL has_preview FROM frames WHERE task_id=?
                                ORDER BY review_order IS NULL,review_order,split,name""",
                                (task_id,))
            return {"id": task_id, "name": task["name"], "classes": json.loads(task["classes"]),
                    "source": json.loads(task["source"]) if task["source"] else None,
                    "images": [dict(row) for row in frames]}

    def frame(self, task_id, frame_id):
        with self.connect() as db:
            return self.frame_response(db, task_id, frame_id)

    def frame_response(self, db, task_id, frame_id):
        frame = dict(self.find_frame(db, task_id, frame_id,
                    FRAME_FIELDS + ",original_boxes,boxes,events,predictions,preview IS NOT NULL AS has_preview"))
        for field in ("original_boxes", "boxes", "events", "predictions"):
            frame[field] = json.loads(frame[field])
        logs = db.execute("SELECT version,created_at,changes FROM history WHERE frame_id=? ORDER BY id DESC",
                          (frame_id,))
        frame["history"] = [{"version": row["version"], "created_at": row["created_at"],
                             "changes": json.loads(row["changes"])} for row in logs]
        return frame

    def image_bytes(self, task_id, frame_id, preview=False):
        with self.connect() as db:
            field = "preview" if preview else "content"
            image = self.find_frame(db, task_id, frame_id, f"name,{field}")
            if image[field] is None:
                raise KeyError("此图片没有对照图")
            return image[field], "image/jpeg" if preview else IMAGE_TYPES[PurePosixPath(image["name"]).suffix.lower()]

    def export_summary(self, task_id):
        with self.connect() as db:
            self.find_task(db, task_id)
            row = db.execute("""SELECT COUNT(*) total, SUM(status IN ('done','negative')) done
                              FROM frames WHERE task_id=?""", (task_id,)).fetchone()
            operations = dict.fromkeys(("add", "delete", "modify"), 0)
            for key, count in db.execute("""SELECT json_extract(change.value,'$.kind'), COUNT(*)
                    FROM history h JOIN frames f ON h.frame_id=f.id JOIN json_each(h.changes) AS change
                    WHERE f.task_id=? GROUP BY 1""", (task_id,)):
                if key in operations:
                    operations[key] = count
            return {"total": row["total"], "done": row["done"], "remaining": row["total"] - row["done"],
                    "added": operations["add"], "deleted": operations["delete"], "modified": operations["modify"]}

    def save(self, task_id, frame_id, payload):
        if not isinstance(payload, dict):
            raise ValueError("保存内容应为一个 JSON 对象")
        with self.connect(write=True) as db:
            task = self.find_task(db, task_id)
            old = self.frame_response(db, task_id, frame_id)
            if type(payload.get("version")) is not int or payload["version"] != old["version"]:
                raise Conflict("此图已经更新，请刷新后再保存")
            boxes, status, note = parse_update(payload, old, len(json.loads(task["classes"])))
            old_events = [(event["id"], event["decision"], event["note"]) for event in old["events"]]
            events = update_events(old["events"], payload.get("events", []))
            changes = make_changes(old, boxes, events, status, note, old_events)
            if changes:
                version = old["version"] + 1
                db.execute("UPDATE frames SET boxes=?,events=?,status=?,note=?,version=? WHERE id=?",
                           (dumps(boxes), dumps(events), status, note, version, frame_id))
                db.execute("INSERT INTO history (frame_id,version,created_at,changes) VALUES (?,?,?,?)",
                           (frame_id, version, now(), dumps(changes)))
            return self.frame_response(db, task_id, frame_id)

    def export(self, task_id):
        stream = SpooledTemporaryFile(max_size=16 * 1024**2)
        try:
            with self.connect() as db, ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                db.execute("BEGIN")
                task = self.find_task(db, task_id)
                entries = db.execute("SELECT * FROM frames WHERE task_id=? ORDER BY split,name", (task_id,))
                list_frames, list_events, changes, splits = [], [], [], set()
                for frame in entries:
                    splits.add(frame["split"])
                    export_frame(db, archive, frame, list_frames, list_events, changes)
                config = {split: f"images/{split}" for split in SPLITS if split in splits}
                config["names"] = json.loads(task["classes"])
                metadata = {"schema_version": 1, "id": task_id, "name": task["name"],
                            "fingerprint": task["fingerprint"], "created_at": task["created_at"],
                            "exported_at": now(), "total": len(list_frames),
                            "done": sum(row["status"] in DONE for row in list_frames),
                            "splits": [split for split in SPLITS if split in splits]}
                archive.writestr("data.yaml", dumps(config) + "\n")
                archive.writestr("review/task.json", dumps(metadata) + "\n")
                if task["source"]:
                    archive.writestr("review/source.json", task["source"] + "\n")
                archive.writestr("review/frames.csv", csv_bytes(
                    ("image", "split", "status", "note", "original_boxes", "boxes"), list_frames))
                archive.writestr("review/events.csv", csv_bytes(
                    ("image", "split", "id", "kind", "head", "box_id", "class_name", "score",
                     "other_box_id", "other_class_name", "other_score", "probiou", "decision", "note"),
                    list_events))
                archive.writestr("review/changes.jsonl", "".join(dumps(row) + "\n" for row in changes))
            stream.seek(0)
            return stream
        except Exception:
            stream.close()
            raise


def event_key(event):
    return tuple(event[key] for key in ("kind", "head", "box_id", "other_box_id"))


def merge_events(old, incoming, name):
    if incoming is None or not old:
        return incoming if incoming is not None else old
    by_key = {event_key(row): row for row in incoming}
    if set(by_key) != {event_key(row) for row in old}:
        raise Conflict(f"图片已关联不同的审核事件：{name}")
    return [{**by_key[event_key(row)], "id": row["id"], "decision": row["decision"], "note": row["note"]}
            for row in old]


def string(value, label, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{label}最多 {limit} 字符")
    return value


def parse_update(payload, current, class_count):
    status = payload.get("status", current["status"])
    note = string(payload.get("note", current["note"]), "图像备注", 2000)
    boxes = payload.get("boxes", current["boxes"])
    if status not in STATUSES or not isinstance(boxes, list) or len(boxes) > 5000:
        raise ValueError("审核状态或旋转框列表无效")
    if status == "negative" and boxes:
        raise ValueError("确认无目标前请先删除当前图像的标注框")
    seen, output = set(), []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("旋转框记录无效")
        box_id = box.get("id")
        if not isinstance(box_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", box_id) or box_id in seen:
            raise ValueError("旋转框编号无效或重复")
        validate_box(box, class_count)
        output.append({key: box[key] for key in ("id", "class_id", "points")})
        seen.add(box_id)
    return output, status, note


def update_events(events, updates):
    if not isinstance(updates, list):
        raise ValueError("审核事件须为列表")
    by_id = {event["id"]: event for event in events}
    seen = set()
    for update in updates:
        if not isinstance(update, dict) or not isinstance(update.get("id"), str):
            raise ValueError("审核事件编号无效")
        event_id = update["id"]
        if event_id in seen or event_id not in by_id:
            raise ValueError("审核事件编号不存在或重复")
        event = by_id[event_id]
        event["decision"] = string(update.get("decision", event["decision"]), "事件判定", 100)
        event["note"] = string(update.get("note", event["note"]), "事件备注", 2000)
        seen.add(event_id)
    return events


def make_changes(current, boxes, events, status, note, old_events):
    changes = []
    old_boxes = {box["id"]: box for box in current["boxes"]}
    new_boxes = {box["id"]: box for box in boxes}
    for box_id in dict.fromkeys((*old_boxes, *new_boxes)):
        before, after = old_boxes.get(box_id), new_boxes.get(box_id)
        if before == after:
            continue
        kind = "add" if before is None else "delete" if after is None else "modify"
        changes.append({"kind": kind, "box_id": box_id, "before": before, "after": after})
    for event, (event_id, before_decision, before_note) in zip(events, old_events):
        before = {"decision": before_decision, "note": before_note}
        after = {key: event[key] for key in ("decision", "note")}
        if before != after:
            changes.append({"kind": "event", "event_id": event_id, "before": before, "after": after})
    for field, value in (("status", status), ("note", note)):
        if current[field] != value:
            changes.append({"kind": field, "before": current[field], "after": value})
    return changes


def export_frame(db, archive, frame, frames, events, changes):
    name, split = frame["name"], frame["split"]
    archive.writestr(f"images/{split}/{name}", frame["content"])
    stem = PurePosixPath(name).stem
    boxes, original = json.loads(frame["boxes"]), json.loads(frame["original_boxes"])
    label = export_labels(boxes, original, frame["original_label"], frame["status"])
    if label is not None:
        archive.writestr(f"labels/{split}/{stem}.txt", label)
    frame_keys = {"image": name, "split": split}
    frames.append({**frame_keys, "status": frame["status"], "note": frame["note"],
                   "original_boxes": len(original), "boxes": len(boxes)})
    events.extend({**frame_keys, **event} for event in json.loads(frame["events"]))
    for entry in db.execute("SELECT version,created_at,changes FROM history WHERE frame_id=? ORDER BY id",
                            (frame["id"],)):
        changes.extend({**frame_keys, "version": entry["version"], "created_at": entry["created_at"], **change}
                       for change in json.loads(entry["changes"]))


def export_labels(boxes, original, raw, status):
    if boxes == original and (raw is not None or status != "negative"):
        return raw
    original_rows = (raw or b"").splitlines(keepends=True)
    source = {box["id"]: box for box in original}
    rows = []
    for box in boxes:
        if source.get(box["id"]) == box:
            line = original_rows[int(box["id"][1:])-1]
            rows.append(line if line.endswith(b"\n") else line + b"\n")
            continue
        points = (value for point in box["points"] for value in point)
        rows.append((" ".join([str(box["class_id"]), *(format(v, ".10g") for v in points)])+"\n").encode())
    return b"".join(rows)


def csv_bytes(fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")
