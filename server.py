"""Loopback-only HTTP server for the standalone YOLO OBB reviewer."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import SpooledTemporaryFile
from urllib.parse import parse_qs, urlsplit
from zipfile import BadZipFile

from archive_io import MAX_UPLOAD
from store import Conflict, Store, dumps, string

ROOT = Path(__file__).resolve().parent
TASK = r"/api/tasks/(?P<task>[a-f0-9]{32})"
FRAME = TASK + r"/images/(?P<image>[a-f0-9]{32})"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8"}
STATIC = {
    "/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css",
    "/static/app.js": "app.js", "/static/styles.css": "styles.css",
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_PUT(self):
        self.dispatch("PUT")

    def dispatch(self, method):
        try:
            if not self.is_local():
                self.send_json({"error": "标注服务仅供本机访问"}, HTTPStatus.FORBIDDEN)
                return
            target = urlsplit(self.path)
            path = target.path
            if method == "GET":
                self.get_route(path)
                return
            if method == "POST":
                if path == "/api/import":
                    name = parse_qs(target.query).get("name", ["审核任务"])[0].strip()
                    name = string(name, "任务名称", 120) or "审核任务"
                    with self.zip_body() as stream:
                        result = self.server.store.import_task(stream, name)
                    self.send_json(result, HTTPStatus.OK if result["existing"] else HTTPStatus.CREATED)
                    return
                route = re.fullmatch(TASK + r"/review", path)
                if route:
                    with self.zip_body() as stream:
                        self.send_json(self.server.store.import_review(route["task"], stream))
                    return
            if method == "PUT":
                route = re.fullmatch(FRAME, path)
                if route:
                    if self.headers.get_content_type() != "application/json":
                        raise ValueError("保存内容须为 JSON")
                    payload = json.loads(self.rfile.read(self.content_length(16 * 1024**2)))
                    self.send_json(self.server.store.save(route["task"], route["image"], payload))
                    return
            raise KeyError("页面或接口不存在")
        except Conflict as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, UnicodeError, BadZipFile, IndexError, TypeError, SyntaxError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except KeyError as exc:
            self.send_json({"error": str(exc.args[0])}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.log_error("%s", exc)
            self.send_json({"error": "本地服务出现错误，请查看启动窗口。"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def get_route(self, path):
        service = self.server.store
        if path == "/api/health":
            self.send_json({"status": "ok"})
            return
        if path == "/api/tasks":
            self.send_json(service.list_tasks())
            return
        route = re.fullmatch(TASK, path)
        if route:
            self.send_json(service.task(route["task"]))
            return
        route = re.fullmatch(TASK + r"/export-summary", path)
        if route:
            self.send_json(service.export_summary(route["task"]))
            return
        route = re.fullmatch(TASK + r"/export", path)
        if route:
            with service.export(route["task"]) as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(0)
                self.send_headers("application/zip", size, disposition=f'attachment; filename="yolo-review-{route["task"][:8]}.zip"')
                shutil.copyfileobj(stream, self.wfile)
            return
        route = re.fullmatch(FRAME + r"(?:/(?P<asset>content|preview))?", path)
        if route:
            if route["asset"]:
                data, kind = service.image_bytes(route["task"], route["image"], route["asset"] == "preview")
                self.send_bytes(data, kind)
            else:
                self.send_json(service.frame(route["task"], route["image"]))
            return
        if path in STATIC:
            name = STATIC[path]
            file = next((p for p in (ROOT / "static" / name, ROOT / name) if p.is_file()), None)
            if file:
                self.send_bytes(file.read_bytes(), STATIC_TYPES[file.suffix])
                return
        raise KeyError("页面或接口不存在")

    def is_local(self):
        try:
            host = urlsplit("//" + self.headers.get("Host", ""))
            allowed = {"127.0.0.1", "localhost"}
            if host.hostname not in allowed or host.username or host.path or host.query or host.fragment:
                return False
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            source = urlsplit(origin)
            return (source.scheme == "http" and source.hostname in allowed and not source.username
                    and (source.port or 80) == self.server.server_port)
        except ValueError:
            return False

    def content_length(self, limit):
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= limit:
            raise ValueError("文件或请求内容为空，或超出大小限制")
        return size

    def zip_body(self):
        if self.headers.get_content_type() not in ("application/zip", "application/octet-stream",
                                                    "application/x-zip-compressed"):
            raise ValueError("请上传 ZIP 文件")
        remaining = self.content_length(MAX_UPLOAD)
        spool = SpooledTemporaryFile(max_size=8 * 1024**2)
        try:
            while remaining:
                chunk = self.rfile.read(min(remaining, 1024**2))
                if not chunk:
                    raise ValueError("ZIP 上传未完成")
                spool.write(chunk)
                remaining -= len(chunk)
            spool.seek(0)
            return spool
        except Exception:
            spool.close()
            raise

    def send_headers(self, kind, size, status=HTTPStatus.OK, disposition=None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()

    def send_json(self, value, status=HTTPStatus.OK):
        self.send_bytes(dumps(value).encode(), "application/json; charset=utf-8", status)

    def send_bytes(self, data, content_type, status=HTTPStatus.OK):
        self.send_headers(content_type, len(data), status)
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="本机 YOLO OBB 标注和复核")
    parser.add_argument("--port", type=int, default=5187)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".local")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.store = Store(args.data_dir.resolve())
        print(f"YOLO OBB 审核平台 http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
