from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_BIND_IP = "0.0.0.0"
DEFAULT_BIND_PORT = 8888
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
MAX_UDP_PACKET = 65535


@dataclass
class FrameInfo:
    frame_id: int = 0
    timestamp: float = 0.0
    jpeg_bytes: bytes = b""
    width: int = 0
    height: int = 0


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = FrameInfo()

    def update(self, jpeg_bytes: bytes, width: int, height: int) -> None:
        with self._cond:
            self._frame = FrameInfo(
                frame_id=self._frame.frame_id + 1,
                timestamp=time.time(),
                jpeg_bytes=jpeg_bytes,
                width=width,
                height=height,
            )
            self._cond.notify_all()

    def snapshot(self) -> FrameInfo:
        with self._lock:
            return FrameInfo(
                frame_id=self._frame.frame_id,
                timestamp=self._frame.timestamp,
                jpeg_bytes=self._frame.jpeg_bytes,
                width=self._frame.width,
                height=self._frame.height,
            )

    def wait_for_new(self, last_frame_id: int, timeout: float = 1.0) -> FrameInfo:
        with self._cond:
            if self._frame.frame_id == last_frame_id:
                self._cond.wait(timeout=timeout)
            return FrameInfo(
                frame_id=self._frame.frame_id,
                timestamp=self._frame.timestamp,
                jpeg_bytes=self._frame.jpeg_bytes,
                width=self._frame.width,
                height=self._frame.height,
            )


class UdpReceiver(threading.Thread):
    def __init__(self, store: FrameStore, bind_ip: str, bind_port: int) -> None:
        super().__init__(daemon=True)
        self._store = store
        self._bind_ip = bind_ip
        self._bind_port = bind_port
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sock = None

    def current_bind(self) -> tuple[str, int]:
        with self._lock:
            return self._bind_ip, self._bind_port

    def set_bind(self, bind_ip: str, bind_port: int) -> None:
        with self._lock:
            self._bind_ip = bind_ip
            self._bind_port = bind_port
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _open_socket(self):
        import socket

        with self._lock:
            bind_ip = self._bind_ip
            bind_port = self._bind_port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_ip, bind_port))
        sock.settimeout(0.5)

        with self._lock:
            self._sock = sock

        return sock, bind_ip, bind_port

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                sock, bind_ip, bind_port = self._open_socket()
            except OSError as exc:
                print(f"[udp] bind failed on {self._bind_ip}:{self._bind_port}: {exc}")
                time.sleep(1.0)
                continue

            print(f"[udp] listening on {bind_ip}:{bind_port}")
            try:
                while not self._stop_event.is_set():
                    with self._lock:
                        if sock is not self._sock:
                            break
                    try:
                        data, _addr = sock.recvfrom(MAX_UDP_PACKET)
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                    if not data:
                        continue

                    arr = np.frombuffer(data, dtype=np.uint8)
                    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if image is None:
                        continue

                    self._store.update(data, int(image.shape[1]), int(image.shape[0]))
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
                with self._lock:
                    if self._sock is sock:
                        self._sock = None


def build_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UDP 图传接收器</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1115;
      --panel: #171a21;
      --line: #2a2f3a;
      --text: #e7eaf0;
      --muted: #9aa4b2;
      --accent: #5b8cff;
      --ok: #24c18a;
      --warn: #f0b44c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.4 "Segoe UI", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #11141a;
      padding: 16px;
    }
    .title {
      margin: 0 0 16px;
      font-size: 18px;
      font-weight: 600;
    }
    .field {
      margin-bottom: 12px;
    }
    label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
    }
    input, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }
    input:focus {
      outline: 2px solid color-mix(in srgb, var(--accent) 35%, transparent);
      border-color: var(--accent);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .btn {
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      font-weight: 600;
    }
    .btn.secondary {
      background: transparent;
      color: var(--text);
    }
    .status {
      margin-top: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      background: rgba(255,255,255,0.02);
      white-space: pre-wrap;
      min-height: 80px;
    }
    .viewer {
      padding: 16px;
    }
    .frame {
      width: 100%;
      height: calc(100vh - 32px);
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #000;
      object-fit: contain;
      display: block;
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .frame { height: 56vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1 class="title">UDP 图传接收器</h1>
      <div class="field">
        <label for="ip">IP</label>
        <input id="ip" value="0.0.0.0" spellcheck="false" />
      </div>
      <div class="field">
        <label for="port">端口</label>
        <input id="port" value="8888" inputmode="numeric" spellcheck="false" />
      </div>
      <div class="row">
        <button class="btn" id="apply">应用</button>
        <button class="btn secondary" id="reload">刷新</button>
      </div>
      <div class="status" id="status">等待连接</div>
    </aside>
    <main class="viewer">
      <img class="frame" id="frame" alt="stream" />
    </main>
  </div>
  <script>
    const ipEl = document.getElementById('ip');
    const portEl = document.getElementById('port');
    const statusEl = document.getElementById('status');
    const frameEl = document.getElementById('frame');

    function setStream() {
      frameEl.src = '/stream.mjpg?t=' + Date.now();
    }

    async function loadStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      ipEl.value = data.bind_ip;
      portEl.value = data.bind_port;
      statusEl.textContent =
        'UDP: ' + data.bind_ip + ':' + data.bind_port + '\\n' +
        'HTTP: ' + data.http_host + ':' + data.http_port + '\\n' +
        'Frame: ' + (data.frame_id || 0) + '\\n' +
        'Size: ' + (data.width || 0) + ' x ' + (data.height || 0);
      setStream();
    }

    async function applyConfig() {
      const body = {
        bind_ip: ipEl.value.trim(),
        bind_port: Number(portEl.value)
      };
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'config failed');
      }
      await loadStatus();
    }

    document.getElementById('apply').addEventListener('click', async () => {
      try {
        await applyConfig();
      } catch (err) {
        statusEl.textContent = String(err);
      }
    });

    document.getElementById('reload').addEventListener('click', loadStatus);
    loadStatus().catch(err => statusEl.textContent = String(err));
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "UdpViewer/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = build_html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/api/status":
            payload = self.server.app.status_payload()
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/stream.mjpg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_id = 0
            try:
                while True:
                    frame = self.server.app.store.wait_for_new(last_id, timeout=1.0)
                    if not frame.jpeg_bytes:
                        continue
                    last_id = frame.frame_id
                    chunk = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame.jpeg_bytes)}\r\n\r\n".encode("ascii")
                        + frame.jpeg_bytes
                        + b"\r\n"
                    )
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except OSError:
                return
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            bind_ip = str(payload["bind_ip"]).strip()
            bind_port = int(payload["bind_port"])
            self.server.app.set_udp_bind(bind_ip, bind_port)
        except Exception as exc:
            data = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        data = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


class ViewerApp:
    def __init__(self, http_host: str, http_port: int, bind_ip: str, bind_port: int) -> None:
        self.http_host = http_host
        self.http_port = http_port
        self.store = FrameStore()
        self.receiver = UdpReceiver(self.store, bind_ip, bind_port)
        self.receiver.start()

    def set_udp_bind(self, bind_ip: str, bind_port: int) -> None:
        self.receiver.set_bind(bind_ip, bind_port)

    def status_payload(self) -> dict:
        bind_ip, bind_port = self.receiver.current_bind()
        frame = self.store.snapshot()
        return {
            "bind_ip": bind_ip,
            "bind_port": bind_port,
            "http_host": self.http_host,
            "http_port": self.http_port,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "width": frame.width,
            "height": frame.height,
        }

    def close(self) -> None:
        self.receiver.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP JPEG 图传接收器")
    parser.add_argument("--bind-ip", default=DEFAULT_BIND_IP, help="UDP 监听 IP")
    parser.add_argument("--bind-port", type=int, default=DEFAULT_BIND_PORT, help="UDP 监听端口")
    parser.add_argument("--http-host", default=DEFAULT_HTTP_HOST, help="网页服务监听 IP")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT, help="网页服务端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ViewerApp(args.http_host, args.http_port, args.bind_ip, args.bind_port)

    class Server(ThreadingHTTPServer):
        pass

    server = Server((args.http_host, args.http_port), ViewerHandler)
    server.app = app  # type: ignore[attr-defined]

    print(f"[http] open http://{args.http_host}:{args.http_port}/")
    print(f"[udp]  bind {args.bind_ip}:{args.bind_port}")

    try:
        server.serve_forever()
    finally:
        app.close()
        server.server_close()


if __name__ == "__main__":
    main()
