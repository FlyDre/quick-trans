import argparse
import json
import os
import queue
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from quick_trans.asr import Asr
from quick_trans.audio import AudioStream
from quick_trans.mt import Translator
from quick_trans.vad import VadSegmenter


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _static_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "webui_static")


def _pick_local_model(relative_path: str, fallback: str) -> str:
    p = os.path.join(_repo_root(), relative_path)
    return p if os.path.isdir(p) else fallback


class _Job:
    def __init__(self, job_id: str, media_path: str, cfg: dict) -> None:
        self.job_id = job_id
        self.media_path = media_path
        self.cfg = cfg
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _emit(self, payload: dict) -> None:
        self.q.put(payload)

    def _run(self) -> None:
        try:
            self._emit({"type": "status", "status": "loading"})
            sample_rate = 16000
            frame_ms = int(self.cfg.get("frame_ms", 30))
            audio = AudioStream(
                input_path=self.media_path,
                sample_rate=sample_rate,
                frame_ms=frame_ms,
                realtime=True,
            )
            vad = VadSegmenter(
                sample_rate=sample_rate,
                frame_ms=frame_ms,
                aggressiveness=int(self.cfg.get("vad_aggressiveness", 2)),
                start_ms=int(self.cfg.get("vad_start_ms", 150)),
                end_ms=int(self.cfg.get("vad_end_ms", 450)),
                min_segment_s=float(self.cfg.get("min_segment_s", 0.4)),
                max_segment_s=float(self.cfg.get("max_segment_s", 10.0)),
            )
            asr = Asr(
                model_name=str(self.cfg.get("asr_model", "small")),
                device=str(self.cfg.get("asr_device", "cuda")),
                compute_type=str(self.cfg.get("asr_compute_type", "int8_float16")),
                beam_size=int(self.cfg.get("asr_beam_size", 1)),
                language=str(self.cfg.get("language", "ja")),
            )
            mt = Translator(
                model_name=str(self.cfg.get("mt_model", "facebook/nllb-200-distilled-600M")),
                device=str(self.cfg.get("mt_device", "cuda")),
                beam_size=int(self.cfg.get("mt_beam_size", 1)),
                hf_token=None,
            )
            self._emit({"type": "status", "status": "running"})
            for seg in vad.segments(audio.frames()):
                asr_segs = asr.transcribe(seg.audio, offset_s=seg.start_s)
                for s in asr_segs:
                    zh = mt.translate_ja_to_zh(s.text)
                    self._emit(
                        {
                            "type": "cue",
                            "start": float(s.start_s),
                            "end": float(s.end_s),
                            "asr": s.text,
                            "zh": zh,
                        }
                    )
            self._emit({"type": "done"})
        except Exception as e:
            self._emit({"type": "error", "message": str(e)})
        finally:
            self.done.set()


class _State:
    def __init__(self, upload_dir: str) -> None:
        self.upload_dir = upload_dir
        self.jobs: dict[str, _Job] = {}
        self.lock = threading.Lock()

    def create_job(self, media_path: str, cfg: dict) -> _Job:
        job_id = uuid.uuid4().hex
        job = _Job(job_id=job_id, media_path=media_path, cfg=cfg)
        with self.lock:
            self.jobs[job_id] = job
        job.start()
        return job

    def get_job(self, job_id: str) -> _Job | None:
        with self.lock:
            return self.jobs.get(job_id)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    n = int(handler.headers.get("Content-Length", "0") or "0")
    return handler.rfile.read(n) if n > 0 else b""


def _parse_content_disposition(v: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in v.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, val = part.split("=", 1)
        k = k.strip().lower()
        val = val.strip().strip('"')
        out[k] = val
    return out


def _parse_multipart_bytes(content_type: str, body: bytes) -> tuple[dict[str, str], tuple[str, bytes] | None]:
    ct = content_type
    boundary = None
    for p in ct.split(";"):
        p = p.strip()
        if p.lower().startswith("boundary="):
            boundary = p.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise ValueError("missing multipart boundary")

    bnd = ("--" + boundary).encode("utf-8")
    parts = body.split(bnd)
    fields: dict[str, str] = {}
    file_part: tuple[str, bytes] | None = None

    for raw in parts:
        raw = raw.strip(b"\r\n")
        if not raw or raw == b"--":
            continue
        if raw.endswith(b"--"):
            raw = raw[:-2]
            raw = raw.strip(b"\r\n")
        header_blob, sep, content = raw.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers: dict[str, str] = {}
        for line in header_blob.split(b"\r\n"):
            if b":" not in line:
                continue
            k, v = line.split(b":", 1)
            headers[k.decode("utf-8", "ignore").strip().lower()] = v.decode("utf-8", "ignore").strip()

        cd = headers.get("content-disposition", "")
        cd_kv = _parse_content_disposition(cd)
        name = cd_kv.get("name") or ""
        filename = cd_kv.get("filename")
        if not name:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]
        if filename:
            if name == "file":
                file_part = (os.path.basename(filename), content)
        else:
            fields[name] = content.decode("utf-8", "ignore")

    return fields, file_part


class _Handler(BaseHTTPRequestHandler):
    server: "WebUiServer"

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, obj: dict, status: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: str, content_type: str) -> None:
        if not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with open(path, "rb") as f:
            raw = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/" or u.path == "/index.html":
            return self._send_file(os.path.join(_static_dir(), "index.html"), "text/html; charset=utf-8")
        if u.path == "/events":
            qs = parse_qs(u.query or "")
            job_id = (qs.get("job") or [""])[0]
            job = self.server.state.get_job(job_id)
            if job is None:
                return self._send_json({"error": "job_not_found"}, status=404)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b": ok\n\n")
            self.wfile.flush()
            last_ping = time.perf_counter()
            while True:
                try:
                    item = job.q.get(timeout=0.5)
                    line = ("data: " + json.dumps(item, ensure_ascii=False) + "\n\n").encode("utf-8")
                    self.wfile.write(line)
                    self.wfile.flush()
                    if item.get("type") in ("done", "error"):
                        break
                except queue.Empty:
                    now = time.perf_counter()
                    if now - last_ping > 10.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = now
                except BrokenPipeError:
                    break
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path != "/start":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        ct = (self.headers.get("Content-Type") or "").lower()
        if "multipart/form-data" not in ct:
            return self._send_json({"error": "expected_multipart_form_data"}, status=400)

        body = _read_body(self)
        try:
            fields, file_part = _parse_multipart_bytes(self.headers.get("Content-Type") or "", body)
        except Exception as e:
            return self._send_json({"error": "bad_multipart", "message": str(e)}, status=400)
        if file_part is None:
            return self._send_json({"error": "missing_file"}, status=400)
        filename, file_bytes = file_part
        job_dir = os.path.join(self.server.state.upload_dir, uuid.uuid4().hex)
        os.makedirs(job_dir, exist_ok=True)
        media_path = os.path.join(job_dir, filename)

        with open(media_path, "wb") as f:
            f.write(file_bytes)

        cpu = str(fields.get("cpu") or "").lower() in ("1", "true", "on", "yes")
        asr_model = str(fields.get("asr_model") or "").strip() or _pick_local_model(r"llm\gpustack\faster-whisper-medium", "medium")
        mt_model = str(fields.get("mt_model") or "").strip() or _pick_local_model(r"llm\facebook\nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M")
        if os.path.isdir(asr_model):
            asr_model = os.path.abspath(asr_model)
        if os.path.isdir(mt_model):
            mt_model = os.path.abspath(mt_model)

        cfg = {
            "frame_ms": 30,
            "vad_aggressiveness": 2,
            "vad_start_ms": 150,
            "vad_end_ms": 450,
            "min_segment_s": 0.4,
            "max_segment_s": 10.0,
            "language": "ja",
            "asr_model": asr_model,
            "asr_device": "cpu" if cpu else "cuda",
            "asr_compute_type": str(fields.get("asr_compute_type") or "int8_float16"),
            "asr_beam_size": int(str(fields.get("asr_beam_size") or "1")),
            "mt_model": mt_model,
            "mt_device": "cpu" if cpu else "cuda",
            "mt_beam_size": int(str(fields.get("mt_beam_size") or "1")),
        }

        job = self.server.state.create_job(media_path=media_path, cfg=cfg)
        self._send_json({"job": job.job_id})


class WebUiServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: _State) -> None:
        super().__init__(server_address, _Handler)
        self.state = state


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="quick-trans-webui")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upload-dir", default=os.path.join(_repo_root(), ".webui_uploads"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    upload_dir = os.path.abspath(args.upload_dir)
    os.makedirs(upload_dir, exist_ok=True)
    state = _State(upload_dir=upload_dir)
    httpd = WebUiServer((args.host, int(args.port)), state=state)
    print(f"Web UI: http://{args.host}:{int(args.port)}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
