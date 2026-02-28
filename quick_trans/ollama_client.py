import json
import os
import urllib.error
import urllib.request


def _base_url(host: str | None) -> str:
    h = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if not h:
        h = "http://127.0.0.1:11434"
    return h.rstrip("/")


def generate(
    *,
    model: str,
    prompt: str,
    system: str | None = None,
    host: str | None = None,
    options: dict | None = None,
    raw: bool = False,
    timeout_s: float = 120.0,
) -> str:
    url = _base_url(host) + "/api/generate"
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if raw:
        payload["raw"] = True
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"Ollama HTTPError: {e.code} {msg}") from e
    except Exception as e:
        raise RuntimeError(f"连接 Ollama 失败：{e}") from e
    obj = json.loads(raw.decode("utf-8", "ignore") or "{}")
    return (obj.get("response") or "").strip()


def tags(*, host: str | None = None, timeout_s: float = 10.0) -> dict:
    url = _base_url(host) + "/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", "ignore") or "{}")
