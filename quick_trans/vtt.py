from dataclasses import dataclass
from typing import TextIO


def _format_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000.0))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str


class WebVttWriter:
    def __init__(self, path: str) -> None:
        self._path = path
        self._f: TextIO | None = None

    def __enter__(self) -> "WebVttWriter":
        self._f = open(self._path, "w", encoding="utf-8", newline="\n")
        self._f.write("WEBVTT\n\n")
        self._f.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None

    def write(self, cue: Cue) -> None:
        if self._f is None:
            raise RuntimeError("writer 未打开")
        text = (cue.text or "").strip()
        if not text:
            return
        self._f.write(f"{_format_ts(cue.start_s)} --> {_format_ts(cue.end_s)}\n")
        self._f.write(text + "\n")
        self._f.flush()
