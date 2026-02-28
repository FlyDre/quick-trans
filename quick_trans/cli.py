import argparse
import os
import sys
from dataclasses import dataclass

if __package__ is None:
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from quick_trans.pipeline import Pipeline


@dataclass(frozen=True)
class Args:
    input: str
    output: str
    realtime: bool
    frame_ms: int
    vad_aggressiveness: int
    vad_start_ms: int
    vad_end_ms: int
    min_segment_s: float
    max_segment_s: float
    asr_model: str
    asr_device: str
    asr_compute_type: str
    asr_beam_size: int
    mt_model: str
    mt_device: str
    mt_beam_size: int
    mt_backend: str
    ollama_host: str | None
    language: str
    hf_token: str | None
    hf_endpoint: str | None
    asr_text: str | None
    cpu: bool


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pick_local_model(relative_path: str, fallback: str) -> str:
    p = os.path.join(_repo_root(), relative_path)
    return p if os.path.isdir(p) else fallback


def _parse_args() -> Args:
    p = argparse.ArgumentParser(prog="quick-trans")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--frame-ms", type=int, default=30)

    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--vad-start-ms", type=int, default=150)
    p.add_argument("--vad-end-ms", type=int, default=450)
    p.add_argument("--min-segment-s", type=float, default=0.4)
    p.add_argument("--max-segment-s", type=float, default=10.0)

    p.add_argument("--asr-model", default=None)
    p.add_argument("--asr-device", default="cuda")
    p.add_argument("--asr-compute-type", default="int8_float16")
    p.add_argument("--asr-beam-size", type=int, default=1)

    p.add_argument("--mt-model", default=None)
    p.add_argument("--mt-device", default="cuda")
    p.add_argument("--mt-beam-size", type=int, default=1)
    p.add_argument("--mt-backend", choices=["nllb", "sakura-ollama"], default="nllb")
    p.add_argument("--ollama-host", default=None)

    p.add_argument("--language", default="ja")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--asr-text", nargs="?", const="auto", default=None)

    a = p.parse_args()
    asr_model = (
        str(a.asr_model).strip().strip("`'\"").rstrip(".") if a.asr_model else _pick_local_model(r"llm\gpustack\faster-whisper-medium", "medium")
    )
    mt_model = (
        str(a.mt_model).strip().strip("`'\"").rstrip(".")
        if a.mt_model
        else (
            _pick_local_model(r"llm\facebook\nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M")
            if a.mt_backend == "nllb"
            else "sakura-1.5b"
        )
    )
    if os.path.isdir(asr_model):
        asr_model = os.path.abspath(asr_model)
    if a.mt_backend == "nllb" and os.path.isdir(mt_model):
        mt_model = os.path.abspath(mt_model)
    asr_text: str | None
    if a.asr_text is None:
        asr_text = None
    else:
        s = str(a.asr_text).strip().strip("`'\"")
        if s.lower() == "auto" or s == "":
            base, _ext = os.path.splitext(os.path.abspath(a.output))
            asr_text = base + ".asr.txt"
        else:
            asr_text = os.path.abspath(s)
    asr_device = "cpu" if a.cpu else a.asr_device
    mt_device = "cpu" if a.cpu else a.mt_device
    return Args(
        input=a.input,
        output=a.output,
        realtime=a.realtime,
        frame_ms=a.frame_ms,
        vad_aggressiveness=a.vad_aggressiveness,
        vad_start_ms=a.vad_start_ms,
        vad_end_ms=a.vad_end_ms,
        min_segment_s=a.min_segment_s,
        max_segment_s=a.max_segment_s,
        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=a.asr_compute_type,
        asr_beam_size=a.asr_beam_size,
        mt_model=mt_model,
        mt_device=mt_device,
        mt_beam_size=a.mt_beam_size,
        mt_backend=a.mt_backend,
        ollama_host=(str(a.ollama_host).strip().strip("`'\"").rstrip("/") if a.ollama_host else None),
        language=a.language,
        hf_token=(str(a.hf_token).strip() if a.hf_token else None),
        hf_endpoint=(str(a.hf_endpoint).strip().strip("`'\"").rstrip("/") if a.hf_endpoint else None),
        asr_text=asr_text,
        cpu=bool(a.cpu),
    )


def main() -> None:
    a = _parse_args()
    input_path = os.path.abspath(a.input)
    output_path = os.path.abspath(a.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pipeline = Pipeline(
        input_path=input_path,
        output_path=output_path,
        realtime=a.realtime,
        frame_ms=a.frame_ms,
        vad_aggressiveness=a.vad_aggressiveness,
        vad_start_ms=a.vad_start_ms,
        vad_end_ms=a.vad_end_ms,
        min_segment_s=a.min_segment_s,
        max_segment_s=a.max_segment_s,
        asr_model=a.asr_model,
        asr_device=a.asr_device,
        asr_compute_type=a.asr_compute_type,
        asr_beam_size=a.asr_beam_size,
        mt_model=a.mt_model,
        mt_device=a.mt_device,
        mt_beam_size=a.mt_beam_size,
        mt_backend=a.mt_backend,
        ollama_host=a.ollama_host,
        language=a.language,
        hf_token=a.hf_token,
        hf_endpoint=a.hf_endpoint,
        asr_text_path=a.asr_text,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
