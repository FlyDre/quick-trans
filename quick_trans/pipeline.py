from dataclasses import dataclass
import os
import time

from quick_trans.asr import Asr
from quick_trans.audio import AudioStream
from quick_trans.mt import Translator
from quick_trans.mt_sakura import SakuraOllamaTranslator
from quick_trans.vad import VadSegmenter
from quick_trans.vtt import Cue, WebVttWriter, _format_ts


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    output_path: str
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
    asr_text_path: str | None


class Pipeline:
    def __init__(self, **kwargs) -> None:
        self._cfg = PipelineConfig(**kwargs)

    def run(self) -> None:
        t_total0 = time.perf_counter()
        if self._cfg.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self._cfg.hf_endpoint

        if self._cfg.hf_token:
            os.environ["HF_TOKEN"] = self._cfg.hf_token
        else:
            os.environ.pop("HF_TOKEN", None)
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

        audio = AudioStream(
            input_path=self._cfg.input_path,
            sample_rate=16000,
            frame_ms=self._cfg.frame_ms,
            realtime=self._cfg.realtime,
        )
        vad = VadSegmenter(
            sample_rate=audio.sample_rate,
            frame_ms=self._cfg.frame_ms,
            aggressiveness=self._cfg.vad_aggressiveness,
            start_ms=self._cfg.vad_start_ms,
            end_ms=self._cfg.vad_end_ms,
            min_segment_s=self._cfg.min_segment_s,
            max_segment_s=self._cfg.max_segment_s,
        )
        asr = Asr(
            model_name=self._cfg.asr_model,
            device=self._cfg.asr_device,
            compute_type=self._cfg.asr_compute_type,
            beam_size=self._cfg.asr_beam_size,
            language=self._cfg.language,
        )
        mt = (
            Translator(
                model_name=self._cfg.mt_model,
                device=self._cfg.mt_device,
                beam_size=self._cfg.mt_beam_size,
                hf_token=self._cfg.hf_token,
            )
            if self._cfg.mt_backend == "nllb"
            else SakuraOllamaTranslator(model_name=self._cfg.mt_model, host=self._cfg.ollama_host)
        )
        t_init = time.perf_counter() - t_total0

        t_vad = 0.0
        t_asr = 0.0
        t_mt = 0.0
        t_vtt = 0.0
        t_asr_text = 0.0
        audio_s = 0.0

        asr_f = None
        if self._cfg.asr_text_path:
            t0 = time.perf_counter()
            os.makedirs(os.path.dirname(self._cfg.asr_text_path) or ".", exist_ok=True)
            asr_f = open(self._cfg.asr_text_path, "w", encoding="utf-8", newline="\n")
            t_init += time.perf_counter() - t0
        try:
            with WebVttWriter(self._cfg.output_path) as vtt:
                it = iter(vad.segments(audio.frames()))
                while True:
                    t0 = time.perf_counter()
                    try:
                        seg = next(it)
                    except StopIteration:
                        break
                    t_vad += time.perf_counter() - t0
                    audio_s = max(audio_s, float(seg.end_s))

                    t1 = time.perf_counter()
                    asr_segs = asr.transcribe(seg.audio, offset_s=seg.start_s)
                    t_asr += time.perf_counter() - t1
                    if not asr_segs:
                        continue
                    for s in asr_segs:
                        if asr_f is not None:
                            t4 = time.perf_counter()
                            asr_f.write(f"{_format_ts(s.start_s)} --> {_format_ts(s.end_s)}\t{s.text}\n")
                            asr_f.flush()
                            t_asr_text += time.perf_counter() - t4
                        t2 = time.perf_counter()
                        zh = mt.translate_ja_to_zh(s.text)
                        t_mt += time.perf_counter() - t2
                        t3 = time.perf_counter()
                        vtt.write(Cue(start_s=s.start_s, end_s=s.end_s, text=f"ASR: {s.text} | ZH: {zh}"))
                        t_vtt += time.perf_counter() - t3
        finally:
            if asr_f is not None:
                asr_f.flush()
                asr_f.close()

        wall_total = time.perf_counter() - t_total0
        if audio_s > 0:
            speed_total = audio_s / wall_total if wall_total > 0 else float("inf")
            rtf_total = wall_total / audio_s
        else:
            speed_total = 0.0
            rtf_total = 0.0

        def _stage_line(name: str, t_stage: float) -> str:
            if t_stage <= 0:
                return f"{name}: {t_stage:.3f}s"
            speed = (audio_s / t_stage) if audio_s > 0 else 0.0
            return f"{name}: {t_stage:.3f}s ({speed:.2f}x)"

        t_accounted = t_init + t_vad + t_asr + t_mt + t_vtt + t_asr_text
        t_other = max(0.0, wall_total - t_accounted)

        print("性能统计")
        print(f"音频时长(已处理): {audio_s:.3f}s")
        print(f"总耗时: {wall_total:.3f}s ({speed_total:.2f}x, RTF={rtf_total:.3f})")
        print(_stage_line("初始化(模型/打开文件)", t_init))
        print(_stage_line("VAD", t_vad))
        print(_stage_line("ASR", t_asr))
        print(_stage_line("翻译", t_mt))
        print(_stage_line("写ASR文本", t_asr_text))
        print(_stage_line("写VTT", t_vtt))
        print(_stage_line("其他开销", t_other))
