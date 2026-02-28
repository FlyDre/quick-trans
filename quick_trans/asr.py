from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class AsrSegment:
    start_s: float
    end_s: float
    text: str


class Asr:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        language: str = "ja",
    ) -> None:
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._language = language

    def transcribe(self, audio_16k_float32: np.ndarray, offset_s: float) -> List[AsrSegment]:
        segments, _info = self._model.transcribe(
            audio_16k_float32,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        out: List[AsrSegment] = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            out.append(AsrSegment(start_s=offset_s + float(s.start), end_s=offset_s + float(s.end), text=text))
        return out
