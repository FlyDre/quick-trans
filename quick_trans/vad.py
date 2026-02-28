from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional, Tuple

import numpy as np
try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None

from quick_trans.audio import AudioFrame


@dataclass(frozen=True)
class SpeechSegment:
    audio: np.ndarray
    start_s: float
    end_s: float


def _float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    y = np.clip(x, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    return y.tobytes()


class _EnergyVad:
    def __init__(self, aggressiveness: int) -> None:
        self._noise_db = -50.0
        self._eps = 1e-8
        margins = {0: 8.0, 1: 10.0, 2: 12.0, 3: 14.0}
        self._margin = margins.get(aggressiveness, 12.0)
        self._floor = -38.0

    def is_speech(self, x: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
        db = 20.0 * float(np.log10(rms + self._eps))
        is_speech = db > max(self._floor, self._noise_db + self._margin)
        if not is_speech:
            self._noise_db = 0.97 * self._noise_db + 0.03 * db
        return is_speech


class VadSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        start_ms: int = 150,
        end_ms: int = 450,
        min_segment_s: float = 0.4,
        max_segment_s: float = 10.0,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("仅支持 10/20/30ms 帧")
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._use_webrtcvad = webrtcvad is not None
        self._vad = webrtcvad.Vad(aggressiveness) if self._use_webrtcvad else _EnergyVad(aggressiveness)
        self._start_frames = max(1, int(start_ms / frame_ms))
        self._end_frames = max(1, int(end_ms / frame_ms))
        self._min_samples = int(min_segment_s * sample_rate)
        self._max_samples = int(max_segment_s * sample_rate)

        self._ring: Deque[Tuple[AudioFrame, bool]] = deque(maxlen=max(self._start_frames, self._end_frames))
        self._collecting = False
        self._buf: List[np.ndarray] = []
        self._seg_start_s: Optional[float] = None
        self._seg_end_s: Optional[float] = None
        self._voiced_frames = 0
        self._unvoiced_frames = 0

    def segments(self, frames: Iterable[AudioFrame]) -> Iterator[SpeechSegment]:
        for f in frames:
            if self._use_webrtcvad:
                is_speech = self._vad.is_speech(_float32_to_pcm16_bytes(f.samples), sample_rate=self._sample_rate)
            else:
                is_speech = self._vad.is_speech(f.samples)

            if not self._collecting:
                self._ring.append((f, is_speech))
                if len(self._ring) >= self._start_frames:
                    voiced = sum(1 for _, s in self._ring if s)
                    if voiced >= int(0.6 * self._start_frames):
                        self._collecting = True
                        frames_to_add = [rf for rf, _ in self._ring]
                        self._ring.clear()
                        self._start_segment(frames_to_add)
                continue

            self._append_frame(f, is_speech)

            if self._should_force_flush():
                seg = self._finish_segment()
                if seg is not None:
                    yield seg
                continue

            if self._unvoiced_frames >= self._end_frames:
                seg = self._finish_segment()
                if seg is not None:
                    yield seg

        seg = self._finish_segment()
        if seg is not None:
            yield seg

    def _start_segment(self, initial_frames: List[AudioFrame]) -> None:
        self._buf = []
        self._voiced_frames = 0
        self._unvoiced_frames = 0
        self._seg_start_s = initial_frames[0].start_s if initial_frames else None
        self._seg_end_s = initial_frames[-1].end_s if initial_frames else None
        for f in initial_frames:
            self._buf.append(f.samples)

    def _append_frame(self, f: AudioFrame, is_speech: bool) -> None:
        self._buf.append(f.samples)
        if self._seg_start_s is None:
            self._seg_start_s = f.start_s
        self._seg_end_s = f.end_s

        if is_speech:
            self._voiced_frames += 1
            self._unvoiced_frames = 0
        else:
            self._unvoiced_frames += 1

    def _should_force_flush(self) -> bool:
        if not self._buf:
            return False
        n_samples = sum(b.shape[0] for b in self._buf)
        return n_samples >= self._max_samples

    def _finish_segment(self) -> Optional[SpeechSegment]:
        if not self._collecting:
            return None
        if not self._buf or self._seg_start_s is None or self._seg_end_s is None:
            self._reset()
            return None

        audio = np.concatenate(self._buf, axis=0)
        if audio.shape[0] < self._min_samples:
            self._reset()
            return None

        seg = SpeechSegment(audio=audio, start_s=self._seg_start_s, end_s=self._seg_end_s)
        self._reset()
        return seg

    def _reset(self) -> None:
        self._collecting = False
        self._ring.clear()
        self._buf = []
        self._seg_start_s = None
        self._seg_end_s = None
        self._voiced_frames = 0
        self._unvoiced_frames = 0
