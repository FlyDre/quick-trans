import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

try:
    import miniaudio  # type: ignore
except Exception:
    miniaudio = None

from quick_trans.ncm import decrypt_ncm


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    start_s: float
    end_s: float


def _decode_any_to_float32_mono(
    input_path: str,
    sample_rate: int,
) -> np.ndarray:
    if miniaudio is None:
        raise RuntimeError("缺少 miniaudio，无法在不依赖 ffmpeg 的情况下解码多种格式")

    if input_path.lower().endswith(".ncm"):
        d = decrypt_ncm(input_path)
        decoded = miniaudio.decode(
            d.audio_bytes,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=sample_rate,
        )
        return np.array(decoded.samples, dtype=np.float32)

    decoded = miniaudio.decode_file(
        input_path,
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,
        sample_rate=sample_rate,
    )
    return np.array(decoded.samples, dtype=np.float32)


class AudioStream:
    def __init__(
        self,
        input_path: str,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        realtime: bool = False,
    ) -> None:
        self._input_path = input_path
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._realtime = realtime

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_samples(self) -> int:
        return int(self._sample_rate * (self._frame_ms / 1000.0))

    def frames(self) -> Iterator[AudioFrame]:
        audio = _decode_any_to_float32_mono(self._input_path, sample_rate=self._sample_rate)
        yield from _frames_from_array(
            audio,
            sample_rate=self._sample_rate,
            frame_samples=self.frame_samples,
            realtime=self._realtime,
        )


def _frames_from_array(
    audio: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    realtime: bool,
) -> Iterator[AudioFrame]:
    t0 = time.perf_counter()
    i = 0
    n = audio.shape[0]
    while i < n:
        j = min(i + frame_samples, n)
        chunk = audio[i:j]
        if chunk.shape[0] < frame_samples:
            pad = np.zeros((frame_samples - chunk.shape[0],), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        start_s = i / sample_rate
        end_s = (i + frame_samples) / sample_rate
        if realtime:
            target = t0 + start_s
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
        yield AudioFrame(samples=chunk, start_s=start_s, end_s=end_s)
        i += frame_samples
