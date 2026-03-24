"""Audio post-processing: normalize, silence insertion, format conversion."""

from __future__ import annotations

import numpy as np


def process_segment(
    pcm: np.ndarray,
    sample_rate: int,
    role: str,
    prev_role: str | None,
    inter_silence_ms: int = 300,
    intra_silence_ms: int = 80,
) -> bytes:
    """Normalize PCM, prepend silence, and convert to int16 LE bytes.

    Parameters
    ----------
    pcm:
        Float32 audio samples.
    sample_rate:
        Sample rate in Hz.
    role:
        Current speaker role/name.
    prev_role:
        Previous speaker role/name (``None`` for first segment).
    inter_silence_ms:
        Silence between different speakers.
    intra_silence_ms:
        Silence between segments of the same speaker.

    Returns
    -------
    bytes
        Raw int16 little-endian PCM.
    """
    audio = pcm.astype(np.float32)

    # 1. Peak-normalize to [-1.0, 1.0]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # 2. Prepend silence
    if prev_role is None:
        silence_ms = 0
    elif role != prev_role:
        silence_ms = inter_silence_ms
    else:
        silence_ms = intra_silence_ms

    if silence_ms > 0:
        silence_samples = int(sample_rate * silence_ms / 1000)
        silence = np.zeros(silence_samples, dtype=np.float32)
        audio = np.concatenate([silence, audio])

    # 3. Float32 → int16 LE
    audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return audio_i16.tobytes()


def estimate_word_timestamps(
    text: str, audio_duration_ms: float
) -> tuple[list[str], list[int], list[int]]:
    """Estimate per-word timestamps proportional to character length.

    Returns
    -------
    tuple
        ``(words, wtimes_ms, wdurations_ms)`` — parallel lists.
    """
    words = text.split()
    if not words:
        return [], [], []

    char_lengths = [len(w) for w in words]
    total_chars = sum(char_lengths)
    if total_chars == 0:
        # Edge case: all empty strings
        equal_dur = int(audio_duration_ms / len(words))
        wtimes = [i * equal_dur for i in range(len(words))]
        wdurations = [equal_dur] * len(words)
        return words, wtimes, wdurations

    wdurations: list[int] = []
    for length in char_lengths:
        dur = int(audio_duration_ms * length / total_chars)
        wdurations.append(max(dur, 1))

    # Adjust last duration so total matches
    assigned = sum(wdurations)
    if assigned != int(audio_duration_ms):
        wdurations[-1] += int(audio_duration_ms) - assigned

    wtimes: list[int] = []
    t = 0
    for dur in wdurations:
        wtimes.append(t)
        t += dur

    return words, wtimes, wdurations
