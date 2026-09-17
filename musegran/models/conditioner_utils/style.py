"""Style/timbre conditioning: extract a reference audio clip."""

import random

import torchaudio


def _load_audio(path, target_sr):
    """Load audio file, resample to target_sr stereo. Returns None on failure."""
    try:
        fmt = path.rsplit('.', 1)[-1] if '.' in path else None
        waveform, sr = torchaudio.load(path, format=fmt, normalize=True)
    except (OSError, RuntimeError):
        return None

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr, dtype=waveform.dtype)(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    return waveform


def _random_clip(wav, total_duration, sample_rate, clip_duration):
    """Extract a random clip from the 30%-70% middle region."""
    clip_frames = sample_rate * clip_duration
    lo = total_duration * 0.3
    hi = total_duration * 0.7 - clip_duration

    if hi < lo:
        start = max((total_duration - clip_duration) / 2, 0)
    else:
        start = random.uniform(lo, hi)

    st = int(start * sample_rate)
    clip = wav[:, st:st + clip_frames]
    return clip if clip.shape[1] == clip_frames else None


def get_style_prompt(x, mode='wav', sample_rate=44100, clip_duration=10):
    """Extract an audio clip for style/timbre conditioning.

    Args:
        x: dict {"audio": tensor/path, "seconds_start", "seconds_end"}
           or legacy list [_, (start, end), audio, prompt_text].
        mode: 'wav' returns audio tensor [2, sr*clip_duration], 'text' returns prompt string.
        sample_rate: target sample rate for the output clip.
        clip_duration: duration of the extracted clip in seconds.
    """
    if isinstance(x, dict):
        seconds_start = float(x['seconds_start'])
        seconds_end = float(x['seconds_end'])
        wav = x.get('audio')
    elif isinstance(x, list):
        _, (seconds_start, seconds_end), wav, _ = x
        seconds_start, seconds_end = float(seconds_start), float(seconds_end)
    else:
        return None

    if mode == 'text':
        return x.get('prompt', '') if isinstance(x, dict) else x[3]

    if isinstance(wav, str):
        wav = _load_audio(wav, sample_rate)
    if wav is None:
        return None

    return _random_clip(wav, seconds_end - seconds_start, sample_rate, clip_duration)
