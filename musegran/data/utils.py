"""Audio processing utilities.

Includes random cropping with timing metadata, channel format conversion,
phase augmentation, MP3 compression for super-resolution training, and
audio preparation helpers used across training and inference.
"""

import math
import random

import pedalboard
import torch
from pedalboard import MP3Compressor
from torch import nn
from torchaudio import transforms as T


class PadCrop(nn.Module):
    """Simple pad-or-crop to a fixed length. Used at inference time."""

    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output


class TimedPadCrop(nn.Module):
    """Random crop that also returns timing metadata for conditioning.

    Randomly selects a fixed-length window from the audio. If the audio is
    shorter than the target length, it is zero-padded on the right.

    Returns:
        cropped: Cropped audio [channels, n_samples]
        t_start: Crop start time in seconds (float, precise)
        t_end: Crop end time in seconds (float, precise)
        seconds_start: Floor of t_start (int, for NumberEmbedder conditioners)
        seconds_total: Ceil of total audio duration (int)
        padding_mask: Binary mask [n_samples], 1 where audio exists

    When paired_source is provided (for VAE-SR paired cropping), returns
    (cropped, cropped_paired, t_start, t_end, seconds_start, seconds_total, padding_mask).
    """

    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor, paired_source=None):
        n_channels, n_samples = source.shape

        upper_bound = max(0, n_samples - self.n_samples)
        offset = random.randint(0, upper_bound) if (self.randomize and n_samples > self.n_samples) else 0

        t_start = offset / self.sample_rate
        t_end = min(offset + self.n_samples, n_samples) / self.sample_rate

        cropped = source.new_zeros([n_channels, self.n_samples])
        cropped[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]

        seconds_start = math.floor(t_start)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1

        if paired_source is not None:
            cropped_paired = paired_source.new_zeros([n_channels, self.n_samples])
            cropped_paired[:, :min(n_samples, self.n_samples)] = paired_source[:, offset:offset + self.n_samples]
            return cropped, cropped_paired, t_start, t_end, seconds_start, seconds_total, padding_mask

        return cropped, t_start, t_end, seconds_start, seconds_total, padding_mask


class TimedLatentPadCrop(nn.Module):
    """Random crop in latent space that also returns timing metadata.

    Operates on pre-computed VAE latents. Infers audio-domain timing from the
    latent length and compression ratio, then crops in latent frames.

    Args:
        n_samples: Target length in audio samples (used to compute latent target length).
        sample_rate: Audio sample rate for timing calculations.
        n_channels: Number of latent channels.
        compress_ratio: VAE temporal compression ratio (audio_samples / latent_frames).
    """

    def __init__(self, n_samples: int, sample_rate: int = 44100, randomize: bool = True,
                 n_channels: int = 64, compress_ratio: int = 2048):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.n_channels = n_channels
        self.compress_ratio = compress_ratio

    def __call__(self, source: torch.Tensor, audio=None):
        n_samples = source.shape[-1] * self.compress_ratio
        latent_len = self.n_samples // self.compress_ratio

        upper_bound = max(0, n_samples - self.n_samples)
        offset = random.randint(0, upper_bound) if (self.randomize and n_samples > self.n_samples) else 0

        t_start = offset / self.sample_rate
        t_end = min(offset + self.n_samples, n_samples) / self.sample_rate

        latent_offset = offset // self.compress_ratio
        actual_latent_len = min(n_samples, self.n_samples) // self.compress_ratio

        cropped = source.new_zeros([self.n_channels, latent_len])
        cropped[:, :actual_latent_len] = source[:, latent_offset:latent_offset + latent_len]

        seconds_start = math.floor(t_start)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        padding_mask = torch.zeros([latent_len])
        padding_mask[:actual_latent_len] = 1

        if audio is not None:
            cropped_audio = audio.new_zeros([2, self.n_samples])
            cropped_audio[:, :min(audio.shape[1], self.n_samples)] = audio[:, offset:offset + self.n_samples]
            return cropped, t_start, t_end, seconds_start, seconds_total, padding_mask, cropped_audio

        return cropped, t_start, t_end, seconds_start, seconds_total, padding_mask, None


class PhaseFlipper(nn.Module):
    """Randomly invert audio phase as a data augmentation."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, signal):
        return -signal if random.random() < self.p else signal


class Mono(nn.Module):
    """Convert multi-channel audio to mono by averaging channels."""

    def __call__(self, signal):
        return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal


class Stereo(nn.Module):
    """Convert audio to stereo (2 channels). Duplicates mono, truncates >2 channels."""

    def __call__(self, signal):
        if signal.dim() == 1:
            return signal.unsqueeze(0).repeat(2, 1)
        if signal.shape[0] == 1:
            return signal.repeat(2, 1)
        if signal.shape[0] > 2:
            return signal[:2, :]
        return signal


def mp3_compress(audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply random-quality MP3 compression as degradation for SR training."""
    audio = audio_tensor.numpy().T
    vbr_quality = random.uniform(0.0, 7.0)
    board = pedalboard.Pedalboard([MP3Compressor(vbr_quality=vbr_quality)])
    processed = board(audio, sample_rate)
    return torch.from_numpy(processed).T


def set_audio_channels(audio: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Convert audio to target number of channels. Expects [batch, channels, samples]."""
    if target_channels == 1:
        return audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            return audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            return audio[:, :2, :]
    return audio


def prepare_audio(audio: torch.Tensor, in_sr: int, target_sr: int, target_length: int,
                  target_channels: int, device: str) -> torch.Tensor:
    """Resample, crop/pad, and set channels for model input."""
    audio = audio.to(device)
    if in_sr != target_sr:
        audio = T.Resample(in_sr, target_sr).to(device)(audio)
    audio = PadCrop(target_length, randomize=False)(audio)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    return set_audio_channels(audio, target_channels)
