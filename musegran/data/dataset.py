"""MuseGran training datasets.

Provides two Dataset classes driven by model_type:
  - DiffusionDataset: for latent diffusion training. Returns (latent_or_None, audio, info).
  - VAEDataset: for autoencoder training. Returns (audio, degraded_audio, info).

Audio loading supports WAV/FLAC/MP3/AAC with automatic resampling.
"""

import json
import logging
import random

import numpy as np
import torch
import torchaudio
from pedalboard.io import AudioFile
from pydub import AudioSegment
from torchaudio import transforms as T

from .load_dataset import load_dataset
from .utils import (
    Mono,
    PhaseFlipper,
    Stereo,
    TimedLatentPadCrop,
    TimedPadCrop,
    mp3_compress,
)

logger = logging.getLogger(__name__)


class _BaseDataset(torch.utils.data.Dataset):
    """Shared audio loading logic for all dataset variants."""

    def __init__(self, jsonl_path: str, sample_size: int, sample_rate: int,
                 random_crop: bool, force_channels: str, use_precomputed_latent: bool = False):
        super().__init__()
        self.pad_crop = TimedPadCrop(sample_size, sample_rate, randomize=random_crop)
        self.sample_rate = sample_rate
        self.use_precomputed_latent = use_precomputed_latent
        self.encoding = torch.nn.Sequential(
            Stereo() if force_channels == "stereo" else torch.nn.Identity(),
            Mono() if force_channels == "mono" else torch.nn.Identity(),
        )

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("%s line %d JSON parse error: %s", jsonl_path, line_num, e)
        logger.info("Dataset: %d samples loaded", len(self.samples))

    def _load_audio(self, filename: str) -> torch.Tensor:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = torch.from_numpy(f.read(f.frames))
                in_sr = f.samplerate
        elif ext == "aac":
            seg = AudioSegment.from_file(filename, format="aac")
            in_sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples())
            if seg.channels > 1:
                audio = torch.tensor(samples.reshape(-1, seg.channels).T).float()
            else:
                audio = torch.tensor(samples).float().unsqueeze(0)
            audio = audio / (1 << (8 * seg.sample_width - 1))
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)
        if in_sr != self.sample_rate:
            audio = T.Resample(in_sr, self.sample_rate)(audio)
        return audio

    def __len__(self):
        return len(self.samples)


class DiffusionDataset(_BaseDataset):
    """Dataset for diffusion model training. Returns (latent_or_None, audio, info)."""

    def __init__(self, jsonl_path, sample_size, sample_rate, random_crop=True, force_channels="stereo", use_precomputed_latent=False):
        super().__init__(jsonl_path, sample_size, sample_rate, random_crop, force_channels, use_precomputed_latent)
        self.augs = torch.nn.Sequential(PhaseFlipper())
        self.latent_pad_crop = TimedLatentPadCrop(sample_size, sample_rate, randomize=random_crop)

    def __getitem__(self, idx):
        json_data = self.samples[idx]
        audio_filename = json_data.get('file_path', '')

        try:
            info = dict(json_data)
            latent = None

            if self.use_precomputed_latent:
                audio = self._load_audio(audio_filename)
                latent = torch.load(info['vae_latent_path'], weights_only=True)
                latent, t_start, t_end, seconds_start, seconds_total, padding_mask, audio = self.latent_pad_crop(latent, audio)
            else:
                audio = self._load_audio(audio_filename)
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
                audio = self.augs(audio)
                audio = self.encoding(audio.clamp(-1, 1))

            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["crop_start"] = t_start
            info["crop_end"] = t_end
            info["padding_mask"] = padding_mask
            info["pre_encoded"] = latent is not None

            info.update(load_dataset(info, audio))

            if info.get("__reject__"):
                return self[random.randrange(len(self))]

            return (latent, audio, info)

        except Exception:
            logger.warning("Error loading %s, retrying with random sample", audio_filename, exc_info=True)
            return self[random.randrange(len(self))]


class VAEDataset(_BaseDataset):
    """Dataset for VAE training. Returns (audio, degraded_audio, info).

    When mode="sr", produces MP3-compressed degraded input for super-resolution.
    When mode="reconstruct", degraded_audio equals audio (identity, for standard AE training).
    """

    def __init__(self, jsonl_path, sample_size, sample_rate, random_crop=True, force_channels="stereo", use_precomputed_latent=False, mode="sr"):
        super().__init__(jsonl_path, sample_size, sample_rate, random_crop, force_channels, use_precomputed_latent)
        self.mode = mode

    def __getitem__(self, idx):
        json_data = self.samples[idx]
        audio_filename = json_data.get('file_path', '')

        try:
            info = dict(json_data)
            audio = self._load_audio(audio_filename)

            if self.mode == "sr" and random.random() > 0.3:
                degraded = mp3_compress(audio, self.sample_rate)
                audio, degraded, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio, degraded)
            else:
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
                degraded = audio

            audio = self.encoding(audio.clamp(-1, 1))
            degraded = self.encoding(degraded.clamp(-1, 1))

            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["crop_start"] = t_start
            info["crop_end"] = t_end
            info["padding_mask"] = padding_mask

            info.update(load_dataset(info, audio))

            return (audio, degraded, info)

        except Exception:
            logger.warning("Error loading %s, retrying with random sample", audio_filename, exc_info=True)
            return self[random.randrange(len(self))]


def collation_fn(samples):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        elif isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], dict):
            b = list(b)
        result.append(b)
    return result


def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, model_type="diffusion_cond", training_config=None, audio_channels=2, num_workers=4):
    force_channels = "mono" if audio_channels == 1 else "stereo"

    common_kwargs = dict(
        jsonl_path=dataset_config["jsonl_path"],
        sample_rate=sample_rate,
        sample_size=sample_size,
        random_crop=dataset_config.get("random_crop", True),
        force_channels=force_channels,
        use_precomputed_latent=dataset_config.get("use_precomputed_latent", False),
    )

    if model_type == "autoencoder":
        mode = (training_config or {}).get("mode", "reconstruct")
        train_set = VAEDataset(**common_kwargs, mode=mode)
    else:
        train_set = DiffusionDataset(**common_kwargs)

    return torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True,
        pin_memory=True, drop_last=True, collate_fn=collation_fn,
    )
