"""VAE reconstruction quality metrics: SI-SDR, STFT loss, and mel-spectrogram distance."""

import numpy as np
import torch
import auraloss
import torchaudio
import librosa
from torch.nn import L1Loss
from pystoi import stoi
from pymcd.mcd import Calculate_MCD
from pesq import pesq


def _to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def _compute_stoi(ref, deg, sr, extended=True):
    ref, deg = _to_numpy(ref), _to_numpy(deg)
    if ref.ndim == 2 and deg.ndim == 2:
        return np.mean([stoi(ref[i], deg[i], sr, extended=extended) for i in range(ref.shape[0])])
    return stoi(ref, deg, sr, extended=extended)


def _compute_pesq(ref, deg, sr, target_sr=16000, mode='wb', device=None):
    if isinstance(ref, torch.Tensor):
        ref = torchaudio.transforms.Resample(sr, target_sr).to(device)(ref).cpu().numpy()
    if isinstance(deg, torch.Tensor):
        deg = torchaudio.transforms.Resample(sr, target_sr).to(device)(deg).cpu().numpy()
    if ref.ndim == 2 and deg.ndim == 2:
        return np.mean([pesq(target_sr, ref[i], deg[i], mode) for i in range(ref.shape[0])])
    return pesq(target_sr, ref, deg, mode)


def _high_freq_mask_stft(loss, sr, cutoff=16000):
    B, F, T = loss.shape
    n_fft = (F - 1) * 2
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr, device=loss.device)
    mask = (freqs > cutoff).unsqueeze(0).unsqueeze(-1).expand(B, F, T)
    return loss[mask].mean()


def _high_freq_mask_mel(loss, sr, cutoff=16000, n_mels=128):
    B, F, T = loss.shape
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)
    mask = torch.tensor(mel_freqs > cutoff, device=loss.device, dtype=torch.bool)
    mask = mask.unsqueeze(0).unsqueeze(-1).expand(B, F, T)
    return loss[mask].mean()


class Scorer:
    """VAE reconstruction quality scorer."""

    def __init__(self, sample_rate, loss_config, device=None):
        self.sample_rate = sample_rate
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        stft_args = loss_config['spectral']['config']
        self.stft = auraloss.freq.STFTLoss(sample_rate=sample_rate, reduction='none').to(self.device)
        self.mel_stft = auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, reduction='none').to(self.device)
        self.sd_stft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_args).to(self.device)
        self.mr_stft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_args).to(self.device)
        self.l1_loss = L1Loss().to(self.device)
        self.mcd_toolbox = Calculate_MCD(MCD_mode="plain")

    def _compute_all_freq_losses(self, ref, deg):
        try:
            pesq_score = float(_compute_pesq(ref.squeeze(0), deg.squeeze(0), self.sample_rate, device=self.device))
        except Exception:
            pesq_score = None

        return {
            "stft": float(self.stft(ref, deg).mean()),
            "mel_stft": float(self.mel_stft(ref, deg).mean()),
            "sum_and_diff_stft": float(self.sd_stft(ref, deg)),
            "multi_resolution_stft": float(self.mr_stft(ref, deg)),
            "multi_resolution_stft_left": float(self.mr_stft(ref[:, 0:1, :], deg[:, 0:1, :])),
            "multi_resolution_stft_right": float(self.mr_stft(ref[:, 1:2, :], deg[:, 1:2, :])),
            "l1_loss": float(self.l1_loss(ref, deg)),
            "mcd": float(self.mcd_toolbox.calculate_mcd(ref.squeeze(0).cpu(), deg.squeeze(0).cpu())),
            "pesq": pesq_score,
            "stoi": float(_compute_stoi(ref.squeeze(0), deg.squeeze(0), self.sample_rate)),
        }

    def _compute_high_freq_losses(self, ref, deg):
        return {
            "stft": float(_high_freq_mask_stft(self.stft(ref, deg), self.sample_rate)),
            "mel_stft": float(_high_freq_mask_mel(self.mel_stft(ref, deg), self.sample_rate)),
        }

    def _compute_pair_losses(self, ref, deg):
        return {
            "all_freq": self._compute_all_freq_losses(ref, deg),
            "high_freq": self._compute_high_freq_losses(ref, deg),
        }

    def _compute_one_vae_losses(self, audio_info):
        audio_path = audio_info["path"]
        start_frame = audio_info.get("start_frame")
        demo_samples = audio_info.get("demo_samples")
        n_each_sample = audio_info.get("n_each_sample")

        y, _ = torchaudio.load(audio_path)
        y = y.unsqueeze(0).to(self.device)

        if n_each_sample == 3:
            compress_wav = y[:, :, start_frame:start_frame + demo_samples]
            decoded_wav = y[:, :, start_frame + demo_samples:start_frame + 2 * demo_samples]
            lossless_wav = y[:, :, start_frame + 2 * demo_samples:start_frame + 3 * demo_samples]
        elif n_each_sample == 2:
            compress_wav = None
            lossless_wav = y[:, :, start_frame:start_frame + demo_samples]
            decoded_wav = y[:, :, start_frame + demo_samples:start_frame + 2 * demo_samples]
        else:
            return {"decoder_vs_lossless": None}

        result = {"decoder_vs_lossless": self._compute_pair_losses(lossless_wav, decoded_wav)}
        if compress_wav is not None:
            result["mp3_compress_vs_lossless"] = self._compute_pair_losses(lossless_wav, compress_wav)

        return result

    def forward(self, input_infos):
        """Score a list of audio segments for VAE reconstruction quality.

        Args:
            input_infos: list of {"path": str, "start_frame": int, "demo_samples": int, "n_each_sample": int}

        Returns:
            list of dicts with decoder_vs_lossless and optionally mp3_compress_vs_lossless metrics
        """
        return [self._compute_one_vae_losses(info) for info in input_infos]
