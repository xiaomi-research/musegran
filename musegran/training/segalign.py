"""SegAlign external alignment encoders: frozen audio models (MERT, MuQ) for segment supervision."""

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor


class ExternalAlignmentEncoder(nn.Module):
    """Frozen audio encoders for External Alignment (EA) in SegAlign."""

    def __init__(self, checkpoint_dir=None, sample_rate=44100, external_alignment_names=("mert",)):
        super().__init__()

        self.external_alignment_names = list(external_alignment_names)
        self.sample_rate = sample_rate
        self.resampler_24k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)

        for name in self.external_alignment_names:
            if name == "mert":
                config = AutoConfig.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
                config.conv_pos_batch_norm = False
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, config=config, cache_dir=checkpoint_dir
                ).eval().requires_grad_(False)
                self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True
                )

            elif name == "muq":
                from muq import MuQ
                self.muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
                self.muq_model.eval().requires_grad_(False)


    @torch.no_grad()
    def _infer_mert(self, mono_24k, actual_lengths_24k, chunk_sec=5):
        """MERT inference with chunking (max 5s per chunk due to memory)."""
        chunk_size = 24000 * chunk_sec
        bsz = mono_24k.shape[0]

        all_chunks = []
        chunk_actual_lengths = []
        num_chunks_per_audio = []

        for i in range(bsz):
            n_chunks = (actual_lengths_24k[i] + chunk_size - 1) // chunk_size
            num_chunks_per_audio.append(n_chunks)
            for start in range(0, actual_lengths_24k[i], chunk_size):
                end = min(start + chunk_size, actual_lengths_24k[i])
                chunk = mono_24k[i, start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        all_chunks = torch.stack(all_chunks)
        chunk_num_features = [(l + 319) // 320 for l in chunk_actual_lengths]

        hidden = self.mert_model(all_chunks).last_hidden_state
        chunk_hidden = [hidden[i, :chunk_num_features[i]] for i in range(len(all_chunks))]

        results = []
        idx = 0
        for i in range(bsz):
            audio_hidden = torch.cat(chunk_hidden[idx:idx + num_chunks_per_audio[i]], dim=0)
            results.append(audio_hidden)
            idx += num_chunks_per_audio[i]

        return results

    @torch.no_grad()
    def _infer_muq(self, mono_24k, actual_lengths_24k, infer_bs=2):
        """MuQ inference with mini-batching."""
        num_features = [l // 960 for l in actual_lengths_24k]
        results = []

        with torch.amp.autocast('cuda', dtype=torch.float32):
            for i in range(0, mono_24k.shape[0], infer_bs):
                batch = mono_24k[i:i + infer_bs]
                hidden = self.muq_model(batch, output_hidden_states=True).last_hidden_state
                for j in range(hidden.shape[0]):
                    results.append(hidden[j, :num_features[i + j]])

        return results


    @staticmethod
    def _normalize_audio(wav, actual_lengths):
        bsz = wav.shape[0]
        means = torch.stack([wav[i, :actual_lengths[i]].mean() for i in range(bsz)])
        stds = torch.stack([wav[i, :actual_lengths[i]].var().add(1e-7).sqrt() for i in range(bsz)])
        return (wav - means.view(-1, 1)) / stds.view(-1, 1)

    def forward(self, batch, metadata, train=True):
        if not train:
            return []

        sample_dur = batch.shape[-1] / self.sample_rate
        wav_secs = [min(m['crop_end'] - m['crop_start'], sample_dur) for m in metadata]

        mono = batch.mean(dim=1)

        mono_24k = self.resampler_24k(mono)
        lengths_24k = [int(s * 24000) for s in wav_secs]
        mono_24k = self._normalize_audio(mono_24k, lengths_24k)

        all_hidden_states = []
        with torch.amp.autocast('cuda', dtype=batch.dtype):
            for name in self.external_alignment_names:
                if name == "mert":
                    all_hidden_states.append(self._infer_mert(mono_24k, lengths_24k))
                elif name == "muq":
                    all_hidden_states.append(self._infer_muq(mono_24k, lengths_24k))
                else:
                    all_hidden_states.append(None)

        return all_hidden_states
