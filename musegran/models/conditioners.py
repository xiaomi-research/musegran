"""Conditioning modules for diffusion models.

Provides text (T5, Qwen, GLAP), audio (style, VAE), structural (chord, beat,
motif, segment), and positional (bar, beat positions) conditioners, plus a
MultiConditioner that applies them jointly.

Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py
"""

import logging
import random
import typing as tp
import warnings
from math import pi

import numpy as np
import torch
from einops import rearrange
from muq import MuQMuLan
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .autoencoders import AutoencoderPretransform, create_autoencoder_from_config
from .conditioner_utils.beats import get_bar_pos, get_beat_pos, get_beat_tensor
from .conditioner_utils.chords import get_chord2chroma_tensor
from .conditioner_utils.glap import GlapBaseBert
from .conditioner_utils.motif import midi_to_pianoroll, batch_pad_pianorolls
from .conditioner_utils.structure import get_structure_tensor_from_json
from .conditioner_utils.style import get_style_prompt
from .builder import load_ckpt_state_dict


class _LearnedPositionalEmbedding(nn.Module):
    """Learnable Fourier feature embedding for scalar inputs."""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class NumberEmbedder(nn.Module):
    """Embed scalar numbers into a feature space via learned Fourier features + linear."""

    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        self.features = features
        self.embedding = nn.Sequential(
            _LearnedPositionalEmbedding(dim),
            nn.Linear(in_features=dim + 1, out_features=features),
        )

    def forward(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        return embedding.view(*shape, self.features)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token from each sequence (for causal LM embeddings)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            allow_null_emb: bool = False,
            seed: int = -1,
            **kwargs
            ):
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        self.rng = torch.Generator()
        seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
        self.rng.manual_seed(seed)

        self.allow_null_emb = allow_null_emb
        if self.allow_null_emb:
            self.null_emb = nn.Parameter(torch.randn(1, 1, output_dim))
            nn.init.normal_(self.null_emb, mean=0.0, std=0.02)

    def _to_proj_device(self, tensor, device):
        """Move tensor to match proj_out weight dtype/device, falling back to specified device."""
        if hasattr(self.proj_out, 'weight'):
            return tensor.to(self.proj_out.weight)
        return tensor.to(device)

    def _ensure_on_device(self, module, device):
        """Move module to device only if not already there."""
        try:
            p = next(module.parameters())
            if p.device != torch.device(device):
                module.to(device)
        except StopIteration:
            pass

    def _apply_dropout_mask(self, embeds, batch_size, drop_prob, fail_mask, seq_len, device):
        """Apply training dropout + null embedding fallback."""
        if self.training and drop_prob > 0:
            rand_drop_mask = (torch.rand(batch_size, generator=self.rng).to(device) < drop_prob)
            final_drop_mask = rand_drop_mask | fail_mask
        else:
            final_drop_mask = fail_mask

        if self.allow_null_emb:
            null_expanded = self.null_emb.expand(batch_size, seq_len, -1).to(device=embeds.device, dtype=embeds.dtype)
        else:
            null_expanded = torch.zeros_like(embeds)
        return torch.where(final_drop_mask.view(-1, 1, 1), null_expanded, embeds)

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class CategoricalConditioner(Conditioner):
    """Categorical conditioner for fixed-class features (e.g. music key, gender)."""

    def __init__(self, output_dim, categories: list, d_model=256, embed_name="embedding", **kwargs):
        super().__init__(dim=d_model, output_dim=output_dim, **kwargs)
        self.categories = categories
        setattr(self, embed_name, nn.Embedding(len(categories) + 1, d_model, padding_idx=0))
        self._embed_name = embed_name

    @property
    def _embedding(self):
        return getattr(self, self._embed_name)

    def _to_indices(self, x):
        return torch.tensor([
            self.categories.index(c) + 1 if c in self.categories else 0
            for c in x
        ], dtype=torch.long)

    def forward(self, x, device, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)

        if self.training and drop_prob > 0:
            rand_vals = torch.rand(batch_size, generator=self.rng)
            x = [None if rand_vals[i] < drop_prob else c for i, c in enumerate(x)]

        if not isinstance(x[0], torch.Tensor):
            x = self._to_indices(x)

        x = x.view(batch_size)

        self._ensure_on_device(self._embedding, device)
        embeds = self._embedding(x.to(device))
        embeds = self.proj_out(embeds).unsqueeze(1)
        mask = torch.ones(batch_size, 1).to(device)

        return (embeds, mask)


class MusicKeyConditioner(CategoricalConditioner):
    """Music key conditioner mapping key names (e.g. 'C major') to embeddings."""

    CLASSES = [
        'E major', 'D minor', 'C minor', 'G minor', 'Bb major', 'Eb major',
        'B minor', 'Eb minor', 'Bb minor', 'E minor', 'F major', 'Ab minor',
        'A minor', 'F# minor', 'B major', 'C# minor', 'G major', 'C# major',
        'F# major', 'A major', 'D major', 'C major', 'F minor', 'Ab major',
    ]

    def __init__(self, output_dim, d_model=256, **kwargs):
        kwargs.pop('num_keys', None)
        super().__init__(output_dim, categories=self.CLASSES, d_model=d_model, embed_name="struct_embed", **kwargs)


class IntConditioner(Conditioner):
    """Conditioner for discrete integer values via learned embedding."""

    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        super().__init__(output_dim, output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def forward(self, ints: tp.List[int], device=None, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(ints)

        if not isinstance(ints, torch.Tensor):
            ints = torch.tensor(ints)
        ints = ints.clamp(self.min_val, self.max_val).to(device)

        self._ensure_on_device(self.int_embedder, device)
        embeds = self.int_embedder(ints).unsqueeze(1)

        fail_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, fail_mask, 1, device)

        return (embeds, torch.ones(embeds.shape[0], 1).to(device))


class NumberConditioner(Conditioner):
    """Conditioner for continuous float values, normalized to [0, 1] range."""

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1, **kwargs):
        super().__init__(output_dim, output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(floats)

        floats = [float(x) for x in floats]
        floats = torch.tensor(floats).to(device)
        floats = floats.clamp(self.min_val, self.max_val)
        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)
        embeds = self.embedder(normalized_floats).unsqueeze(1)

        fail_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, fail_mask, 1, device)

        return (embeds, torch.ones(batch_size, 1).to(device))


class T5Conditioner(Conditioner):
    """Text conditioner using a frozen T5 encoder with optional gradient passthrough."""

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
                 "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
                 "google/flan-t5-xl", "google/flan-t5-xxl"]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: int = 128,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import T5EncoderModel

        self.max_length = max_length
        self.enable_grad = enable_grad

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str], drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)

        if self.training and drop_prob > 0:
            rand_vals = torch.rand(batch_size, generator=self.rng)
            texts = [
                "" if rand_vals[i] < drop_prob else t
                for i, t in enumerate(texts)
            ]

        self._ensure_on_device(self.model, device)
        self._ensure_on_device(self.proj_out, device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.amp.autocast('cuda', dtype=torch.float16), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return (embeddings, attention_mask)


class GlapTextConditioner(Conditioner):
    """Text conditioner using a pre-trained GLAP (music-language) model."""

    def __init__(self, d_model, glap_model, output_dim: int, **kwargs):
        super().__init__(dim=d_model, output_dim=output_dim)
        self.glap_model = glap_model

    def forward(self, texts, device, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(texts)

        if self.training and drop_prob > 0:
            rand_vals = torch.rand(batch_size, generator=self.rng)
            texts = [
                "" if rand_vals[i] < drop_prob else t
                for i, t in enumerate(texts)
            ]

        texts = ["" if t is None else t for t in texts]

        self._ensure_on_device(self.glap_model, device)
        embeds = self.glap_model(texts, device)
        embeds = self.proj_out(embeds)
        mask = torch.ones(batch_size, 1).to(device)

        return (embeds, mask)


class ChordConditioner(Conditioner):
    """Encodes chord annotations into frame-level chroma embeddings."""

    def __init__(self, output_dim, sample_size, drop=False, sample_rate=44100, downsampling_ratio=2048, **kwargs):
        self.sample_size = sample_size
        self.drop = drop
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        n_chroma = 12
        super().__init__(dim=n_chroma, output_dim=output_dim, **kwargs)

    def forward(self, x, device, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        num_frames = self.sample_size // self.downsampling_ratio

        processed_list = []
        is_missing_list = []

        for item in x:
            is_missing_list.append(item is None)
            tensor = get_chord2chroma_tensor(
                chord_infos=item,
                sample_size=num_frames,
                drop=self.drop,
            )
            processed_list.append(tensor)

        batch_tensor = self._to_proj_device(torch.stack(processed_list).to(device), device)
        fail_mask = torch.tensor(is_missing_list, device=device, dtype=torch.bool)
        embeds = self.proj_out(batch_tensor)
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, fail_mask, num_frames, device)

        embeds = embeds.permute(0, 2, 1)
        return (embeds, torch.ones(batch_size, num_frames, device=device))


class BeatConditioner(Conditioner):
    """Encodes beat annotations into frame-level embeddings."""

    def __init__(self, output_dim, sample_size, sample_rate=44100, downsampling_ratio=2048, **kwargs):
        self.sample_size = sample_size
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        beat_channel = 1
        super().__init__(dim=beat_channel, output_dim=output_dim, **kwargs)

    def forward(self, x, device, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        num_frames = self.sample_size // self.downsampling_ratio
        batch_size = len(x)

        processed_list = []
        is_missing_list = []

        for item in x:
            is_missing_list.append(item is None)
            tensor = get_beat_tensor(beat_infos=item, sample_size=num_frames)
            processed_list.append(tensor)

        batch_tensor = self._to_proj_device(torch.stack(processed_list).to(device), device)
        fail_mask = torch.tensor(is_missing_list, device=device, dtype=torch.bool)
        embeds = self.proj_out(batch_tensor)
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, fail_mask, num_frames, device)

        embeds = embeds.permute(0, 2, 1)
        return (embeds, torch.ones(batch_size, num_frames, device=device))


class MotifMelodyConditioner(Conditioner):
    """Encodes motif melody (MIDI piano-roll) into sequence embeddings."""

    def __init__(self, output_dim, n_pitch=128, **kwargs):
        self.mode = kwargs.pop('mode', 'crs_attn')
        self.sample_rate = kwargs.pop('sample_rate', 44100)
        self.downsampling_ratio = kwargs.pop('downsampling_ratio', 2048)
        if self.mode == 'concat':
            self.sample_size = kwargs.pop("sample_size")

        self.n_pitch = n_pitch
        self.pitch_start = kwargs.pop("pitch_start", 0)
        self.pitch_end = self.pitch_start + self.n_pitch

        super().__init__(dim=n_pitch, output_dim=output_dim, **kwargs)

    def forward(self, x, device, drop_prob=0.0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        fs = self.sample_rate / self.downsampling_ratio

        if self.mode == 'concat':
            num_frames = self.sample_size // self.downsampling_ratio
            pianorolls = [midi_to_pianoroll(item, fs=fs, max_len=num_frames,
                                           pitch_start=self.pitch_start, pitch_end=self.pitch_end) for item in x]
            batched, lengths = batch_pad_pianorolls(pianorolls, self.n_pitch, device, max_len=num_frames)
        else:
            pianorolls = [midi_to_pianoroll(item, fs=fs, max_len=int(fs * 24),
                                           pitch_start=self.pitch_start, pitch_end=self.pitch_end) for item in x]
            batched, lengths = batch_pad_pianorolls(pianorolls, self.n_pitch, device)

        embeds = self._to_proj_device(batched, device)
        embeds = self.proj_out(embeds)

        fail_mask = (lengths == 0)
        if embeds.shape[1] == 0:
            embeds = torch.zeros((batch_size, 1, self.output_dim), device=device, dtype=embeds.dtype)

        seq_len = embeds.shape[1]
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, fail_mask, seq_len, device)

        if self.mode == 'concat':
            embeds = embeds.permute(0, 2, 1)
            mask = torch.ones(batch_size, num_frames, device=device)
        else:
            seq_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
            mask = (seq_ids < lengths.clamp(min=1).unsqueeze(1)).long()

        return (embeds, mask)


_DEFAULT_STRUCTURE = {
    'latent_boundaries': [[0, 0]],
    'audio_boundaries': [[0, 0]],
    'mood': [''],
    'instrument': [''],
    'rms': [None],
    'combined_text': [''],
}


class StructureConditioner(Conditioner):
    """Segment-level structure conditioner base class.

    Parses segment boundaries from structure annotations, encodes per-segment
    text via subclass-defined encode_texts(), and fills frame-level embeddings.
    """

    def __init__(self, d_model, field, sample_size, output_dim: int, **kwargs):
        super().__init__(
            dim=d_model, output_dim=output_dim,
            allow_null_emb=kwargs.get('allow_null_emb', False)
        )
        self.field = field
        self.sample_size = sample_size
        self.downsampling_ratio = kwargs.get('downsampling_ratio', 2048)
        self.compress_ratio = kwargs.get('compress_ratio', 1)
        self.update_metadata = kwargs.get('update_metadata', False)
        self.seg_drop_prob = kwargs.get('seg_drop_prob', 0.1)

    def encode_texts(self, texts: tp.List[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of texts to embeddings [N, d_model]. Must be overridden."""
        raise NotImplementedError

    def _encode_segments(self, boundaries: list, device: torch.device):
        """Extract and encode per-segment texts. Override for multi-field fusion."""
        texts = [seg for item in boundaries for seg in item[self.field]]
        is_empty = [t == '' for t in texts]
        text_embeds = self.encode_texts(texts, device)
        return text_embeds, is_empty

    def forward(self, x, device, drop_prob=0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(x)
        num_frames = self.sample_size // self.downsampling_ratio
        mask_boundaries = []
        self._train_info = {}

        boundaries = []
        is_missing_list = []
        for item in x:
            parsed_structure = get_structure_tensor_from_json(
                structure_infos=item, sample_size=num_frames,
                compress_ratio=self.compress_ratio,
                seg_drop=0.05 if self.training else 0
            )
            is_missing_list.append(parsed_structure is None)
            boundaries.append(_DEFAULT_STRUCTURE if parsed_structure is None else parsed_structure)
        fail_mask = torch.tensor(is_missing_list, device=device, dtype=torch.bool)

        output = torch.zeros((batch_size, num_frames // self.compress_ratio, self.dim), device=device)

        seg_drop_maps = []
        segment_embeds_list = []
        segment_boundaries_list = []

        with torch.amp.autocast('cuda', dtype=torch.float32), torch.inference_mode():
            text_embeds, is_empty = self._encode_segments(boundaries, device)

            j = 0
            for i in range(batch_size):
                mask_idx = []
                start_idx = j
                seg_drop_map = {}
                for lb, ab in zip(
                    boundaries[i]['latent_boundaries'],
                    boundaries[i]['audio_boundaries']
                ):
                    if is_empty[j] or lb[1] - lb[0] == 0:
                        mask_idx.append(j - start_idx)
                    seg_dropped = self.training and random.random() < self.seg_drop_prob
                    seg_drop_map[round(ab[0], 3)] = seg_dropped
                    if lb[1] - lb[0] == 0 or seg_dropped:
                        if self.allow_null_emb:
                            mask_boundaries.append([i, lb[0], lb[1]])
                        j += 1
                        continue
                    output[i][lb[0]:lb[1], :] = text_embeds[j].expand(lb[1] - lb[0], self.dim)
                    j += 1
                end_idx = j

                if self.training:
                    seg_drop_maps.append(seg_drop_map)
                    if self.update_metadata:
                        segment_boundaries_list.append(boundaries[i]['audio_boundaries'])
                        if getattr(self, 'collect_embeds', True):
                            segment_embeds_list.append((text_embeds[start_idx:end_idx], mask_idx))

            assert j == text_embeds.shape[0]

        if self.training:
            self._train_info['seg_drop_map'] = seg_drop_maps
            if self.update_metadata:
                self._train_info['segment_boundaries'] = segment_boundaries_list
                if segment_embeds_list:
                    self._train_info['segment_embeds'] = segment_embeds_list

        is_empty_mask = (output.view(batch_size, -1).abs().sum(dim=1) == 0).to(device)

        embeds = self._to_proj_device(output, device)
        embeds = self.proj_out(embeds)

        if self.allow_null_emb and len(mask_boundaries) > 0:
            for bs_idx, lb0, lb1 in mask_boundaries:
                embeds[bs_idx, lb0:lb1, :] = self.null_emb.squeeze(0).expand(lb1 - lb0, -1)

        combined_fail = fail_mask | is_empty_mask
        output_len = num_frames // self.compress_ratio
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, combined_fail, output_len, device)

        if self.compress_ratio == 1:
            embeds = embeds.permute(0, 2, 1)

        return (embeds, torch.ones(batch_size, output_len).to(device))


class QwenStructureConditioner(StructureConditioner):
    """Structure conditioner using Qwen text encoder."""

    def __init__(self, d_model, tokenizer, model, field, sample_size, output_dim, **kwargs):
        super().__init__(d_model, field, sample_size, output_dim, **kwargs)
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = kwargs.get('max_length', 8192)

    def encode_texts(self, texts, device):
        self._ensure_on_device(self.model, device)
        batch_dict = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        outputs = self.model(**batch_dict)
        return last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])


class GLAPStructureConditioner(StructureConditioner):
    """Structure conditioner using GLAP text encoder."""

    def __init__(self, d_model, glap_model, field, sample_size, output_dim, **kwargs):
        super().__init__(d_model, field, sample_size, output_dim, **kwargs)
        self.glap_model = glap_model

    def encode_texts(self, texts, device):
        self._ensure_on_device(self.glap_model, device)
        return self.glap_model(texts, device)


class MuQStructureConditioner(StructureConditioner):
    """Structure conditioner using MuQ-MuLan text encoder."""

    def __init__(self, d_model, muq_model, field, sample_size, output_dim, **kwargs):
        super().__init__(d_model, field, sample_size, output_dim, **kwargs)
        self.muq_model = muq_model

    def encode_texts(self, texts, device):
        self._ensure_on_device(self.muq_model, device)
        return self.muq_model(texts=texts)


class FusedStructureConditioner(StructureConditioner):
    """Fused structure conditioner that concatenates Qwen + GLAP embeddings."""

    def __init__(
        self,
        qwen_d_model: int,
        tokenizer,
        model,
        qwen_field: str,
        glap_d_model: int,
        glap_model,
        glap_field: str,
        sample_size: int = 65536,
        output_dim: int = 768,
        **kwargs
    ):
        fused_dim = qwen_d_model + glap_d_model

        super().__init__(
            d_model=fused_dim,
            field=qwen_field,
            sample_size=sample_size,
            output_dim=output_dim,
            **kwargs
        )

        self.qwen_field = qwen_field
        self.glap_field = glap_field
        self.qwen_d_model = qwen_d_model
        self.glap_d_model = glap_d_model

        self.qwen_encoder = QwenStructureConditioner(
            d_model=qwen_d_model,
            tokenizer=tokenizer,
            model=model,
            field=qwen_field,
            sample_size=sample_size,
            output_dim=qwen_d_model,
            project_out=False,
            **{k: v for k, v in kwargs.items() if k != 'allow_null_emb'}
        )
        self.glap_encoder = GLAPStructureConditioner(
            d_model=glap_d_model,
            glap_model=glap_model,
            field=glap_field,
            sample_size=sample_size,
            output_dim=glap_d_model,
            project_out=False,
            **{k: v for k, v in kwargs.items() if k != 'allow_null_emb'}
        )

    def encode_texts(self, texts: tp.List[str], device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def _encode_segments(self, boundaries: list, device: torch.device):
        qwen_texts = [seg for item in boundaries for seg in item[self.qwen_field]]
        glap_texts = [seg for item in boundaries for seg in item[self.glap_field]]
        assert len(qwen_texts) == len(glap_texts), \
            f"Segment count mismatch: qwen={len(qwen_texts)}, glap={len(glap_texts)}"
        is_empty = [q == '' and g == '' for q, g in zip(qwen_texts, glap_texts)]
        qwen_embeds = self.qwen_encoder.encode_texts(qwen_texts, device).unsqueeze(1)
        glap_embeds = self.glap_encoder.encode_texts(glap_texts, device)
        text_embeds = torch.cat([qwen_embeds, glap_embeds], dim=-1)
        return text_embeds, is_empty


class EmbStructureConditioner(Conditioner):
    """Embedding-based structure conditioner for fixed-vocabulary features (e.g. RMS bins)."""

    def __init__(self, d_model, output_dim, sample_size, key_name='rms',
                 bin_edges=None, seg_drop_prob=0.2, downsampling_ratio=2048, **kwargs):
        self.sample_size = sample_size
        self.key_name = key_name
        self.downsampling_ratio = downsampling_ratio
        self.seg_drop_prob = seg_drop_prob

        super().__init__(dim=d_model, output_dim=output_dim, allow_null_emb=kwargs.get('allow_null_emb', False))
        self.bin_edges = torch.tensor(bin_edges or [-27, -21, -18, -15, -12, 0], dtype=torch.float32)
        self.cond_embed = nn.Embedding(len(self.bin_edges) + 1, d_model, padding_idx=0)

    def forward(self, x, device, drop_prob=0, seg_drop_map=None) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        num_frames = self.sample_size // self.downsampling_ratio
        batch_size = len(x)

        boundaries = []
        is_missing_list = []
        for item in x:
            parsed_structure = get_structure_tensor_from_json(structure_infos=item, sample_size=num_frames)
            is_missing_list.append(parsed_structure is None)
            boundaries.append(_DEFAULT_STRUCTURE if parsed_structure is None else parsed_structure)
        fail_mask = torch.tensor(is_missing_list, device=device, dtype=torch.bool)

        self._ensure_on_device(self.cond_embed, device)
        raw_values = [seg for item in boundaries for seg in item[self.key_name]]

        clean_values = []
        none_mask = []

        for v in raw_values:
            if v is None:
                clean_values.append(0)
                none_mask.append(True)
            else:
                val = v.item() if isinstance(v, torch.Tensor) else v
                clean_values.append(val)
                none_mask.append(False)

        input_tensor = torch.tensor(clean_values, device=device)
        none_mask_tensor = torch.tensor(none_mask, device=device).unsqueeze(1)

        bin_edges = self.bin_edges.to(device=device, dtype=input_tensor.dtype)
        bin_indices = torch.bucketize(input_tensor, bin_edges, right=False)
        final_indices = bin_indices + 1
        final_indices[bin_indices == len(bin_edges)] = 0
        latent = self.cond_embed(final_indices.long())

        if any(none_mask):
            latent = latent.masked_fill(none_mask_tensor, 0.0)

        j = 0
        mask_boundaries = []
        output = torch.zeros((batch_size, num_frames, self.dim)).to(device)
        for i in range(batch_size):
            coupled_drop_map = seg_drop_map[i] if seg_drop_map is not None else None
            for lb, ab in zip(boundaries[i]['latent_boundaries'], boundaries[i]['audio_boundaries']):
                ab_key = round(ab[0], 3)
                if coupled_drop_map is not None and ab_key in coupled_drop_map:
                    seg_dropped = coupled_drop_map[ab_key]
                else:
                    seg_dropped = self.training and random.random() < self.seg_drop_prob
                if lb[1] - lb[0] == 0 or seg_dropped:
                    if self.allow_null_emb:
                        mask_boundaries.append([i, lb[0], lb[1]])
                    j += 1
                    continue
                output[i][lb[0]:lb[1], :] = latent[j].unsqueeze(0).expand(lb[1] - lb[0], self.dim)
                j += 1
        assert j == latent.shape[0]

        is_empty_mask = (output.view(batch_size, -1).abs().sum(dim=1) == 0).to(device)

        embeds = self._to_proj_device(output, device)
        embeds = self.proj_out(embeds)

        if self.allow_null_emb and len(mask_boundaries) > 0:
            for bs_idx, lb0, lb1 in mask_boundaries:
                embeds[bs_idx, lb0:lb1, :] = self.null_emb.squeeze(0).expand(lb1 - lb0, -1)

        combined_fail = fail_mask | is_empty_mask
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, combined_fail, num_frames, device)

        embeds = embeds.permute(0, 2, 1).contiguous()

        return (embeds, torch.ones(batch_size, num_frames).to(device))


class VAEStyleConditioner(Conditioner):
    """Style conditioner using VAE latent encoding of audio."""

    def __init__(self, output_dim: int, **kwargs):
        self.sample_rate = kwargs["sample_rate"]

        pt_cfg = kwargs["pretransform_config"]
        inner_ae = create_autoencoder_from_config({"sample_rate": self.sample_rate, "model": pt_cfg["config"]})
        pretransform = AutoencoderPretransform(
            inner_ae,
            scale=pt_cfg.get("scale", 1.0),
            model_half=pt_cfg.get("model_half", False),
            iterate_batch=pt_cfg.get("iterate_batch", False),
        )
        if kwargs.get("pretransform_ckpt_path") is not None:
            pretransform.load_state_dict(load_ckpt_state_dict(kwargs["pretransform_ckpt_path"]))

        pretransform.eval()
        pretransform.requires_grad_(False)

        super().__init__(pretransform.encoded_channels, output_dim, allow_null_emb=kwargs.get('allow_null_emb', False))
        self.pretransform = pretransform

        self.downsample = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x, device, drop_prob=0):
        batch_size = len(x)

        processed_list = []
        is_missing_list = []

        audio_len = int(self.sample_rate * 10)
        input_channels = 2

        for item in x:
            audio_tensor = get_style_prompt(item, sample_rate=self.sample_rate)
            is_missing_list.append(audio_tensor is None)
            processed_list.append(torch.zeros((input_channels, audio_len)) if audio_tensor is None else audio_tensor)

        batch_tensor = torch.stack(processed_list).to(device)
        fail_mask = torch.tensor(is_missing_list, device=device, dtype=torch.bool)

        with torch.no_grad():
            latent = self.pretransform.encode(batch_tensor)

        latent = self.downsample(latent).permute(0, 2, 1)

        embeds = self._to_proj_device(latent, device)
        embeds = self.proj_out(embeds)

        num_frames = embeds.shape[1]
        embeds = self._apply_dropout_mask(embeds, batch_size, drop_prob, fail_mask, num_frames, device)

        return (embeds, torch.ones(batch_size, num_frames, device=device))


class BarPositions(nn.Module):
    """Provides bar-level positional encoding from beat annotations."""

    def __init__(self, sample_size, patch_size=1, downsampling_ratio=2048, seed=-1, **kwargs):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.downsampling_ratio = downsampling_ratio

        self.rng = torch.Generator()
        seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
        self.rng.manual_seed(seed)

    def forward(self, x, device, drop_prob=0) -> tp.Optional[torch.Tensor]:
        if self.training and torch.rand(1, generator=self.rng).item() < drop_prob:
            return None

        num_frames = self.sample_size // self.downsampling_ratio // self.patch_size
        if not self.training and x[0] is None:
            return None

        results = [get_bar_pos(beat_infos=cond, sample_size=num_frames) for cond in x]
        if None in results:
            return None

        bar_pos = torch.stack([r[0] for r in results], dim=0).to(device)
        if (bar_pos == 0).all(dim=-1).any():
            return None
        return bar_pos


class BeatPositions(nn.Module):
    """Provides beat-level positional encoding from beat annotations."""

    def __init__(self, sample_size, patch_size=1, downsampling_ratio=2048, seed=-1):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.downsampling_ratio = downsampling_ratio

        self.rng = torch.Generator()
        seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
        self.rng.manual_seed(seed)

    def forward(self, x, device, drop_prob=0) -> tp.Optional[torch.Tensor]:
        if self.training and torch.rand(1, generator=self.rng).item() < drop_prob:
            return None

        num_frames = self.sample_size // self.downsampling_ratio // self.patch_size
        if not self.training and x[0] is None:
            return None

        results = [get_beat_pos(beat_infos=cond, sample_size=num_frames) for cond in x]
        if None in results:
            return None

        beat_pos = torch.stack(results, dim=0).to(device)
        if (beat_pos == 0).all(dim=-1).any():
            return None
        return beat_pos


class MultiConditioner(nn.Module):
    """Applies multiple conditioners to a batch of metadata dicts.

    Resolves each conditioner's key from the metadata, handles coupled
    bar/beat dropout, and propagates seg_drop_map between structure conditioners.
    """

    _KEY_MAP = {
        'bar_pos': 'beat',
        'beat_pos': 'beat',
        'glap-text': 'prompt',
        'struct-combined-text': 'structure',
        'struct-rms': 'structure',
    }

    def __init__(self, conditioners: tp.Dict[str, Conditioner],
                 drop_prob: tp.Optional[tp.Dict[str, float]] = None):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.drop_prob = drop_prob or {}

    def _resolve_key(self, key: str, metadata: dict) -> str:
        """Resolve the metadata key for a given conditioner key."""
        if key in metadata:
            return key
        if key in self._KEY_MAP:
            mapped = self._KEY_MAP[key]
            if mapped in metadata:
                return mapped
        raise ValueError(f"Conditioner key '{key}' not found in batch metadata")

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]],
                device: tp.Union[torch.device, str]) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        output = {}
        training_outputs = {}

        pos_keys = {'bar_pos', 'beat_pos'}
        pos_drop_prob = max((self.drop_prob.get(k, 0) for k in pos_keys if k in self.conditioners), default=0)
        pos_should_drop = torch.rand(1).item() < pos_drop_prob if pos_drop_prob > 0 else False

        for key, conditioner in self.conditioners.items():
            conditioner_inputs = []
            for x in batch_metadata:
                condition_key = self._resolve_key(key, x)
                val = x[condition_key]
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                conditioner_inputs.append(val)

            drop = (1.0 if pos_should_drop else 0) if key in pos_keys else self.drop_prob.get(key, 0)

            extra_kwargs = {}
            if isinstance(conditioner, EmbStructureConditioner):
                for prev_cond in self.conditioners.values():
                    info = getattr(prev_cond, '_train_info', None)
                    if info and 'seg_drop_map' in info:
                        extra_kwargs['seg_drop_map'] = info['seg_drop_map']
                        break

            output[key] = conditioner(conditioner_inputs, device, drop, **extra_kwargs)

            train_info = getattr(conditioner, '_train_info', None)
            if train_info:
                training_outputs[key] = train_info

        return output, training_outputs


CONDITIONER_REGISTRY: tp.Dict[str, type] = {
    "t5": T5Conditioner,
    "glap-text": GlapTextConditioner,
    "chord": ChordConditioner,
    "beat": BeatConditioner,
    "motif": MotifMelodyConditioner,
    "vae-style": VAEStyleConditioner,
    "qwen-structure": QwenStructureConditioner,
    "glap-structure": GLAPStructureConditioner,
    "muq-structure": MuQStructureConditioner,
    "qwen-glap-fuse-structure": FusedStructureConditioner,
    "emb-structure": EmbStructureConditioner,
    "music_key": MusicKeyConditioner,
    "bar_pos": BarPositions,
    "beat_pos": BeatPositions,
    "int": IntConditioner,
    "number": NumberConditioner,
}

_SHARED_DEPS: tp.Dict[str, tp.Dict[str, tp.Callable]] = {
    "qwen-structure":          {"tokenizer": "qwen_tokenizer", "model": "qwen_model"},
    "qwen-glap-fuse-structure": {"tokenizer": "qwen_tokenizer", "model": "qwen_model", "glap_model": "glap_model"},
    "muq-structure":           {"muq_model": "muq_model"},
    "glap-text":               {"glap_model": "glap_model"},
    "glap-structure":          {"glap_model": "glap_model"},
}

_SHARED_FACTORIES: tp.Dict[str, tp.Callable] = {
    "qwen_tokenizer": lambda: AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left'),
    "qwen_model": lambda: AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B').eval().requires_grad_(False),
    "muq_model": lambda: MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").eval().requires_grad_(False),
    "glap_model": lambda: GlapBaseBert().eval().requires_grad_(False),
}

_POS_TYPES = {"bar_pos", "beat_pos"}
_MINIMAL_CONFIG_TYPES = {"t5"}


def create_multi_conditioner(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """Create a MultiConditioner from a conditioning config dictionary."""
    cond_dim = config["cond_dim"]
    sample_rate = config.get("sample_rate", 44100)
    sample_size = config.get("sample_size")
    downsampling_ratio = config.get("downsampling_ratio", 2048)
    drop_prob = config.get("drop_prob", {'prompt': 0.1})

    shared = {}
    conditioners = {}

    for conditioner_info in config["configs"]:
        cid = conditioner_info["id"]
        ctype = conditioner_info["type"]

        if ctype in _POS_TYPES:
            kwargs = {**conditioner_info["config"], "downsampling_ratio": downsampling_ratio, "sample_size": sample_size}
        elif ctype in _MINIMAL_CONFIG_TYPES:
            kwargs = {**conditioner_info["config"], "output_dim": cond_dim}
        else:
            kwargs = {"output_dim": cond_dim, "sample_rate": sample_rate, "downsampling_ratio": downsampling_ratio}
            kwargs.update(conditioner_info["config"])
            if sample_size is not None:
                kwargs.setdefault("sample_size", sample_size)

        for param_name, shared_key in _SHARED_DEPS.get(ctype, {}).items():
            if shared_key not in shared:
                shared[shared_key] = _SHARED_FACTORIES[shared_key]()
            kwargs[param_name] = shared[shared_key]

        cls = CONDITIONER_REGISTRY.get(ctype)
        if cls is None:
            raise ValueError(f"Unknown conditioner type: {ctype}")
        conditioners[cid] = cls(**kwargs)

    return MultiConditioner(conditioners, drop_prob=drop_prob)


create_multi_conditioner_from_conditioning_config = create_multi_conditioner
