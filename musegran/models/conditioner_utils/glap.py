"""GLAP (Generalized Language-Audio Pretraining) encoder wrapper.

Uses the public HuggingFace model: mispeech/GLAP (Apache 2.0)
Weights are cached locally by transformers on first download.
"""

from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel


class GlapBaseBert(torch.nn.Module):
    """GLAP encoder for text and audio embedding extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.glap = AutoModel.from_pretrained("mispeech/GLAP", trust_remote_code=True)

    def forward(self, text_input: Sequence[str], device: torch.device) -> torch.Tensor:
        # Upstream encode_text hardcodes padded_ids on CPU — reimplement with
        # proper device placement.
        glap = self.glap
        tokenizer = glap._get_tokenizer()
        encoder_fn = tokenizer.create_encoder(lang="eng_Latn")

        all_token_ids: List[List[int]] = []
        max_seq_len = glap.config.text_max_seq_len
        for t in text_input:
            all_token_ids.append(encoder_fn(t)[:max_seq_len])

        max_len = max(len(ids) for ids in all_token_ids) if all_token_ids else 0
        batch_size = len(all_token_ids)

        padded_ids = torch.full((batch_size, max_len), tokenizer.pad_idx, dtype=torch.long, device=device)
        padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(all_token_ids):
            padded_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            padding_mask[i, len(ids):] = True

        with torch.no_grad():
            sentence_embeddings = glap.text_encoder(padded_ids, padding_mask)
        text_embeds = F.normalize(glap.text_proj(sentence_embeddings), dim=-1)
        return text_embeds.unsqueeze(1)
