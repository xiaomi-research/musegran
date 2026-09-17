"""SongEval scorer wrapper.

Requires:
    1. The official SongEval repo cloned into this directory:
       git clone https://github.com/QiaoHezworworworworworkkkk/SongEval score_tools/songeval/SongEval
    2. pip install muq

Place the checkpoint at score_tools/songeval/SongEval/ckpt/model.safetensors
"""

import os

import librosa
import numpy as np
import torch
from muq import MuQ
from safetensors.torch import load_file
from tqdm import tqdm

_SONGEVAL_DIR = os.path.join(os.path.dirname(__file__), "SongEval")
DEFAULT_CKPT = os.path.join(_SONGEVAL_DIR, "ckpt", "model.safetensors")


def _load_generator(device):
    """Instantiate the SongEval Generator model from the official repo."""
    import sys
    sys.path.insert(0, _SONGEVAL_DIR)
    from model import Generator
    sys.path.pop(0)

    model = Generator(
        in_features=1024,
        ffd_hidden_size=4096,
        num_classes=5,
        attn_layer_num=4,
    ).to(device).eval()
    return model


class Scorer:
    """SongEval scorer: evaluates audio on Coherence, Musicality, Memorability, Clarity, Naturalness."""

    def __init__(self, device=None, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or DEFAULT_CKPT
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._setup()

    @torch.no_grad()
    def _setup(self):
        model = _load_generator(self.device)
        state_dict = load_file(self.checkpoint_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)

        self.model = model
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = self.muq.to(self.device).eval()

    @torch.no_grad()
    def forward(self, input_infos):
        """Score a list of audio segments.

        Args:
            input_infos: list of {"path": str, "start_time": float, "end_time": float}

        Returns:
            list of dicts with keys: Coherence, Musicality, Memorability, Clarity, Naturalness
        """
        results = []
        for item in tqdm(input_infos):
            score = self._score_one(item["path"], item.get("start_time"), item.get("end_time"))
            if score is not None:
                results.append(score)
        return results

    @torch.no_grad()
    def _score_one(self, input_path, start_time=None, end_time=None):
        if start_time is None:
            offset = 0.0
            duration = None
        else:
            offset = start_time
            duration = end_time - start_time
            if duration <= 0:
                return None

        try:
            wav, _ = librosa.load(input_path, sr=24000, offset=offset, duration=duration)
        except Exception:
            return None

        audio = torch.tensor(wav).unsqueeze(0).to(self.device)
        output = self.muq(audio, output_hidden_states=True)
        input_data = output["hidden_states"][6]

        scores = self.model(input_data).squeeze(0)
        return {
            'Coherence': round(scores[0].item(), 4),
            'Musicality': round(scores[1].item(), 4),
            'Memorability': round(scores[2].item(), 4),
            'Clarity': round(scores[3].item(), 4),
            'Naturalness': round(scores[4].item(), 4),
        }
