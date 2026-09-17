"""Audiobox Aesthetics scorer wrapper.

Requires the official audiobox-aesthetics package:
    git clone https://github.com/facebookresearch/audiobox-aesthetics
    cd audiobox-aesthetics && pip install -e .
"""

import torch

from audiobox_aesthetics.infer import AesPredictor


class Scorer:
    """Audiobox Aesthetics scorer.

    Evaluates audio quality on axes: CE (Colorfulness/Expressiveness),
    CU (Clarity/Understanding), PC (Production Complexity), PQ (Production Quality).
    """

    def __init__(self, device=None, checkpoint_path=None):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.predictor = AesPredictor(checkpoint_pth=checkpoint_path, data_col="path")
        self.predictor.model.to(self.device)

    @torch.no_grad()
    def forward(self, input_infos):
        """Score a list of audio segments.

        Args:
            input_infos: list of {"path": str, "start_time": float, "end_time": float}

        Returns:
            list of dicts with keys: CE, CU, PC, PQ (float scores)
        """
        return self.predictor.forward(input_infos)
