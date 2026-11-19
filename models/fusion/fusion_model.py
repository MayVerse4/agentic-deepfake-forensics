"""
Fusion Model -- the real thing that combined_prob_fake_UNCALIBRATED was
always a placeholder for.

Takes the trained evidence models' scores (Spatial, Frequency, Temporal
probabilities) plus a flag for whether Temporal was even available (still
images and short/untracked clips have no temporal score), and learns a
calibrated combination instead of an unweighted mean.

Two input layouts, picked with use_frequency:
  use_frequency=False -> [spatial, temporal, temporal_available]
      (fusion_model_best.pth, 2026-09-24: trained while the FreqNet-lite
      frequency model was collapsed at ~0.5 AUC, so frequency was dropped)
  use_frequency=True  -> [spatial, frequency, temporal, temporal_available]
      (for the retrained frequency model)

Deliberately small: 3-4 input features, not images -- this is a tabular
problem, not a vision problem. A deep network would just overfit the
few thousand rows this generates.
"""

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, hidden_dim=16, use_frequency=False):
        super().__init__()
        self.use_frequency = use_frequency
        in_dim = 4 if use_frequency else 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """x: (B, 3 or 4) -- see build_input"""
        return self.net(x).squeeze(-1)  # (B,) logits

    @staticmethod
    def build_input(spatial, frequency, temporal, use_frequency=False):
        """
        spatial, frequency: float in [0, 1]
        temporal: float in [0, 1] or None
        use_frequency: must match the model the input is for (model.use_frequency)
        Returns a list ready to stack into a tensor.
        """
        temporal_available = 1.0 if temporal is not None else 0.0
        temporal_value = temporal if temporal is not None else 0.5  # neutral filler, flag carries the real signal
        if use_frequency:
            return [spatial, frequency, temporal_value, temporal_available]
        return [spatial, temporal_value, temporal_available]

    @classmethod
    def load(cls, path, device, hidden_dim=16):
        """Loads either layout; which one is read from the first layer's width."""
        state = torch.load(path, map_location=device)
        use_frequency = state["net.0.weight"].shape[1] == 4
        model = cls(hidden_dim=hidden_dim, use_frequency=use_frequency).to(device)
        model.load_state_dict(state)
        model.eval()
        return model

