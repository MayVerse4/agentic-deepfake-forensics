"""
Temporal Model (Phase 2) -- v2: SIMPLIFIED architecture.

Changed after repeated training instability with the original custom
CNN + cross-attention design (documented across several debugging
rounds: pos_weight direction, LR/warmup, BatchNorm->GroupNorm -- each
fix bought partial stability, none held cleanly at full scale).

Two changes, backed by the deepfake-detection literature rather than
guesswork this time:

1. FrameCNN is now a PRETRAINED ResNet18, not a from-scratch custom CNN.
   This is the standard, proven choice in the field (ResNeXt50+LSTM,
   EfficientNet-B0+BiLSTM, etc. all report 90-99% on FF++/Celeb-DF) --
   unlike the Frequency Model, where training from scratch was correct
   because spectral/FFT input doesn't match natural-image statistics,
   RGB video frames ARE natural images, exactly what ImageNet
   pretraining is built for. Using a from-scratch CNN here was fighting
   the literature, not following it.

2. Fusion is now simple CONCATENATION, not cross-attention. Cross-
   attention was the recurring source of instability today. Under time
   pressure, a stable, simple baseline that actually finishes training
   beats a more sophisticated mechanism that doesn't. The temporal-
   frequency anomaly signal (the novelty) is still used -- its embedding
   is concatenated onto the LSTM's final hidden state before
   classification -- just fused more simply than before.

If time remains after this baseline trains cleanly, cross-attention
fusion can be reintroduced on TOP of this stable foundation, rather than
being the thing blocking any working result at all.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ImageNet normalization stats -- required since FrameCNN now uses
# pretrained weights, which expect input normalized this way, not just
# scaled to 0-1 (which is what train_temporal_model.py's dataset provides).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# =============================================================
# Pretrained ResNet18 per-frame feature extractor
# =============================================================
class FrameCNN(nn.Module):
    def __init__(self, embed_dim=128, freeze_backbone=False):
        super().__init__()
        self.freeze_backbone_flag = freeze_backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Drop the final classification layer -- keep everything up to
        # the 512-dim pooled feature.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def unfreeze_backbone(self, top_layers_only=True):
        """
        Unfreezes backbone parameters. If top_layers_only is True,
        unfreezes layer3 and layer4 of ResNet18 while keeping early stem layers frozen.
        """
        if top_layers_only:
            for i, child in enumerate(self.backbone):
                if i >= 6:  # layer3 (idx 6) and layer4 (idx 7)
                    for p in child.parameters():
                        p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def forward(self, x):
        # x: (B*T, 3, H, W), expected 0-1 normalized RGB (from the dataset)
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
        x = (x - mean) / std

        feat = self.backbone(x).flatten(1)  # (B*T, 512)
        return self.norm(self.proj(feat))   # (B*T, embed_dim) normalized


# =============================================================
# Small CNN over the temporal-frequency anomaly map
# =============================================================
class AnomalyCNN(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GroupNorm(8, 16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.norm(self.proj(feat))


# =============================================================
# Temporal Model: pretrained FrameCNN -> LSTM, optionally concatenated
# with AnomalyCNN's embedding (use_anomaly=False for bare RGB+LSTM Option B)
# =============================================================
class TemporalModel(nn.Module):
    def __init__(self, embed_dim=128, lstm_hidden=128, freeze_backbone=False, use_anomaly=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_anomaly = use_anomaly

        self.frame_cnn = FrameCNN(embed_dim, freeze_backbone=freeze_backbone)
        
        if self.use_anomaly:
            self.anomaly_cnn = AnomalyCNN(embed_dim)
            in_dim = lstm_hidden + embed_dim
        else:
            self.anomaly_cnn = None
            in_dim = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=lstm_hidden,
            batch_first=True, bidirectional=False
        )

        self.fuse_norm = nn.LayerNorm(in_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        if self.frame_cnn.freeze_backbone_flag:
            self.frame_cnn.backbone.eval()
        return self

    def unfreeze_backbone(self, top_layers_only=True):
        self.frame_cnn.unfreeze_backbone(top_layers_only=top_layers_only)

    def forward(self, frames, anomaly_map=None):
        """
        frames:       (B, T, 3, H, W) -- RGB frame sequence, 0-1 normalized
        anomaly_map:  (B, 1, H, W)    -- optional temporal-frequency anomaly map
        """
        B, T, C, H, W = frames.shape

        frame_embeds = self.frame_cnn(frames.view(B * T, C, H, W))
        frame_embeds = frame_embeds.view(B, T, self.embed_dim)

        lstm_out, (h_n, c_n) = self.lstm(frame_embeds)
        h_lstm = h_n[-1]  # (B, lstm_hidden)

        if self.use_anomaly and anomaly_map is not None:
            anomaly_embed = self.anomaly_cnn(anomaly_map)  # (B, embed_dim)
            fused = torch.cat([h_lstm, anomaly_embed], dim=1)
        else:
            anomaly_embed = None
            fused = h_lstm

        fused_norm = self.fuse_norm(fused)
        logit = self.classifier(fused_norm)

        return logit.squeeze(-1), anomaly_embed  # second value kept for interface compatibility

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TemporalModel()
    dummy_frames = torch.rand(2, 16, 3, 224, 224)   # 0-1 range, like the real dataset
    dummy_anomaly = torch.randn(2, 1, 224, 224)

    logit, _ = model(dummy_frames, dummy_anomaly)
    print(f"Output shape: {logit.shape}")
    print(f"Total parameters: {model.num_parameters:,}")
    print(f"Trainable parameters: {model.num_trainable_parameters:,}")
