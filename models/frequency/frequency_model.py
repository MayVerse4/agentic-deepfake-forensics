"""
Frequency Model (Phase 1) -- FreqNet-inspired architecture.

Based on: Tan et al., "Frequency-Aware Deepfake Detection: Improving
Generalizability through Frequency Space Learning" (AAAI 2024,
arXiv:2403.07240). Reference implementation: github.com/chuangchuangtan/
FreqNet-DeepfakeDetection. This file is our own implementation of the
paper's described architecture (equations 3-7), not a copy of their code.

Why this architecture, not a bigger pretrained backbone:
- 1.9M-parameter FreqNet beat a 304M-parameter competitor by 9.8% mean
  accuracy on unseen generation sources -- exactly the generalization
  problem a small student-project dataset will run into.
- Trained from scratch on purpose: spectral data has different statistics
  from natural images, so ImageNet pretraining doesn't transfer well here.
- Small enough to train on CPU/a single consumer GPU in reasonable time,
  which matters for a project without a dedicated GPU cluster.

Three modules from the paper:
  1. HFRI  (High-Frequency Representation of Image)  -- input preprocessing
  2. HFRF  (High-Frequency Representation of Feature) -- inside the network
  3. FCL   (Frequency Convolutional Layer)            -- inside the network
"""

import torch
import torch.nn as nn


# =============================================================
# HFRF: High-frequency filtering of FEATURE MAPS (spatial variant)
# =============================================================
class SpatialHighFreqFilter(nn.Module):
    """
    Zeroes out the low-frequency (center) region of a feature map's 2D FFT,
    keeping only high-frequency content -- per Eq. 4 in the paper, using
    the same W/4, H/4 center-zeroing radius.
    """

    def __init__(self, ratio=0.25):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        # x: (B, C, H, W)
        f = torch.fft.fft2(x, dim=(-2, -1))
        f_shifted = torch.fft.fftshift(f, dim=(-2, -1))

        H, W = x.shape[-2], x.shape[-1]
        cy, cx = H // 2, W // 2
        ry, rx = max(1, int(H * self.ratio)), max(1, int(W * self.ratio))

        mask = torch.ones_like(f_shifted.real)
        mask[..., max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx] = 0

        f_filtered = f_shifted * mask
        f_unshifted = torch.fft.ifftshift(f_filtered, dim=(-2, -1))
        return torch.fft.ifft2(f_unshifted, dim=(-2, -1)).real


# =============================================================
# HFRF: High-frequency filtering along the CHANNEL dimension
# =============================================================
class ChannelHighFreqFilter(nn.Module):
    """
    Same idea as SpatialHighFreqFilter but applies the FFT/high-pass along
    the channel dimension instead of spatial dimensions -- the paper runs
    both variants at different network depths (Eq. 6).
    """

    def __init__(self, ratio=0.25):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        # x: (B, C, H, W) -> FFT along channel dim (dim=1)
        f = torch.fft.fft(x, dim=1)
        f_shifted = torch.fft.fftshift(f, dim=1)

        C = x.shape[1]
        c_center = C // 2
        r = max(1, int(C * self.ratio))

        mask = torch.ones_like(f_shifted.real)
        mask[:, max(0, c_center - r):c_center + r, :, :] = 0

        f_filtered = f_shifted * mask
        f_unshifted = torch.fft.ifftshift(f_filtered, dim=1)
        return torch.fft.ifft(f_unshifted, dim=1).real


# =============================================================
# FCL: Frequency Convolutional Layer
# =============================================================
class FrequencyConvLayer(nn.Module):
    """
    Transforms a feature map to the frequency domain, applies a SEPARATE
    conv layer to the amplitude spectrum and the phase spectrum (Eq. 7),
    then transforms back with iFFT. This is what makes the network learn
    IN the frequency domain, not just FROM frequency-domain input.
    """

    def __init__(self, channels):
        super().__init__()
        self.amp_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.phase_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        f = torch.fft.fft2(x, dim=(-2, -1))
        amplitude = torch.abs(f)
        phase = torch.angle(f)

        amplitude = self.amp_conv(amplitude)
        phase = self.phase_conv(phase)

        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        f_new = torch.complex(real, imag)

        return torch.fft.ifft2(f_new, dim=(-2, -1)).real


# =============================================================
# Lightweight residual block (no pretraining, per the paper's philosophy)
# =============================================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# =============================================================
# FreqNet-lite: full model
# =============================================================
class FrequencyModel(nn.Module):
    """
    Input: x_h -- the HFRI output (3, 224, 224), i.e. the high-frequency-
    filtered image, NOT the raw magnitude spectrum. See
    frequency_generator.compute_high_freq_image() for how this is produced
    upstream during preprocessing.

    Output: a single real/fake probability (sigmoid). This model IS allowed
    to classify -- it's a Phase 1 model, not preprocessing. Its raw score
    also gets consumed as one signal by the Fusion Network later.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block1 = ResBlock(32, 64, stride=2)     # 224 -> 112
        self.hfrf_spatial = SpatialHighFreqFilter()

        self.block2 = ResBlock(64, 128, stride=2)    # 112 -> 56
        self.hfrf_channel = ChannelHighFreqFilter()

        self.fcl = FrequencyConvLayer(128)

        self.block3 = ResBlock(128, 256, stride=2)   # 56 -> 28

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_h):
        x = self.stem(x_h)

        x = self.block1(x)
        x = x + self.hfrf_spatial(x)     # keep high-freq features prominent

        x = self.block2(x)
        x = x + self.hfrf_channel(x)

        x = self.fcl(x)                  # learn directly in frequency space

        x = self.block3(x)
        x = self.pool(x)

        logit = self.classifier(x)
        return logit.squeeze(-1)         # raw logit; apply sigmoid outside

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================
# ResNet18 on the same HFRI input
# =============================================================
class FrequencyResNet(nn.Module):
    """
    Same input as FrequencyModel -- the high-pass-filtered image (HFRI), so
    the model still only sees high-frequency content -- but read by an
    ImageNet-pretrained ResNet18 instead of FreqNet-lite.

    Why it exists: on FF++ face crops FreqNet-lite plateaued at ~0.62 val
    AUC however the input was normalized, while this backbone reached 0.85
    after 2 epochs on the identical input (ablation, 2026-09-30). Pretraining
    helps even on spectral input because the early filters are generic edge /
    texture detectors, which is exactly what high-pass residuals consist of.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x_h):
        return self.backbone(x_h).squeeze(-1)  # raw logit; apply sigmoid outside

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def compute_high_freq_image_gpu(frames, size=224, mask_radius=0.25, norm="minmax"):
    """
    HFRI on a batch: frames (B, 3, H, W) float RGB in 0-255 -> high-pass image.
    Single source of truth for training AND inference; the numpy version in
    preprocessing/frequency_generator.py drifted (radius 0.15 vs 0.25) before.

    norm='minmax' squeezes each image into [0, 1] using its extreme values. The
    FFT high-pass leaves large ringing spikes at the image borders, so min-max
    lets those spikes set the range and the face content ends up in a ~3% band
    around 0.5 (per-image pixel std ~0.034). norm='std' standardizes each image
    to zero mean / unit variance instead, which keeps the face content usable.
    """
    f = torch.fft.fft2(frames, dim=(-2, -1))
    f_shifted = torch.fft.fftshift(f, dim=(-2, -1))

    h, w = size, size
    cy, cx = h // 2, w // 2
    ry, rx = int(h * mask_radius), int(w * mask_radius)

    f_shifted[:, :, cy - ry:cy + ry, cx - rx:cx + rx] = 0

    f_unshifted = torch.fft.ifftshift(f_shifted, dim=(-2, -1))
    x_h = torch.fft.ifft2(f_unshifted, dim=(-2, -1)).real

    batch_size = x_h.shape[0]
    x_h_view = x_h.reshape(batch_size, -1)
    if norm == "std":
        mean = x_h_view.mean(dim=1).view(batch_size, 1, 1, 1)
        std = x_h_view.std(dim=1).view(batch_size, 1, 1, 1)
        return (x_h - mean) / (std + 1e-8)
    x_h_min = x_h_view.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    x_h_max = x_h_view.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    return (x_h - x_h_min) / (x_h_max - x_h_min + 1e-8)


def load_frequency_model(path, device):
    """Loads either architecture; the arch is read from the checkpoint's keys."""
    state = torch.load(path, map_location=device)
    state = state.get("model_state_dict", state)
    arch = "resnet18" if any(k.startswith("backbone.") for k in state) else "freqnet"
    model = build_frequency_model(arch, pretrained=False).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_frequency_model(arch="freqnet", pretrained=True):
    """arch: 'freqnet' (FreqNet-lite, trained from scratch) or 'resnet18'."""
    if arch == "freqnet":
        return FrequencyModel()
    if arch == "resnet18":
        return FrequencyResNet(pretrained=pretrained)
    raise ValueError(f"Unknown frequency arch: {arch}")


if __name__ == "__main__":
    model = FrequencyModel()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {model.num_parameters:,}")
