"""
Spatial Model (Track A) -- EfficientNet-B0, fine-tuned in two phases on
Kaggle (frozen-head warmup, then partial backbone unfreeze). Reaches
~89.5% validation AUC.

Moved here unchanged from Track A's spatial_model_inference.py so all
three evidence models live under models/<name>/ consistently.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class DeepfakeSpatialModel(nn.Module):
    """Must match the Kaggle training architecture exactly, or state_dict loading fails."""

    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


class SpatialModelInference:
    """
    Loads the trained Spatial CNN and exposes predict() for a single face
    crop. CPU-capable -- no GPU required for single-image inference.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, weights_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepfakeSpatialModel()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def predict(self, face_crop_bgr):
        """
        face_crop_bgr: numpy array, shape (224, 224, 3), BGR -- exactly what
        face_cropper.py produces. Returns P(fake) in [0, 1].
        """
        rgb = face_crop_bgr[:, :, ::-1]  # BGR -> RGB
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()

        return probability
