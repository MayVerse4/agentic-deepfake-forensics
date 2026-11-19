"""
Inference script for the Frequency Model (Phase 1).
Loads a trained checkpoint and evaluates a single face crop image.
"""
import os
import sys
import argparse
import cv2
import torch

# Ensure we can import from preprocessing and models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "preprocessing"))

from frequency_model import FrequencyModel  # type: ignore
from frequency_generator import FrequencyGenerator  # type: ignore

def predict(image_path, checkpoint_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = FrequencyModel().to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
        model.eval()

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Preprocess
    freq_gen = FrequencyGenerator()
    x_h = freq_gen.compute_high_freq_image(frame) # (H, W, 3), 0-1
    x_h = torch.from_numpy(x_h).permute(2, 0, 1).float().unsqueeze(0).to(device) # -> (1, 3, H, W)
    
    # Predict
    with torch.no_grad():
        logit = model(x_h)
        score = torch.sigmoid(logit).item()
        
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single face crop.")
    parser.add_argument("image_path", help="Path to the face crop image")
    parser.add_argument("--checkpoint", default="checkpoints/frequency_model_best.pth", help="Path to the model checkpoint")
    args = parser.parse_args()
    
    score = predict(args.image_path, args.checkpoint)
    print(f"Image: {args.image_path}")
    print(f"Fake Probability Score: {score:.4f} (1.0 = Fake, 0.0 = Real)")
    if score > 0.5:
        print("Prediction: FAKE")
    else:
        print("Prediction: REAL")
