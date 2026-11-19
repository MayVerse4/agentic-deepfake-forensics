"""
Training script for Frequency Model with Data Augmentation (Horizontal Flip).

Kept strictly SEPARATE from the baseline training script (train_frequency_model.py)
to protect existing baseline weights (frequency_model_best.pth).

Saved Checkpoints:
  - checkpoints/frequency_model_aug_best.pth
  - checkpoints/frequency_model_aug_final.pth
  - checkpoints/frequency_model_aug_resume.pth
"""

import os
import sys
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "preprocessing"))
from frequency_generator import FrequencyGenerator  # type: ignore # noqa: E402


class FrequencyDatasetAugmented(Dataset):
    """
    Loads real/fake face images from a split folder containing Real/ and
    Fake/ subfolders.
    
    Applies Random Horizontal Flip to the spatial image BEFORE computing HFRI.
    """

    def __init__(self, split_dir, augment=True):
        self.freq_gen = FrequencyGenerator()
        self.samples = []
        self.augment = augment

        real_dir = os.path.join(split_dir, "Real")
        fake_dir = os.path.join(split_dir, "Fake")

        if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
            raise RuntimeError(
                f"Expected {split_dir}/Real and {split_dir}/Fake to exist."
            )

        for path in os.listdir(real_dir):
            self.samples.append((os.path.join(real_dir, path), 0.0))
        for path in os.listdir(fake_dir):
            self.samples.append((os.path.join(fake_dir, path), 1.0))

        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}/Real or {split_dir}/Fake.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frame = cv2.imread(path)
        
        resized = cv2.resize(frame, (224, 224))

        # Safe frequency-domain augmentation: Random Horizontal Flip on spatial RGB image
        if self.augment and np.random.rand() > 0.5:
            resized = cv2.flip(resized, 1)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float()  # -> (3, H, W)

        return x, torch.tensor(label, dtype=torch.float32)


def compute_high_freq_image_gpu(frames, size=224):
    """
    Produces the high-pass-filtered image on the GPU for an entire batch.
    Matches preprocessing/frequency_generator.py logic exactly.
    """
    f = torch.fft.fft2(frames, dim=(-2, -1))
    f_shifted = torch.fft.fftshift(f, dim=(-2, -1))

    h, w = size, size
    cy, cx = h // 2, w // 2
    ry, rx = h // 4, w // 4

    f_shifted[:, :, cy - ry:cy + ry, cx - rx:cx + rx] = 0

    f_unshifted = torch.fft.ifftshift(f_shifted, dim=(-2, -1))
    x_h = torch.fft.ifft2(f_unshifted, dim=(-2, -1)).real

    # Normalize per image in the batch
    batch_size = x_h.shape[0]
    x_h_view = x_h.view(batch_size, -1)
    x_h_min = x_h_view.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    x_h_max = x_h_view.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    
    x_h = (x_h - x_h_min) / (x_h_max - x_h_min + 1e-8)
    return x_h


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    n = 0

    with torch.no_grad():
        for x, labels in tqdm(loader, desc="Validation"):
            x, labels = x.to(device), labels.to(device)
            x_h = compute_high_freq_image_gpu(x)
            logits = model(x_h)
            loss = criterion(logits, labels)

            total_loss += loss.item() * x_h.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            
            is_real = (labels == 0.0)
            is_fake = (labels == 1.0)
            real_correct += ((preds == labels) & is_real).sum().item()
            real_total += is_real.sum().item()
            fake_correct += ((preds == labels) & is_fake).sum().item()
            fake_total += is_fake.sum().item()

            n += x_h.size(0)

    val_loss = total_loss / n
    val_acc = correct / n
    val_real_acc = real_correct / real_total if real_total > 0 else 0.0
    val_fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0

    return val_loss, val_acc, val_real_acc, val_fake_acc


def train(train_dir, val_dir, epochs=100, batch_size=32, lr=1e-3, decay_every=10, decay_rate=0.8, early_stopping_patience=15, resume=False):
    from frequency_model import FrequencyModel  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUGMENTED EXPERIMENT] Training on: {device}")

    train_dataset = FrequencyDatasetAugmented(train_dir, augment=True)
    val_dataset = FrequencyDatasetAugmented(val_dir, augment=False)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "temporal"))
    from balanced_sampler import BalancedUndersampleSampler
    
    # Class balancing with BalancedUndersampleSampler
    labels = [s[1] for s in train_dataset.samples]
    sampler = BalancedUndersampleSampler(labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train samples (Augmented): {len(train_dataset)} | Val samples (Clean): {len(val_dataset)}")

    model = FrequencyModel().to(device)
    print(f"Model parameters: {model.num_parameters:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=decay_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    # SEPARATE CHECKPOINT PATHS (prevents overwriting baseline weights)
    resume_path = "checkpoints/frequency_model_aug_resume.pth"
    best_path = "checkpoints/frequency_model_aug_best.pth"
    final_path = "checkpoints/frequency_model_aug_final.pth"

    if resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed at epoch {start_epoch+1} with best_val_loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, labels = x.to(device), labels.to(device)
            x_h = compute_high_freq_image_gpu(x)

            optimizer.zero_grad()
            logits = model(x_h)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_h.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()

        scheduler.step()

        train_loss = total_loss / len(train_dataset)
        train_acc = correct / len(train_dataset)
        val_loss, val_acc, val_real_acc, val_fake_acc = evaluate(model, val_loader, device, criterion)

        gap_flag = "  <-- train/val gap widening, watch for overfitting" \
            if (train_acc - val_acc) > 0.15 else ""

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
            f"(Real: {val_real_acc:.4f}, Fake: {val_fake_acc:.4f}){gap_flag}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  --> Saved new best augmented weights to {best_path} (loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }, resume_path)

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement in val_loss for {early_stopping_patience} epochs.")
            break

    torch.save(model.state_dict(), final_path)
    print(f"Best val loss: {best_val_loss:.4f} -> {best_path}")
    print(f"Final epoch weights -> {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="e.g. Dataset/Train")
    parser.add_argument("--val_dir", required=True, help="e.g. Dataset/Validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, early_stopping_patience=args.patience, resume=args.resume)
