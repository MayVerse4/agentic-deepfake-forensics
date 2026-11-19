"""
Training script for the Frequency Model (Phase 1).

IMPORTANT -- READ FIRST:
Preprocessing has no concept of "real" or "fake" -- it never sees labels,
by design (that's the whole point of it not producing verdicts). Training
this model needs an actual LABELED deepfake dataset, which is a separate
thing from anything your pipeline has produced so far.

You need real/fake face images. Reasonable options, easiest first:
  - FaceForensics++ (the standard academic benchmark, matches your master
    prompt's own reference to it)
  - Celeb-DF v2
  - The ForenSynths training set the FreqNet paper itself uses (real
    images from LSUN + fake images from ProGAN) -- download_dataset.sh
    in the FreqNet repo automates this

This script expects a Kaggle-style pre-split dataset:
    Dataset/
        Train/
            real/   *.jpg
            fake/   *.jpg
        Validation/
            real/   *.jpg
            fake/   *.jpg
        Test/
            real/   *.jpg
            fake/   *.jpg
(matches manjilkarki/deepfake-and-real-images on Kaggle directly)

Run: python train_frequency_model.py --train_dir Dataset/Train --val_dir Dataset/Validation
"""

import os
import sys
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "preprocessing"))
from frequency_generator import FrequencyGenerator  # type: ignore # noqa: E402
from frequency_model import compute_high_freq_image_gpu  # type: ignore # noqa: E402


class FrequencyDataset(Dataset):
    def __init__(self, split_dir):
        self.freq_gen = FrequencyGenerator()
        self.samples = []

        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
            raise RuntimeError(
                f"Expected {split_dir}/real and {split_dir}/fake to exist. "
                "Point split_dir at the Train, Test, or Validation folder "
                "of the dataset, not its parent."
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
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float()  # -> (3, H, W)

        return x, torch.tensor(label, dtype=torch.float32)

def evaluate(model, loader, device, criterion, mask_radius=0.25, norm="minmax"):
    model.eval()
    total_loss = 0.0
    correct = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    n = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, labels in tqdm(loader, desc="Validation"):
            x, labels = x.to(device), labels.to(device)
            x_h = compute_high_freq_image_gpu(x, mask_radius=mask_radius, norm=norm)
            logits = model(x_h)
            loss = criterion(logits, labels)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

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
    val_auc = roc_auc_score(torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy())
    return val_loss, val_acc, val_real_acc, val_fake_acc, val_auc


def train(train_dir, val_dir, epochs=100, batch_size=32, lr=1e-4, decay_every=10, decay_rate=0.8, early_stopping_patience=15, resume=False, finetune_from=None, save_name="frequency_model_finetuned_ff.pth", mask_radius=0.25, arch="freqnet", norm="minmax"):
    from frequency_model import build_frequency_model  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset = FrequencyDataset(train_dir)
    val_dataset = FrequencyDataset(val_dir)
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "temporal"))
    from balanced_sampler import BalancedUndersampleSampler
    
    # Class balancing with BalancedUndersampleSampler
    labels = [s[1] for s in train_dataset.samples]
    sampler = BalancedUndersampleSampler(labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} (Balanced 50/50 sampling active)")

    model = build_frequency_model(arch).to(device)
    print(f"Arch: {arch} | norm: {norm} | mask_radius: {mask_radius}")
    print(f"Model parameters: {model.num_parameters:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=decay_rate)

    # Best checkpoint is picked by val AUC: the score feeds Fusion as a ranking
    # signal, and val loss rewards hedging near 0.5 (which is how the old
    # checkpoints collapsed).
    best_val_auc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    os.makedirs("checkpoints", exist_ok=True)
    resume_path = f"checkpoints/resume_{save_name}"
    
    if finetune_from and os.path.exists(finetune_from) and not resume:
        print(f"Fine-tuning from existing weights: {finetune_from}...")
        checkpoint = torch.load(finetune_from, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded base weights for fine-tuning.")
    
    if resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed at epoch {start_epoch+1} with best_val_loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        train_n = 0

        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, labels = x.to(device), labels.to(device)
            x_h = compute_high_freq_image_gpu(x, mask_radius=mask_radius, norm=norm)

            optimizer.zero_grad()
            logits = model(x_h)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_h.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            train_n += x_h.size(0)

        scheduler.step()

        train_loss = total_loss / train_n if train_n > 0 else 0.0
        train_acc = correct / train_n if train_n > 0 else 0.0
        val_loss, val_acc, val_real_acc, val_fake_acc, val_auc = evaluate(model, val_loader, device, criterion, mask_radius=mask_radius, norm=norm)

        gap_flag = "  <-- train/val gap widening, watch for overfitting" \
            if (train_acc - val_acc) > 0.15 else ""

        phase_info = ""
        if hasattr(model, "fcl"):
            phase_info = (f"\n      -> fcl.phase_conv.weight norm: {torch.norm(model.fcl.phase_conv.weight).item():.4f}, "
                          f"mean: {torch.mean(model.fcl.phase_conv.weight).item():.4f}")

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_auc: {val_auc:.4f} "
            f"(real_acc: {val_real_acc:.4f}, fake_acc: {val_fake_acc:.4f}){gap_flag}{phase_info}",
            flush=True,
        )

        best_val_loss = min(best_val_loss, val_loss)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoints/{save_name}")
            print(f"New best validation AUC {val_auc:.4f}! Saved to checkpoints/{save_name}", flush=True)
        else:
            patience_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter
        }, resume_path)

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement in val AUC for {early_stopping_patience} epochs.")
            break

    torch.save(model.state_dict(), f"checkpoints/final_{save_name}")
    print(f"Best val AUC: {best_val_auc:.4f} -> checkpoints/{save_name}")
    print(f"Final epoch weights -> checkpoints/final_{save_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="e.g. Dataset/Train")
    parser.add_argument("--val_dir", required=True, help="e.g. Dataset/Validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    parser.add_argument("--finetune_from", type=str, default="../../checkpoints_kaggle_baseline/frequency_model_aug_best.pth")
    parser.add_argument("--save_name", type=str, default="frequency_model_finetuned_ff.pth")
    parser.add_argument("--mask_radius", type=float, default=0.25, help="Radius ratio for high pass filter (e.g. 0.15 or 0.25)")
    parser.add_argument("--arch", choices=["freqnet", "resnet18"], default="freqnet")
    parser.add_argument("--norm", choices=["minmax", "std"], default="minmax", help="HFRI normalization (see compute_high_freq_image_gpu)")
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, early_stopping_patience=args.patience, resume=args.resume, finetune_from=args.finetune_from, save_name=args.save_name, mask_radius=args.mask_radius, arch=args.arch, norm=args.norm)