"""
Training script for the Temporal Model (Phase 2).

DATASET: FaceForensics++ (c23), from Kaggle --
    kaggle.com/datasets/xdxd003/ff-c23
No approval form needed, unlike the official ondyari/FaceForensics route --
this is already hosted, download directly.

EXPECTED FOLDER STRUCTURE (matches the Kaggle download exactly):
    data_root/
        original/          <- real videos
        Deepfakes/          <- fake
        Face2Face/          <- fake
        FaceSwap/           <- fake
        FaceShifter/        <- fake
        NeuralTextures/     <- fake

LESSON APPLIED FROM PHASE 1: the temporal-frequency FFT is computed on
the GPU from the start this time (compute_temporal_frequency_gpu below),
not per-item on CPU in the Dataset -- that CPU bottleneck cost days on
the Frequency Model and there's no reason to repeat it here.
"""

import os
import sys
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "preprocessing"))
sys.path.insert(0, os.path.dirname(__file__))
from video_processor import VideoProcessor  # noqa: E402
from sequence_builder import SequenceBuilder  # noqa: E402


class TemporalDataset(Dataset):
    """
    Loads one random 16-frame sequence per video, from the real
    FaceForensics++ (c23) folder structure as hosted on Kaggle
    (kaggle.com/datasets/xdxd003/ff-c23):

        data_root/
            original/          <- REAL videos
            Deepfakes/          <- FAKE (pooled together)
            Face2Face/          <- FAKE
            FaceSwap/           <- FAKE
            FaceShifter/        <- FAKE
    - Real samples: label = 0.0
    - Fake samples: label = 1.0
    """

    FAKE_FOLDERS = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]
    _shared_face_detector = None

    def __init__(self, data_root, sequence_length=16, image_size=224, max_cache_size=4000, face_detector=None, fake_folders_subset="all"):
        self.video_processor = VideoProcessor()
        self.sequence_builder = SequenceBuilder()
        if face_detector is None:
            if TemporalDataset._shared_face_detector is None:
                from face_detector import FaceDetector
                TemporalDataset._shared_face_detector = FaceDetector()
            face_detector = TemporalDataset._shared_face_detector
        self.face_detector = face_detector
        from face_crop_helper import FaceCropHelper
        self.face_cropper = FaceCropHelper(self.face_detector, output_size=image_size)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.samples = []
        self.cache = {}
        self.max_cache_size = max_cache_size

        real_dir = None
        for candidate in ["original", "Real", "real"]:
            path = os.path.join(data_root, candidate)
            if os.path.isdir(path):
                real_dir = path
                break

        if real_dir is None:
            raise RuntimeError(f"Expected a real videos folder (original/ or Real/) inside {data_root}.")

        for f in os.listdir(real_dir):
            if not f.startswith('.'):
                self.samples.append((os.path.join(real_dir, f), 0.0))

        if fake_folders_subset == "all":
            fake_folders = self.FAKE_FOLDERS + ["Fake", "fake"]
        else:
            fake_folders = [f.strip() for f in fake_folders_subset.split(",")]

        found_fakes = False
        for method in fake_folders:
            fake_dir = os.path.join(data_root, method)
            if os.path.isdir(fake_dir):
                found_fakes = True
                for f in os.listdir(fake_dir):
                    if not f.startswith('.'):
                        self.samples.append((os.path.join(fake_dir, f), 1.0))

        if not self.samples or not found_fakes:
            raise RuntimeError(
                f"No video samples found under {data_root}. Expected 'original' or 'Real' "
                f"and at least one fake directory ({fake_folders})."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self.cache:
            frames_uint8, label_val = self.cache[idx]
            return frames_uint8.float() / 255.0, label_val

        path, label = self.samples[idx]

        raw_frames = self.video_processor.extract_frames(path)
        frame_data = [{"frame_id": i, "original_frame": f} for i, f in enumerate(raw_frames)]

        sequences = self.sequence_builder.build_video_sequences(frame_data)
        seq = random.choice(sequences) if sequences else frame_data  # fallback for very short clips

        raw_frame_list = [item["original_frame"] for item in seq[:self.sequence_length]]
        cropped_frames = self.face_cropper.crop_sequence(raw_frame_list)

        frames_tensor = []
        for frame in cropped_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_tensor.append(torch.from_numpy(rgb).permute(2, 0, 1))

        # Pad if the sequence came up short
        while len(frames_tensor) < self.sequence_length:
            frames_tensor.append(frames_tensor[-1].clone())

        frames_uint8 = torch.stack(frames_tensor).to(torch.uint8)  # (T, 3, H, W) uint8 tensor (2.4 MB)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if len(self.cache) < self.max_cache_size:
            self.cache[idx] = (frames_uint8, label_tensor)

        return frames_uint8.float() / 255.0, label_tensor


def compute_temporal_frequency_gpu(frames):
    """
    GPU-batched version of frequency_generator.compute_temporal_frequency().
    Computed once per batch here, not per-item in the Dataset -- this is
    the fix applied preemptively, based on the Phase 1 CPU-bottleneck
    lesson (that one cost ~50 hours before it got caught).

    frames: (B, T, 3, H, W), 0-1 normalized RGB
    returns: (B, 1, H, W) anomaly map
    """
    # Grayscale: standard luminance weights
    gray = (0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2])  # (B, T, H, W)

    freq = torch.fft.rfft(gray, dim=1)  # FFT along the TIME axis
    magnitude = torch.abs(freq)         # (B, T//2+1, H, W)
    magnitude = torch.log1p(magnitude)

    # Normalize per-sample
    B = magnitude.shape[0]
    mag_flat = magnitude.reshape(B, -1)
    mag_min = mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

    # Drop DC component (index 0), sum the rest -> anomaly map
    anomaly_map = magnitude[:, 1:].sum(dim=1) if magnitude.shape[1] > 1 else magnitude[:, 0]
    return anomaly_map.unsqueeze(1)  # (B, 1, H, W)


def evaluate(model, loader, device, criterion, use_anomaly=True):
    model.eval()
    total_loss, n = 0.0, 0
    real_correct, fake_correct = 0, 0
    total_real, total_fake = 0, 0
    pred_fakes = 0

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Validation"):
            frames, labels = frames.to(device), labels.to(device)
            anomaly_map = compute_temporal_frequency_gpu(frames) if use_anomaly else None

            with torch.amp.autocast('cuda'):
                logits, _ = model(frames, anomaly_map)
                loss = criterion(logits, labels)

            total_loss += loss.item() * frames.size(0)
            preds = (torch.sigmoid(logits.float()) > 0.5).float()
            
            pred_fakes += (preds == 1.0).sum().item()
            real_mask = (labels == 0.0)
            fake_mask = (labels == 1.0)

            total_real += real_mask.sum().item()
            total_fake += fake_mask.sum().item()

            real_correct += ((preds == 0.0) & real_mask).sum().item()
            fake_correct += ((preds == 1.0) & fake_mask).sum().item()
            n += frames.size(0)

    val_loss = total_loss / n
    real_acc = real_correct / max(total_real, 1)
    fake_acc = fake_correct / max(total_fake, 1)
    balanced_acc = (real_acc + fake_acc) / 2.0
    overall_acc = (real_correct + fake_correct) / n

    return val_loss, overall_acc, real_acc, fake_acc, balanced_acc, pred_fakes


def train(train_dir, val_dir, epochs=100, batch_size=16, lr=1e-4,
          warmup_epochs=2, early_stopping_patience=15, use_anomaly=True, fake_folders="all", checkpoint_suffix="",
          unfreeze_backbone=False, resume_from=""):
    from temporal_model import TemporalModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model Configuration: {'RGB + Temporal Anomaly Fusion' if use_anomaly else 'Bare RGB + LSTM (Option B Baseline)'}")
    print(f"Hyperparameters: LR={lr}, Warmup Epochs={warmup_epochs}, Total Epochs={epochs}")
    print(f"Fake Folders Subset: {fake_folders}")

    train_dataset = TemporalDataset(train_dir, fake_folders_subset=fake_folders)
    val_dataset = TemporalDataset(val_dir, fake_folders_subset=fake_folders)

    # Class imbalance handling: WeightedRandomSampler for 50:50 real/fake mini-batches
    labels_list = [sample[1] for sample in train_dataset.samples]
    real_count = sum(1 for l in labels_list if l == 0.0)
    fake_count = sum(1 for l in labels_list if l == 1.0)
    print(f"Dataset counts - Real: {real_count} | Fake: {fake_count}")

    from balanced_sampler import BalancedUndersampleSampler

    sampler = BalancedUndersampleSampler(labels_list)
    print(f"Balanced sampler: {sampler.n} real + {sampler.n} fake per epoch "
          f"({len(sampler)} total, undersampled from {fake_count} fakes)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)} (Balanced 50/50 batches) | Val samples: {len(val_dataset)}")

    model = TemporalModel(freeze_backbone=True, use_anomaly=use_anomaly).to(device)
    if unfreeze_backbone:
        model.unfreeze_backbone(top_layers_only=True)
    print(f"Initial model parameters: {model.num_parameters:,} (Trainable: {model.num_trainable_parameters:,})")

    criterion = nn.BCEWithLogitsLoss()
    
    if unfreeze_backbone:
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'frame_cnn.backbone' in n:
                backbone_params.append(p)
            else:
                head_params.append(p)
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    scaler = torch.amp.GradScaler('cuda')

    best_val_balanced_acc = 0.0
    patience_counter = 0
    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_balanced_acc = checkpoint.get('best_val_balanced_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed at epoch {start_epoch} with Best Val Acc: {best_val_balanced_acc*100:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            frames, labels = frames.to(device), labels.to(device)
            anomaly_map = compute_temporal_frequency_gpu(frames) if use_anomaly else None

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(frames, anomaly_map)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * frames.size(0)
            preds = (torch.sigmoid(logits.float()) > 0.5).float()
            correct += (preds == labels).sum().item()
            n += frames.size(0)

        scheduler.step()

        train_loss = total_loss / n
        train_acc = correct / n
        val_loss, val_acc, real_acc, fake_acc, balanced_acc, pred_fakes = evaluate(
            model, val_loader, device, criterion, use_anomaly=use_anomaly
        )

        gap_flag = "  <-- train/val gap widening, watch for overfitting" \
            if (train_acc - balanced_acc) > 0.15 else ""

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
            f"(real_acc: {real_acc*100:.1f}%, fake_acc: {fake_acc*100:.1f}%, balanced_acc: {balanced_acc*100:.1f}%) "
            f"[pred fakes: {pred_fakes}/{len(val_dataset)}]{gap_flag}"
        )

        if balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = balanced_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoints/temporal_model_best{checkpoint_suffix}.pth")
            print(f" -> Saved new best model checkpoint! (Balanced Acc: {best_val_balanced_acc*100:.2f}%)")
        else:
            patience_counter += 1

        # Periodic checkpointing every 20 epochs
        if (epoch + 1) % 20 == 0:
            milestone_path = f"checkpoints/temporal_model_epoch_{epoch+1}{checkpoint_suffix}.pth"
            torch.save(model.state_dict(), milestone_path)
            print(f" -> Saved milestone checkpoint: {milestone_path}")

        # Save resume state after every epoch
        resume_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_balanced_acc': best_val_balanced_acc,
            'patience_counter': patience_counter
        }
        torch.save(resume_state, f"checkpoints/temporal_model_resume{checkpoint_suffix}.pth")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement in balanced_acc for {early_stopping_patience} epochs.")
            break

    torch.save(model.state_dict(), f"checkpoints/temporal_model_final{checkpoint_suffix}.pth")
    print(f"Training complete! Best val balanced accuracy: {best_val_balanced_acc*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no_anomaly", action="store_true", help="Run bare RGB + LSTM baseline (Option B)")
    parser.add_argument("--fake_folders", type=str, default="all", help="Comma-separated list of fake folders to use, or 'all'")
    parser.add_argument("--checkpoint_suffix", type=str, default="", help="Suffix to append to checkpoint files")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze top layers of ResNet18 backbone")
    parser.add_argument("--resume_from", type=str, default="", help="Path to resume checkpoint file")
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, warmup_epochs=args.warmup_epochs,
          early_stopping_patience=args.patience, use_anomaly=not args.no_anomaly,
          fake_folders=args.fake_folders, checkpoint_suffix=args.checkpoint_suffix,
          unfreeze_backbone=args.unfreeze_backbone, resume_from=args.resume_from)