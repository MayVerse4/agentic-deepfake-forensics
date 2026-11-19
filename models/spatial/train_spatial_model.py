import os
import torch
import torch.nn as nn
from collections import Counter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from spatial_model import DeepfakeSpatialModel

def train(train_dir, val_dir, epochs=100, batch_size=32, lr=1e-4, decay_every=10, decay_rate=0.8, early_stopping_patience=15, finetune_from=None, save_name="spatial_model_finetuned_ff.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Spatial Model on: {device}")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # We want Fake (manipulated) to be 1, Real (original) to be 0
    real_idx = train_dataset.class_to_idx['real']
    fake_idx = train_dataset.class_to_idx['fake']

    def map_target(target):
        return 1.0 if target == fake_idx else 0.0

    # Class balancing with WeightedRandomSampler (handles 5:1 fake:real imbalance)
    class_counts = Counter(train_dataset.targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} (Balanced 50/50 sampling active)")

    model = DeepfakeSpatialModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=decay_every, gamma=decay_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    
    checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_path = os.path.join(checkpoints_dir, save_name)
    
    if finetune_from and os.path.exists(finetune_from):
        print(f"Fine-tuning from existing weights: {finetune_from}...")
        checkpoint = torch.load(finetune_from, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded base weights for fine-tuning.")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for x, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x = x.to(device)
            mapped_targets = torch.tensor([map_target(t.item()) for t in targets], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, mapped_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == mapped_targets).sum().item()

        scheduler.step()

        train_loss = total_loss / len(train_dataset)
        train_acc = correct / len(train_dataset)
        
        # Validation with per-class accuracy metrics
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_real_correct = 0
        val_real_total = 0
        val_fake_correct = 0
        val_fake_total = 0

        with torch.no_grad():
            for x, targets in tqdm(val_loader, desc="Validation"):
                x = x.to(device)
                mapped_targets = torch.tensor([map_target(t.item()) for t in targets], dtype=torch.float32).unsqueeze(1).to(device)
                logits = model(x)
                loss = criterion(logits, mapped_targets)
                val_loss_total += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                val_correct += (preds == mapped_targets).sum().item()
                
                is_real = (mapped_targets == 0.0)
                is_fake = (mapped_targets == 1.0)
                val_real_correct += ((preds == mapped_targets) & is_real).sum().item()
                val_real_total += is_real.sum().item()
                val_fake_correct += ((preds == mapped_targets) & is_fake).sum().item()
                val_fake_total += is_fake.sum().item()
                
        val_loss = val_loss_total / len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        val_real_acc = val_real_correct / val_real_total if val_real_total > 0 else 0.0
        val_fake_acc = val_fake_correct / val_fake_total if val_fake_total > 0 else 0.0

        gap_flag = "  <-- train/val gap widening, watch for overfitting" if (train_acc - val_acc) > 0.15 else ""
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} (real_acc: {val_real_acc:.4f}, fake_acc: {val_fake_acc:.4f}){gap_flag}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best validation loss! Saved to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement in val_loss for {early_stopping_patience} epochs.")
            break

    print(f"Best val loss: {best_val_loss:.4f} -> {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for fine-tuning
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--finetune_from", type=str, default="../../checkpoints_kaggle_baseline/spatial_model_aug_best.pth")
    parser.add_argument("--save_name", type=str, default="spatial_model_finetuned_ff.pth")
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, early_stopping_patience=args.patience, finetune_from=args.finetune_from, save_name=args.save_name)

