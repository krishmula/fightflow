import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

import argparse
import sys
from pathlib import Path

# Need to ensure src/ is on the path since we're inside scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.peak_dataset import PeakFrameDataset
from src.models.baseline_cnn.model import BaselineCNN
from src.models.advanced_cnn.model import AdvancedCNN
from src.engine.trainer import train_one_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="Train CNN models (Baseline or Advanced)")
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'advanced'], 
                        help='Which model to train')
    args = parser.parse_args()

    print(f"Initializing training for FightFlow {args.model.capitalize()} CNN...")

    # Configuration
    CSV_PATH = "data/annotations.csv"
    VIDEO_DIR = "data/downloaded-videos"
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 3e-4 # AdamW learning rate

    # In our case, MPS is usually the fastest deep learning backend on Mac M1/M2/M3s
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS backend!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("WARNING: Falling back to CPU.")

    val_videos = ['V7.mp4']
    
    import pandas as pd
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Have you annotated yet?")
    
    df = pd.read_csv(CSV_PATH)
    all_videos = df['video_id'].unique().tolist()
    train_videos = [v for v in all_videos if v not in val_videos]

    print(f"Training on videos: {train_videos}")
    print(f"Validating on videos: {val_videos}")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # RandomHorizontalFlip omitted: flipping a jab produces a visually southpaw cross,
        # corrupting the label. See docs/annotation-process.md.
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PeakFrameDataset(CSV_PATH, VIDEO_DIR, train_videos, transform=train_transform)
    val_dataset = PeakFrameDataset(CSV_PATH, VIDEO_DIR, val_videos, transform=val_transform)

    print(f"Train Dataset size: {len(train_dataset)} annotations")
    print(f"Val Dataset size: {len(val_dataset)} annotations")
    
    if len(train_dataset) == 0:
        print("Please annotate training videos first!")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        val_loader = None

    if args.model == 'advanced':
        model = AdvancedCNN(num_classes=4).to(device)
        writer = SummaryWriter("runs/advanced_cnn")
        best_model_path_template = "checkpoints/advanced_cnn_best.pth"
    else:
        model = BaselineCNN(num_classes=4).to(device)
        writer = SummaryWriter("runs/baseline_cnn")
        best_model_path_template = "checkpoints/baseline_cnn_best.pth"

    # Class distribution weighting
    weights = torch.tensor([0.2, 1.0, 1.0, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Initialize Checkpoints & TensorBoard
    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    # Big Loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        if val_loader:
            val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device, epoch)
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

            # Log to TensorBoard
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            # Per-class breakdown every epoch to catch majority-class collapse early
            target_names = ['none', 'straight', 'hook', 'uppercut']
            actual_present_labels = sorted(list(set(all_labels)))
            filtered_target_names = [target_names[i] for i in actual_present_labels]
            print(classification_report(all_labels, all_preds, target_names=filtered_target_names, zero_division=0))

            scheduler.step(val_acc)

            # Checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path_template)
                print(f"🌟 New Best Validation Accuracy! Model saved to {best_model_path_template}")
        else:
            # If no validation set, just save the best training accuracy
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), best_model_path_template)
                print(f"🌟 New Best Train Accuracy! Model saved to {best_model_path_template}")

    writer.close()

if __name__ == '__main__':
    main()
