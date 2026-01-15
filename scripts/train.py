#!/usr/bin/env python3
"""
vSafe Brand Classifier Training Script

Trains a MobileNetV3-Small model to classify login page screenshots by brand.
Optimized for browser inference with ONNX Runtime Web.

Usage:
    python train.py --data_dir ../data/screenshots --epochs 50
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class BrandDataset(Dataset):
    """Dataset for brand classification from screenshots."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset(data_dir):
    """
    Load dataset from directory structure.

    Expected structure:
        data/screenshots/
            google/
                image1.png
                image2.png
            microsoft/
                image1.png
            ...

    Returns:
        image_paths: List of image file paths
        labels: List of integer labels
        class_names: List of brand names
    """
    data_path = Path(data_dir)

    image_paths = []
    labels = []
    class_names = []

    # Get all brand directories
    brand_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    for class_idx, brand_dir in enumerate(brand_dirs):
        brand_name = brand_dir.name
        class_names.append(brand_name)

        # Get all images in this brand directory
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            for image_path in brand_dir.glob(ext):
                image_paths.append(str(image_path))
                labels.append(class_idx)

    print(f"Loaded {len(image_paths)} images across {len(class_names)} brands")
    for idx, name in enumerate(class_names):
        count = labels.count(idx)
        print(f"  {name}: {count} images")

    return image_paths, labels, class_names


def create_model(num_classes, pretrained=True):
    """
    Create MobileNetV3-Small model for brand classification.

    MobileNetV3-Small is chosen because:
    - Small model size (~2.5MB)
    - Fast inference (<100ms on WebGPU)
    - Good accuracy for image classification
    """
    if pretrained:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    # Replace classifier head for our number of classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def get_transforms(train=True):
    """Get image transforms for training or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train vSafe brand classifier')
    parser.add_argument('--data_dir', type=str, default='../data/screenshots',
                        help='Path to screenshot data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--output_dir', type=str, default='../models/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--min_images', type=int, default=5,
                        help='Minimum images per class to train')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    image_paths, labels, class_names = load_dataset(args.data_dir)

    if len(image_paths) == 0:
        print("ERROR: No images found! Please add screenshots to data/screenshots/{brand}/")
        return

    if len(class_names) < 2:
        print("ERROR: Need at least 2 brands to train. Please add more screenshots.")
        return

    # Check minimum images per class
    for idx, name in enumerate(class_names):
        count = labels.count(idx)
        if count < args.min_images:
            print(f"WARNING: {name} has only {count} images (minimum: {args.min_images})")

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=args.val_split,
        stratify=labels,
        random_state=42
    )

    print(f"\nTrain set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")

    # Create datasets
    train_dataset = BrandDataset(train_paths, train_labels, transform=get_transforms(train=True))
    val_dataset = BrandDataset(val_paths, val_labels, transform=get_transforms(train=False))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    num_classes = len(class_names)
    print(f"\nCreating MobileNetV3-Small model with {num_classes} classes")
    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
                'num_classes': num_classes
            }, checkpoint_path)
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")

    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes
    }, final_path)

    # Save class names
    class_names_path = output_dir.parent / 'exports' / 'class_names.json'
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Class names saved to: {class_names_path}")
    print(f"\nNext step: Run export_onnx.py to convert to ONNX format")


if __name__ == '__main__':
    main()
