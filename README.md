# vSafe Training

Training infrastructure for the vSafe phishing detection browser extension.

## Overview

This repository contains:
- Training data (login page screenshots organized by brand)
- PyTorch training scripts for MobileNetV3-Small
- ONNX export pipeline with INT8 quantization
- GitHub Actions for automatic retraining

## Directory Structure

```
vSafe-training/
├── data/
│   ├── screenshots/          # Training images by brand
│   │   ├── google/
│   │   ├── microsoft/
│   │   └── ...
│   └── labels.json           # Optional metadata
├── scripts/
│   ├── train.py              # PyTorch training script
│   └── export_onnx.py        # ONNX conversion
├── models/
│   ├── checkpoints/          # PyTorch checkpoints (gitignored)
│   └── exports/              # ONNX models for release
├── .github/workflows/
│   └── train-model.yml       # Auto-training workflow
└── requirements.txt
```

## Quick Start

### 1. Add Training Data

Add screenshots to `data/screenshots/{brand}/`:

```
data/screenshots/
├── google/
│   ├── screenshot1.png
│   ├── screenshot2.png
├── microsoft/
│   ├── screenshot1.png
└── ...
```

Minimum requirements:
- At least 2 brands
- At least 5 images per brand (more is better)

### 2. Train Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
cd scripts
python train.py --data_dir ../data/screenshots --epochs 50

# Export to ONNX
python export_onnx.py --checkpoint ../models/checkpoints/best_model.pth
```

### 3. Deploy to Extension

```bash
# Copy model files to extension
cp models/exports/brand_classifier.ort ../vSafe-ext/models/
cp models/exports/class_names.json ../vSafe-ext/models/

# Rebuild extension
cd ../vSafe-ext
npm run build
```

## Automatic Training

When you push new screenshots to `data/screenshots/`, GitHub Actions will:

1. Train a new model
2. Export to ONNX with INT8 quantization
3. Create a GitHub Release with the model files

## Model Architecture

- **Base**: MobileNetV3-Small (pretrained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: Brand classification probabilities
- **Size**: ~2.5MB (quantized)
- **Inference**: <100ms on WebGPU

## Contributing Training Data

The vSafe extension can upload screenshots directly to this repository (requires GitHub token configuration). This creates a feedback loop:

1. Extension captures screenshots
2. Screenshots uploaded here
3. GitHub Actions retrains model
4. New model released
5. Extension downloads update

## License

MIT
