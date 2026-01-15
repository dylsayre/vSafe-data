#!/usr/bin/env python3
"""
vSafe ONNX Export Script

Converts PyTorch model to ONNX format with INT8 quantization.
Output is optimized for ONNX Runtime Web inference in the browser.

Usage:
    python export_onnx.py --checkpoint ../models/checkpoints/best_model.pth
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def create_model(num_classes):
    """Create MobileNetV3-Small model architecture."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def export_to_onnx(model, output_path, opset_version=14):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (14 is widely supported)
    """
    model.eval()

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Exported ONNX model to: {output_path}")

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

    return output_path


def quantize_model(input_path, output_path):
    """
    Apply INT8 dynamic quantization to reduce model size.

    This typically reduces model size by 4x with minimal accuracy loss.
    """
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True
    )

    # Compare sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nQuantization results:")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Reduction: {(1 - quantized_size/original_size) * 100:.1f}%")

    return output_path


def convert_to_ort(input_path, output_path):
    """
    Convert ONNX model to ORT format for optimized inference.

    The .ort format includes pre-optimized graph for faster loading.
    """
    import onnxruntime as ort

    # Create session with optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(output_path)

    # This creates the optimized model
    _ = ort.InferenceSession(str(input_path), sess_options)

    print(f"Converted to ORT format: {output_path}")

    # Check final size
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Final model size: {final_size:.2f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export vSafe model to ONNX')
    parser.add_argument('--checkpoint', type=str, default='../models/checkpoints/best_model.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output_dir', type=str, default='../models/exports',
                        help='Output directory for ONNX models')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='Apply INT8 quantization')
    parser.add_argument('--no-quantize', dest='quantize', action='store_false',
                        help='Skip quantization')
    parser.add_argument('--ort', action='store_true', default=True,
                        help='Convert to ORT format')
    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Please run train.py first to create a model checkpoint.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model info
    class_names = checkpoint.get('class_names', [])
    num_classes = checkpoint.get('num_classes', len(class_names))
    val_acc = checkpoint.get('val_acc', 0)

    print(f"Model info:")
    print(f"  Classes: {num_classes} ({', '.join(class_names)})")
    print(f"  Validation accuracy: {val_acc:.2f}%")

    # Create model and load weights
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print(f"\n{'='*50}")
    print("Step 1: Export to ONNX")
    onnx_path = output_dir / 'brand_classifier.onnx'
    export_to_onnx(model, str(onnx_path))

    # Quantize
    if args.quantize:
        print(f"\n{'='*50}")
        print("Step 2: Quantize to INT8")
        quantized_path = output_dir / 'brand_classifier_int8.onnx'
        quantize_model(str(onnx_path), str(quantized_path))
        final_onnx = quantized_path
    else:
        final_onnx = onnx_path

    # Convert to ORT
    if args.ort:
        print(f"\n{'='*50}")
        print("Step 3: Convert to ORT format")
        ort_path = output_dir / 'brand_classifier.ort'
        convert_to_ort(str(final_onnx), str(ort_path))

    # Save class names
    class_names_path = output_dir / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"\nClass names saved to: {class_names_path}")

    # Save model metadata
    metadata = {
        'version': '1.0.0',
        'num_classes': num_classes,
        'class_names': class_names,
        'input_shape': [1, 3, 224, 224],
        'val_accuracy': val_acc,
        'quantized': args.quantize,
        'architecture': 'MobileNetV3-Small'
    }
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print(f"\n{'='*50}")
    print("Export complete!")
    print(f"\nFiles created in {output_dir}:")
    for f in output_dir.iterdir():
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")

    print(f"\nTo use in vSafe extension:")
    print(f"  1. Copy {output_dir / 'brand_classifier.ort'} to vSafe-ext/models/")
    print(f"  2. Copy {output_dir / 'class_names.json'} to vSafe-ext/models/")
    print(f"  3. Rebuild the extension: npm run build")


if __name__ == '__main__':
    main()
