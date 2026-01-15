#!/usr/bin/env python3
"""
Import Training Data from vSafe Extension

Exports screenshots from the extension's IndexedDB and organizes them
into the training directory structure.

Usage:
    python import_from_extension.py --input export.json --output ../data/screenshots
"""

import os
import json
import base64
import argparse
from pathlib import Path


def decode_data_url(data_url):
    """Decode base64 data URL to bytes."""
    # Format: data:image/png;base64,<data>
    if ',' in data_url:
        header, data = data_url.split(',', 1)
        return base64.b64decode(data)
    return base64.b64decode(data_url)


def import_screenshots(input_file, output_dir):
    """
    Import screenshots from exported JSON.

    Expected JSON format:
    {
        "data": [
            {
                "brand": "google",
                "screenshot": "data:image/png;base64,...",
                "timestamp": 1234567890,
                "url": "https://..."
            }
        ]
    }
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load export file
    with open(input_file, 'r') as f:
        export_data = json.load(f)

    screenshots = export_data.get('data', [])
    print(f"Found {len(screenshots)} screenshots to import")

    imported = 0
    skipped = 0

    for item in screenshots:
        brand = item.get('brand', 'unknown').lower()
        timestamp = item.get('timestamp', 0)
        screenshot = item.get('screenshot', '')

        if not screenshot:
            print(f"  Skipping: no screenshot data")
            skipped += 1
            continue

        # Create brand directory
        brand_dir = output_path / brand
        brand_dir.mkdir(exist_ok=True)

        # Generate filename
        filename = f"{timestamp}.png"
        filepath = brand_dir / filename

        # Skip if already exists
        if filepath.exists():
            print(f"  Skipping: {filepath} already exists")
            skipped += 1
            continue

        # Decode and save
        try:
            image_data = decode_data_url(screenshot)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            print(f"  Saved: {filepath}")
            imported += 1
        except Exception as e:
            print(f"  Error saving {filepath}: {e}")
            skipped += 1

    print(f"\nImport complete!")
    print(f"  Imported: {imported}")
    print(f"  Skipped: {skipped}")

    # Show summary by brand
    print(f"\nScreenshots by brand:")
    for brand_dir in sorted(output_path.iterdir()):
        if brand_dir.is_dir():
            count = len(list(brand_dir.glob('*.png')))
            print(f"  {brand_dir.name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Import screenshots from vSafe extension')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to exported JSON file')
    parser.add_argument('--output', type=str, default='../data/screenshots',
                        help='Output directory for screenshots')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return

    import_screenshots(args.input, args.output)


if __name__ == '__main__':
    main()
