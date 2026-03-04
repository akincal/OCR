#!/bin/bash

# Script to pre-download TrOCR models for OCR API
# This script downloads pre-trained TrOCR models from Hugging Face
# into the ./models directory in the format expected by the Python
# OCR server (Transformers + PyTorch, no ONNX conversion).

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║       TrOCR Model Download Script                     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Create models directory if it doesn't exist
MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "📦 Installing required Python packages..."
pip3 install -q torch transformers huggingface-hub

echo ""
echo "📥 Downloading TrOCR handwritten model from Hugging Face..."
echo "   Model: microsoft/trocr-base-handwritten"
echo ""

# Create Python script to download and cache the model
cat > /tmp/download_trocr.py << 'EOF'
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def main():
    model_name = "microsoft/trocr-base-handwritten"
    models_dir = "./models"

    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading model: {model_name}")

    # Download processor
    print("Downloading processor...")
    processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=models_dir)
    processor.save_pretrained(models_dir)
    print("✓ Processor saved")

    # Download model weights (PyTorch)
    print("Downloading model weights (PyTorch)...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=models_dir)
    model.save_pretrained(models_dir)
    print("✓ Model weights saved")

    print("\n✅ Models downloaded successfully!")
    print(f"📁 Models saved to: {os.path.abspath(models_dir)}")

if __name__ == "__main__":
    main()
EOF

# Run the Python script
python3 /tmp/download_trocr.py

echo ""
echo "✅ Model download complete!"
echo ""
echo "Downloaded files:"
ls -lh "$MODELS_DIR"
echo ""
echo "🚀 You can now start the OCR API server"

# Cleanup
rm -f /tmp/download_trocr.py

echo ""
echo "════════════════════════════════════════════════════════"
