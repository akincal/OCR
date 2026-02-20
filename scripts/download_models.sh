#!/bin/bash

# Script to download all OCR models for the multi-engine OCR API
# Downloads: TrOCR (printed), PaddleOCR, EasyOCR models

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║       OCR Multi-Engine Model Download Script           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "Installing required Python packages..."
pip3 install -q torch transformers huggingface-hub paddlepaddle paddleocr easyocr

# --- 1. TrOCR (printed - better for documents) ---
echo ""
echo "1/3  Downloading TrOCR printed model from Hugging Face..."
echo "     Model: microsoft/trocr-base-printed"
echo ""

cat > /tmp/download_trocr.py << 'EOF'
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model_name = "microsoft/trocr-base-printed"
models_dir = "./models"

print(f"Downloading TrOCR processor: {model_name}")
processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=models_dir)
print("  Processor downloaded")

print(f"Downloading TrOCR model: {model_name}")
model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=models_dir)
print("  Model downloaded")

print("TrOCR download complete!")
EOF

python3 /tmp/download_trocr.py
rm -f /tmp/download_trocr.py

# --- 2. PaddleOCR ---
echo ""
echo "2/3  Downloading PaddleOCR models (detection + recognition + angle)..."
echo ""

cat > /tmp/download_paddle.py << 'EOF'
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

print("Initializing PaddleOCR (will download models)...")
ocr = PaddleOCR(use_angle_cls=True, lang="tr", use_gpu=False, show_log=True)

print("Running test inference to verify models...")
img = np.ones((100, 300, 3), dtype=np.uint8) * 255
result = ocr.ocr(img, cls=True)
print("PaddleOCR download complete!")
EOF

python3 /tmp/download_paddle.py
rm -f /tmp/download_paddle.py

# --- 3. EasyOCR ---
echo ""
echo "3/3  Downloading EasyOCR models (Turkish + English)..."
echo ""

cat > /tmp/download_easyocr.py << 'EOF'
import easyocr
import numpy as np

models_dir = "./models"
print("Initializing EasyOCR reader (will download models)...")
reader = easyocr.Reader(["tr", "en"], gpu=False, model_storage_directory=models_dir)

print("Running test inference to verify models...")
img = np.ones((100, 300, 3), dtype=np.uint8) * 255
result = reader.readtext(img, detail=0)
print("EasyOCR download complete!")
EOF

python3 /tmp/download_easyocr.py
rm -f /tmp/download_easyocr.py

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  All models downloaded successfully!                   ║"
echo "╠════════════════════════════════════════════════════════╣"
echo "║  Engines ready:                                        ║"
echo "║    - Tesseract (system package, no download needed)    ║"
echo "║    - EasyOCR (CRAFT + CRNN, Turkish + English)         ║"
echo "║    - TrOCR (ViT encoder + Transformer decoder)         ║"
echo "║    - PaddleOCR (DB + CRNN, angle classification)       ║"
echo "║    - Ensemble (combines Tesseract + EasyOCR + Paddle)  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Models directory contents:"
ls -lh "$MODELS_DIR" 2>/dev/null || echo "  (models cached in PaddleOCR/HuggingFace default dirs)"
echo ""
echo "You can now start the OCR API server!"
echo ""
echo "════════════════════════════════════════════════════════"
