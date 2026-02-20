# Multi-stage build for OCR API

# Stage 1: Build Go binary
FROM golang:1.21-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /app/ocr-server ./cmd/server

# Stage 2: Runtime stage with Python
FROM python:3.10-slim-bookworm

# Install Tesseract OCR, Hunspell Turkish, and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    wget \
    tesseract-ocr \
    tesseract-ocr-tur \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    libtesseract-dev \
    hunspell \
    hunspell-tr \
    libhunspell-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-compatible Python packages (core)
# NumPy 1.24.3 is the last version without mandatory AVX2 CPU instructions
RUN pip install --no-cache-dir \
    pytesseract==0.3.10 \
    Pillow==10.2.0 \
    numpy==1.24.3 \
    opencv-python-headless==4.8.1.78

# PyTorch 1.13.1+cpu — last version without AVX2 requirement
RUN pip install --no-cache-dir \
    torch==1.13.1+cpu torchvision==0.14.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# EasyOCR without its torch/torchvision deps (already installed above)
RUN pip install --no-cache-dir --no-deps easyocr

# Transformers for TrOCR + EasyOCR transitive deps
RUN pip install --no-cache-dir \
    transformers==4.30.2 \
    scikit-image \
    scipy \
    pyclipper \
    shapely \
    python-bidi \
    PyYAML

# PaddleOCR engine + Hunspell Python bindings
RUN pip install --no-cache-dir \
    paddlepaddle==2.5.2 \
    paddleocr==2.7.3 \
    pyhunspell

WORKDIR /app

COPY --from=builder /app/ocr-server .

COPY scripts/ ./scripts/

RUN mkdir -p /app/models /app/uploads

ENV PORT=8080 \
    MODEL_PATH=/app/models \
    GIN_MODE=release \
    PYTHON_PATH=python3 \
    NNPACK_DISABLE=1 \
    OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD wget --no-verbose --tries=1 -O /dev/null http://localhost:8080/health || exit 1

CMD ["./ocr-server"]
