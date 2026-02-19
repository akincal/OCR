# Multi-stage build for OCR API

# Stage 1: Build Go binary
FROM golang:1.21-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /app/ocr-server ./cmd/server

# Stage 2: Runtime stage with Python
# Use Python 3.10 because torch 1.13.1 (last version without AVX2 requirement) supports 3.7-3.10
FROM python:3.10-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# torch 1.13.1+cpu is the last version that does NOT require AVX2 CPU instructions.
# PyTorch 2.x wheels use AVX2 which causes SIGILL on older server CPUs / VMs without AVX2.
RUN pip install --no-cache-dir \
    torch==1.13.1+cpu torchvision==0.14.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
# Install easyocr without its torch/torchvision deps (already installed above)
RUN pip install --no-cache-dir --no-deps easyocr
# Install remaining dependencies (easyocr deps + transformers for TrOCR)
RUN pip install --no-cache-dir \
    transformers==4.30.2 \
    Pillow \
    opencv-python-headless \
    numpy \
    scikit-image \
    scipy \
    pyclipper \
    shapely \
    python-bidi \
    PyYAML

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/ocr-server .

# Copy scripts directory (needed by Python OCR inference)
COPY scripts/ ./scripts/

# Create models and uploads directories
RUN mkdir -p /app/models /app/uploads

# Set environment variables
ENV PORT=8080 \
    MODEL_PATH=/app/models \
    GIN_MODE=release \
    PYTHON_PATH=python3 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD wget --no-verbose --tries=1 -O /dev/null http://localhost:8080/health || exit 1

# Run the application
CMD ["./ocr-server"]
