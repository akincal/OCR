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

# Build NNPACK stub — provides a no-op nnp_initialize that returns "unsupported hardware"
# This prevents SIGILL crashes on CPUs that don't support NNPACK's SIMD instructions
RUN printf '#include <stddef.h>\n\
enum nnp_status { nnp_status_success = 0, nnp_status_unsupported_hardware = 12 };\n\
enum nnp_status nnp_initialize(void) { return nnp_status_unsupported_hardware; }\n\
int nnp_deinitialize(void) { return 0; }\n\
' > /tmp/stub_nnpack.c && \
    gcc -shared -fPIC -o /tmp/libstub_nnpack.so /tmp/stub_nnpack.c

# Stage 2: Runtime stage with Python
FROM python:3.11-slim-bookworm

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
# Install torch + torchvision together from CPU index to ensure version compatibility
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Install easyocr without its torch/torchvision deps (already installed above)
RUN pip install --no-cache-dir --no-deps easyocr
# Install remaining dependencies
RUN pip install --no-cache-dir \
    transformers \
    Pillow \
    opencv-python-headless \
    numpy \
    scikit-image \
    scipy \
    pyclipper \
    shapely \
    python-bidi \
    PyYAML \
    ninja

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/ocr-server .

# Copy NNPACK stub library from builder
COPY --from=builder /tmp/libstub_nnpack.so /usr/local/lib/libstub_nnpack.so

# Copy scripts directory (needed by Python OCR inference)
COPY scripts/ ./scripts/

# Create models and uploads directories
RUN mkdir -p /app/models /app/uploads

# Set environment variables
ENV PORT=8080 \
    MODEL_PATH=/app/models \
    GIN_MODE=release \
    PYTHON_PATH=python3 \
    LD_PRELOAD=/usr/local/lib/libstub_nnpack.so \
    NNPACK_DISABLE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD wget --no-verbose --tries=1 -O /dev/null http://localhost:8080/health || exit 1

# Run the application
CMD ["./ocr-server"]
