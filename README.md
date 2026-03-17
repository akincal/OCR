# OCR API - Handwriting Text Recognition

A powerful REST API for Optical Character Recognition (OCR) specifically optimized for handwritten text extraction. Built with Go and powered by Microsoft's TrOCR (Transformer-based OCR) model running on CPU via ONNX Runtime.

## Features

- **Handwriting Recognition**: State-of-the-art accuracy for handwritten text using TrOCR
- **Image Preprocessing**: Advanced preprocessing pipeline including:
  - Noise reduction
  - Contrast enhancement (CLAHE)
  - Adaptive thresholding
  - Automatic deskewing
  - Smart polarity detection
- **Confidence Scores**: Returns confidence levels for OCR results
- **Multiple Input Formats**: Support for file upload and base64 encoded images
- **Batch Processing**: Process multiple images in a single request
- **CPU Optimized**: Runs efficiently on CPU without requiring GPU
- **RESTful API**: Clean and easy-to-use REST API
- **Docker Support**: Ready-to-deploy Docker container

## Quick Start

### Prerequisites

- Go 1.21 or later
- OpenCV 4.x
- ONNX Runtime 1.16+
- Python 3.8+ (for model download script)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OCR
```

2. Install dependencies:
```bash
make install-deps
```

3. Download TrOCR models:
```bash
make download-models
```

4. Run the server:
```bash
make run
```

The API will be available at `http://localhost:8080`

### Using Docker

The easiest way to run the API is using Docker:

```bash
# Build and run with Docker Compose
make docker-build
make docker-run

# Or manually
docker build -t ocr-api .
docker run -p 8080:8080 -v $(pwd)/models:/app/models ocr-api
```

## API Documentation

### Endpoints

#### Health Check
```bash
GET /health
```

Returns API health status and model information.

#### OCR with File Upload
```bash
POST /api/v1/ocr
Content-Type: multipart/form-data

Parameters:
- image: Image file (required)
- enable_deskew: Enable automatic deskewing (default: true)
- return_processed: Return preprocessed image (default: false)
```

Example:
```bash
curl -X POST \
  -F "image=@handwritten.jpg" \
  -F "enable_deskew=true" \
  http://localhost:8080/api/v1/ocr
```

#### OCR with JSON/Base64
```bash
POST /api/v1/ocr/json
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "enable_deskew": true,
  "return_processed": false
}
```

Example:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "enable_deskew": true}' \
  http://localhost:8080/api/v1/ocr/json
```

#### Batch OCR
```bash
POST /api/v1/ocr/batch
Content-Type: multipart/form-data

Parameters:
- images: Multiple image files (up to 10)
```

Example:
```bash
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  http://localhost:8080/api/v1/ocr/batch
```

#### Model Information
```bash
GET /api/v1/model/info
```

Returns information about the loaded TrOCR model.

### Response Format

Success response:
```json
{
  "success": true,
  "result": {
    "text": "Recognized text from the image",
    "confidence": 0.95,
    "language": "en"
  },
  "processing_time_ms": 234.56
}
```

Error response:
```json
{
  "success": false,
  "error": "Error description",
  "processing_time_ms": 12.34
}
```

## Project Structure

```
OCR/
├── cmd/
│   └── server/           # Main application entry point
│       └── main.go
├── internal/
│   ├── ocr/             # TrOCR engine and inference
│   │   └── trocr.go
│   ├── preprocessing/   # Image preprocessing
│   │   └── image.go
│   └── models/          # Model definitions
├── pkg/
│   └── api/             # API handlers and routes
│       ├── handlers.go
│       └── routes.go
├── configs/             # Configuration files
├── scripts/             # Setup and utility scripts
│   ├── download_models.sh
│   └── test_api.sh
├── examples/            # Example clients
│   └── client.py
├── models/              # TrOCR model files (gitignored)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

## Configuration

Configuration is done via environment variables:

```bash
# Server
PORT=8080                    # Server port
GIN_MODE=release            # Gin mode (debug/release)

# Model
MODEL_PATH=./models         # Path to model files

# Processing
MAX_UPLOAD_SIZE=10485760    # Max upload size (10MB)
MAX_BATCH_SIZE=10           # Max batch size
```

Create a `.env` file from the example:
```bash
cp .env.example .env
```

## Development

### Running Tests
```bash
make test
```

### Testing the API
```bash
# Basic API test
./scripts/test_api.sh

# Test with an image
TEST_IMAGE=./sample.jpg ./scripts/test_api.sh
```

### Using the Python Client
```bash
python3 examples/client.py path/to/image.jpg
```

## Model Information

This API uses **Microsoft TrOCR** (Transformer-based OCR), specifically the `trocr-base-handwritten` variant which is optimized for handwritten text recognition.

### Model Details
- **Base Model**: microsoft/trocr-base-handwritten
- **Architecture**: Vision Transformer (ViT) encoder + Transformer decoder
- **Input Size**: 384x384 pixels
- **Runtime**: ONNX Runtime (CPU optimized)

### Downloading Models

The models are downloaded automatically using the setup script:
```bash
./scripts/download_models.sh
```

This script:
1. Downloads the TrOCR model from Hugging Face
2. Converts it to ONNX format for CPU inference
3. Saves the models to the `./models` directory

Models are approximately 500MB in size.

## Performance

Performance metrics on standard hardware (4-core CPU):

- Single image processing: ~200-300ms
- Batch processing (5 images): ~800-1000ms
- Memory usage: ~500MB-1GB (depending on batch size)

## Supported Image Formats

- JPEG / JPG
- PNG
- BMP
- TIFF / TIF

## Limitations

- Maximum image upload size: 10MB
- Maximum batch size: 10 images
- CPU-only inference (no GPU acceleration in current version)
- English language focus (can be extended to other languages)

## Troubleshooting

### Models not loading
Ensure model files are present:
```bash
ls -lh models/
# Should show: encoder_model.onnx, decoder_model.onnx
```

### OpenCV errors
Install OpenCV development libraries:
```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# macOS
brew install opencv
```

### Go build errors
Ensure CGO is enabled:
```bash
export CGO_ENABLED=1
go build ./cmd/server
```


## License

[MIT License](LICENSE)

## Acknowledgments

- [Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) - Transformer-based OCR model
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
- [GoCV](https://gocv.io/) - Go bindings for OpenCV
- [Gin](https://gin-gonic.com/) - Web framework


