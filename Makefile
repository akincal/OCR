.PHONY: help build run test clean docker-build docker-run download-models install-deps

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-deps: ## Install Go dependencies
	@echo "Installing Go dependencies..."
	go mod download
	go mod tidy
	@echo "✓ Dependencies installed"

download-models: ## Download TrOCR models
	@echo "Downloading TrOCR models..."
	./scripts/download_models.sh

build: ## Build the application
	@echo "Building OCR API server..."
	CGO_ENABLED=1 go build -o bin/ocr-server ./cmd/server
	@echo "✓ Build complete: bin/ocr-server"

run: ## Run the application
	@echo "Starting OCR API server..."
	go run ./cmd/server/main.go

test: ## Run tests
	@echo "Running tests..."
	go test -v ./...

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -rf uploads/
	@echo "✓ Clean complete"

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t ocr-api:latest .
	@echo "✓ Docker image built"

docker-run: ## Run Docker container
	@echo "Starting Docker container..."
	docker-compose up -d
	@echo "✓ Container started"
	@echo "API available at http://localhost:8080"

docker-stop: ## Stop Docker container
	@echo "Stopping Docker container..."
	docker-compose down
	@echo "✓ Container stopped"

docker-logs: ## Show Docker logs
	docker-compose logs -f

setup: install-deps download-models ## Complete setup (install deps + download models)
	@echo "✓ Setup complete!"
	@echo ""
	@echo "To start the server:"
	@echo "  make run          # Run locally"
	@echo "  make docker-run   # Run in Docker"

dev: ## Run in development mode with hot reload
	@echo "Starting development server..."
	@command -v air > /dev/null 2>&1 || (echo "Installing air..." && go install github.com/cosmtrek/air@latest)
	air

fmt: ## Format Go code
	@echo "Formatting code..."
	go fmt ./...
	@echo "✓ Code formatted"

lint: ## Run linter
	@echo "Running linter..."
	@command -v golangci-lint > /dev/null 2>&1 || (echo "golangci-lint not installed" && exit 1)
	golangci-lint run
	@echo "✓ Linting complete"
