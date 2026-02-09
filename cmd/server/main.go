package main

import (
	"fmt"
	"log"
	"ocr-api/internal/ocr"
	"ocr-api/pkg/api"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/gin-gonic/gin"
)

func main() {
	port := getEnv("PORT", "8080")
	ginMode := getEnv("GIN_MODE", "debug")
	pythonPath := getEnv("PYTHON_PATH", "python3")

	gin.SetMode(ginMode)
	printBanner()

	// Determine script path
	execDir, _ := os.Getwd()
	scriptPath := filepath.Join(execDir, "scripts", "ocr_inference.py")

	// Initialize TrOCR engine (starts Python inference server)
	log.Println("Initializing TrOCR engine...")
	log.Println("Python script:", scriptPath)

	ocrEngine := ocr.NewTrOCREngine(pythonPath, scriptPath)

	// Start OCR engine (downloads model on first run)
	log.Println("Starting Python OCR server (model will be downloaded on first run)...")
	log.Println("This may take a few minutes on first launch...")
	if err := ocrEngine.Initialize(); err != nil {
		log.Printf("WARNING: OCR engine failed to start: %v", err)
		log.Println("API will start but OCR requests will fail.")
		log.Println("Make sure Python dependencies are installed:")
		log.Println("  pip3 install torch transformers Pillow")
	} else {
		log.Println("TrOCR engine initialized successfully")
	}

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down...")
		ocrEngine.Close()
		os.Exit(0)
	}()

	// Create handler and router
	handler := api.NewHandler(ocrEngine)
	router := gin.Default()
	api.SetupMiddleware(router)
	api.SetupRoutes(router, handler)

	printEndpoints()

	address := fmt.Sprintf("0.0.0.0:%s", port)
	log.Printf("OCR API server running at http://localhost:%s", port)

	if err := router.Run(address); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func printBanner() {
	fmt.Println(`
  ___   ____ ____      _    ____ ___
 / _ \ / ___|  _ \    / \  |  _ \_ _|
| | | | |   | |_) |  / _ \ | |_) | |
| |_| | |___|  _ <  / ___ \|  __/| |
 \___/ \____|_| \_\/_/   \_\_|  |___|

 Handwriting OCR API - TrOCR + Go
 Version: 1.0.0
`)
}

func printEndpoints() {
	fmt.Println(`
Endpoints:
  GET  /                    API info
  GET  /health              Health check
  GET  /api/v1/model/info   Model info
  POST /api/v1/ocr          OCR (file upload)
  POST /api/v1/ocr/json     OCR (base64 JSON)
  POST /api/v1/ocr/batch    Batch OCR
`)
}
