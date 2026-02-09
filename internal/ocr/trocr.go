package ocr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// OCRResult contains OCR output with confidence
type OCRResult struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	Lines      int     `json:"lines,omitempty"`
	Success    bool    `json:"success"`
	Error      string  `json:"error,omitempty"`
}

// TrOCREngine handles TrOCR inference via Python subprocess
type TrOCREngine struct {
	pythonPath    string
	scriptPath    string
	serverURL     string
	serverMode    bool
	serverProcess *exec.Cmd
	initialized   bool
	mu            sync.Mutex
}

// NewTrOCREngine creates a new TrOCR engine
func NewTrOCREngine(pythonPath, scriptPath string) *TrOCREngine {
	if pythonPath == "" {
		pythonPath = "python3"
	}
	return &TrOCREngine{
		pythonPath: pythonPath,
		scriptPath: scriptPath,
		serverURL:  "http://127.0.0.1:5555",
		serverMode: true,
	}
}

// Initialize starts the Python OCR server
func (e *TrOCREngine) Initialize() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.initialized {
		return nil
	}

	// Verify python and script exist
	if _, err := exec.LookPath(e.pythonPath); err != nil {
		return fmt.Errorf("python3 not found: %w", err)
	}
	if _, err := os.Stat(e.scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("OCR script not found: %s", e.scriptPath)
	}

	if e.serverMode {
		return e.startServer()
	}

	e.initialized = true
	return nil
}

// startServer launches the Python OCR inference server
func (e *TrOCREngine) startServer() error {
	cmd := exec.Command(e.pythonPath, e.scriptPath, "--server", "--port", "5555")
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start OCR server: %w", err)
	}
	e.serverProcess = cmd

	// Wait for server to be ready
	maxRetries := 60 // Model loading can take a while
	for i := 0; i < maxRetries; i++ {
		time.Sleep(2 * time.Second)

		resp, err := http.Get(e.serverURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				e.initialized = true
				return nil
			}
		}
	}

	return fmt.Errorf("OCR server failed to start within timeout")
}

// RecognizeFromBytes performs OCR on image bytes via HTTP to Python server
func (e *TrOCREngine) RecognizeFromBytes(imageBytes []byte) (*OCRResult, error) {
	if !e.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}

	if e.serverMode {
		return e.recognizeViaServer(imageBytes)
	}
	return e.recognizeViaSubprocess(imageBytes)
}

// recognizeViaServer sends image to Python HTTP server
func (e *TrOCREngine) recognizeViaServer(imageBytes []byte) (*OCRResult, error) {
	resp, err := http.Post(e.serverURL+"/ocr", "application/octet-stream", bytes.NewReader(imageBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to call OCR server: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var result OCRResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

// recognizeViaSubprocess runs Python script as a one-shot subprocess
func (e *TrOCREngine) recognizeViaSubprocess(imageBytes []byte) (*OCRResult, error) {
	// Write image to temp file
	tmpFile, err := os.CreateTemp("", "ocr-*.png")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(imageBytes); err != nil {
		tmpFile.Close()
		return nil, fmt.Errorf("failed to write temp file: %w", err)
	}
	tmpFile.Close()

	// Run Python script
	cmd := exec.Command(e.pythonPath, e.scriptPath, tmpFile.Name())
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("OCR inference failed: %w", err)
	}

	var result OCRResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse OCR output: %w", err)
	}

	return &result, nil
}

// RecognizeFromFile performs OCR on an image file
func (e *TrOCREngine) RecognizeFromFile(imagePath string) (*OCRResult, error) {
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read image file: %w", err)
	}
	return e.RecognizeFromBytes(imageBytes)
}

// Close stops the Python server
func (e *TrOCREngine) Close() error {
	if e.serverProcess != nil {
		return e.serverProcess.Process.Kill()
	}
	return nil
}

// GetModelInfo returns information about the model
func (e *TrOCREngine) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model":       "microsoft/trocr-base-handwritten",
		"runtime":     "Python + PyTorch",
		"script":      filepath.Base(e.scriptPath),
		"server_mode": e.serverMode,
		"server_url":  e.serverURL,
		"initialized": e.initialized,
	}
}

// IsReady checks if the engine is ready
func (e *TrOCREngine) IsReady() bool {
	return e.initialized
}
