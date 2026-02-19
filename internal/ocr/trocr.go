package ocr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
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
	processAlive  bool
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

	if e.initialized && e.processAlive {
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
	e.processAlive = true
	return nil
}

// startServer launches the Python OCR inference server
func (e *TrOCREngine) startServer() error {
	// Kill any previous process
	if e.serverProcess != nil && e.serverProcess.Process != nil {
		_ = e.serverProcess.Process.Kill()
		_ = e.serverProcess.Wait()
		e.serverProcess = nil
	}

	cmd := exec.Command(e.pythonPath, e.scriptPath, "--server", "--port", "5555")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start OCR server: %w", err)
	}
	e.serverProcess = cmd
	e.processAlive = true
	log.Printf("[OCR] Python server process started (PID: %d)", cmd.Process.Pid)

	// Monitor the process in background — detect crashes
	go func() {
		err := cmd.Wait()
		e.mu.Lock()
		e.processAlive = false
		e.initialized = false
		e.mu.Unlock()
		if err != nil {
			log.Printf("[OCR] Python server process (PID: %d) exited with error: %v", cmd.Process.Pid, err)
		} else {
			log.Printf("[OCR] Python server process (PID: %d) exited normally", cmd.Process.Pid)
		}
	}()

	// Wait for server to be ready
	maxRetries := 90 // Model loading can take a while (up to 3 minutes)
	for i := 0; i < maxRetries; i++ {
		time.Sleep(2 * time.Second)

		// Check if process is still alive
		if !e.processAlive {
			return fmt.Errorf("OCR server process crashed during startup")
		}

		resp, err := http.Get(e.serverURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				e.initialized = true
				log.Printf("[OCR] Python server is ready at %s", e.serverURL)
				return nil
			}
		}
	}

	return fmt.Errorf("OCR server failed to start within timeout")
}

// ensureServer checks if the Python server is alive and restarts it if needed
func (e *TrOCREngine) ensureServer() error {
	e.mu.Lock()
	alive := e.processAlive
	e.mu.Unlock()

	if alive {
		// Quick health check
		client := &http.Client{Timeout: 3 * time.Second}
		resp, err := client.Get(e.serverURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
	}

	// Server is down — restart it
	log.Println("[OCR] Python server is not responding, restarting...")
	e.mu.Lock()
	e.initialized = false
	e.processAlive = false
	e.mu.Unlock()

	// Re-initialize (will call startServer)
	return e.Initialize()
}

// RecognizeFromBytes performs OCR on image bytes via HTTP to Python server
func (e *TrOCREngine) RecognizeFromBytes(imageBytes []byte) (*OCRResult, error) {
	if e.serverMode {
		// Ensure the Python server is alive; restart if needed
		if err := e.ensureServer(); err != nil {
			return nil, fmt.Errorf("OCR server unavailable: %w", err)
		}

		result, err := e.recognizeViaServer(imageBytes)
		if err != nil {
			// Connection may have been lost mid-request — try one restart
			log.Printf("[OCR] Request failed, attempting server restart: %v", err)
			if restartErr := e.ensureServer(); restartErr != nil {
				return nil, fmt.Errorf("OCR server restart failed: %w", restartErr)
			}
			return e.recognizeViaServer(imageBytes)
		}
		return result, nil
	}

	if !e.initialized {
		return nil, fmt.Errorf("engine not initialized")
	}
	return e.recognizeViaSubprocess(imageBytes)
}

// recognizeViaServer sends image to Python HTTP server
func (e *TrOCREngine) recognizeViaServer(imageBytes []byte) (*OCRResult, error) {
	// Use a dedicated client with a long timeout — OCR on CPU can take minutes
	client := &http.Client{
		Timeout: 5 * time.Minute,
		Transport: &http.Transport{
			DisableKeepAlives: true, // Use fresh connection each time
		},
	}

	req, err := http.NewRequest("POST", e.serverURL+"/ocr", bytes.NewReader(imageBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Connection", "close")

	log.Printf("[OCR] Sending %d bytes to Python server...", len(imageBytes))
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call OCR server: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	log.Printf("[OCR] Got response (%d bytes, status %d)", len(body), resp.StatusCode)

	var result OCRResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response (body=%q): %w", string(body[:min(len(body), 200)]), err)
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
