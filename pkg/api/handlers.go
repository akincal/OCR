package api

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"ocr-api/internal/ocr"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// Handler holds dependencies for API handlers
type Handler struct {
	ocrEngine *ocr.TrOCREngine
}

// NewHandler creates a new API handler
func NewHandler(ocrEngine *ocr.TrOCREngine) *Handler {
	return &Handler{
		ocrEngine: ocrEngine,
	}
}

// OCRRequest represents the OCR request body (JSON/base64)
type OCRRequest struct {
	Image string `json:"image" binding:"required"`
}

// OCRResponse represents the OCR response
type OCRResponse struct {
	Success        bool           `json:"success"`
	Result         *ocr.OCRResult `json:"result,omitempty"`
	ProcessingTime float64        `json:"processing_time_ms"`
	Error          string         `json:"error,omitempty"`
}

// BatchOCRResponse represents batch OCR response
type BatchOCRResponse struct {
	Success        bool             `json:"success"`
	Results        []*ocr.OCRResult `json:"results,omitempty"`
	ProcessingTime float64          `json:"processing_time_ms"`
	TotalImages    int              `json:"total_images"`
	Error          string           `json:"error,omitempty"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status    string                 `json:"status"`
	Timestamp string                 `json:"timestamp"`
	Version   string                 `json:"version"`
	ModelInfo map[string]interface{} `json:"model_info,omitempty"`
}

// HealthCheck handles health check endpoint
func (h *Handler) HealthCheck(c *gin.Context) {
	status := "healthy"
	if !h.ocrEngine.IsReady() {
		status = "degraded - OCR engine not ready"
	}

	c.JSON(http.StatusOK, HealthResponse{
		Status:    status,
		Timestamp: time.Now().Format(time.RFC3339),
		Version:   "1.0.0",
		ModelInfo: h.ocrEngine.GetModelInfo(),
	})
}

// RecognizeText handles OCR request for uploaded image
func (h *Handler) RecognizeText(c *gin.Context) {
	startTime := time.Now()

	// Parse multipart form
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, OCRResponse{
			Success:        false,
			Error:          "No image file provided. Use form field 'image'",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}
	defer file.Close()

	// Validate file type
	if !isValidImageType(header.Filename) {
		c.JSON(http.StatusBadRequest, OCRResponse{
			Success:        false,
			Error:          "Invalid image format. Supported: jpg, jpeg, png, bmp, tiff",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	// Read file bytes
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, OCRResponse{
			Success:        false,
			Error:          "Failed to read image file",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	// Perform OCR
	result, err := h.ocrEngine.RecognizeFromBytes(fileBytes)
	if err != nil {
		c.JSON(http.StatusInternalServerError, OCRResponse{
			Success:        false,
			Error:          fmt.Sprintf("OCR recognition failed: %v", err),
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	c.JSON(http.StatusOK, OCRResponse{
		Success:        true,
		Result:         result,
		ProcessingTime: float64(time.Since(startTime).Milliseconds()),
	})
}

// RecognizeTextBatch handles batch OCR requests
func (h *Handler) RecognizeTextBatch(c *gin.Context) {
	startTime := time.Now()

	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, BatchOCRResponse{
			Success:        false,
			Error:          "Failed to parse multipart form",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	files := form.File["images"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, BatchOCRResponse{
			Success:        false,
			Error:          "No image files provided. Use form field 'images'",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	maxBatchSize := 10
	if len(files) > maxBatchSize {
		c.JSON(http.StatusBadRequest, BatchOCRResponse{
			Success:        false,
			Error:          fmt.Sprintf("Batch size exceeds maximum of %d images", maxBatchSize),
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	var results []*ocr.OCRResult
	for _, fileHeader := range files {
		file, err := fileHeader.Open()
		if err != nil {
			continue
		}
		fileBytes, err := io.ReadAll(file)
		file.Close()
		if err != nil {
			continue
		}

		result, err := h.ocrEngine.RecognizeFromBytes(fileBytes)
		if err != nil {
			results = append(results, &ocr.OCRResult{
				Success: false,
				Error:   err.Error(),
			})
			continue
		}
		results = append(results, result)
	}

	c.JSON(http.StatusOK, BatchOCRResponse{
		Success:        true,
		Results:        results,
		ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		TotalImages:    len(results),
	})
}

// RecognizeTextJSON handles OCR request with base64 encoded image
func (h *Handler) RecognizeTextJSON(c *gin.Context) {
	startTime := time.Now()

	var req OCRRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, OCRResponse{
			Success:        false,
			Error:          "Invalid request format. Expected JSON with 'image' field (base64)",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	// Decode base64 image
	imageData := req.Image
	if strings.HasPrefix(imageData, "data:image") {
		parts := strings.SplitN(imageData, ",", 2)
		if len(parts) == 2 {
			imageData = parts[1]
		}
	}

	imageBytes, err := base64.StdEncoding.DecodeString(imageData)
	if err != nil {
		c.JSON(http.StatusBadRequest, OCRResponse{
			Success:        false,
			Error:          "Invalid base64 image data",
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	// Perform OCR
	result, err := h.ocrEngine.RecognizeFromBytes(imageBytes)
	if err != nil {
		c.JSON(http.StatusInternalServerError, OCRResponse{
			Success:        false,
			Error:          fmt.Sprintf("OCR recognition failed: %v", err),
			ProcessingTime: float64(time.Since(startTime).Milliseconds()),
		})
		return
	}

	c.JSON(http.StatusOK, OCRResponse{
		Success:        true,
		Result:         result,
		ProcessingTime: float64(time.Since(startTime).Milliseconds()),
	})
}

// GetModelInfo returns information about the OCR model
func (h *Handler) GetModelInfo(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"model":   h.ocrEngine.GetModelInfo(),
	})
}

func isValidImageType(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	validExts := []string{".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
	for _, validExt := range validExts {
		if ext == validExt {
			return true
		}
	}
	return false
}
