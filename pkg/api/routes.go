package api

import (
	"github.com/gin-gonic/gin"
)

// SetupRoutes configures all API routes
func SetupRoutes(router *gin.Engine, handler *Handler) {
	// Health check
	router.GET("/health", handler.HealthCheck)
	router.HEAD("/health", handler.HealthCheck)
	router.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "OCR API Server",
			"version": "1.0.0",
			"endpoints": []string{
				"GET  /health - Health check",
				"GET  /api/v1/model/info - Model information",
				"POST /api/v1/ocr - OCR with file upload",
				"POST /api/v1/ocr/json - OCR with JSON/base64",
				"POST /api/v1/ocr/batch - Batch OCR processing",
			},
		})
	})

	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// OCR endpoints
		ocr := v1.Group("/ocr")
		{
			// Single image OCR with file upload
			ocr.POST("", handler.RecognizeText)

			// Single image OCR with JSON body (base64)
			ocr.POST("/json", handler.RecognizeTextJSON)

			// Batch OCR
			ocr.POST("/batch", handler.RecognizeTextBatch)
		}

		// Model information
		model := v1.Group("/model")
		{
			model.GET("/info", handler.GetModelInfo)
		}
	}
}

// SetupMiddleware configures middleware
func SetupMiddleware(router *gin.Engine) {
	// CORS middleware
	router.Use(corsMiddleware())

	// Request logging
	router.Use(gin.Logger())

	// Recovery middleware
	router.Use(gin.Recovery())

	// Size limit middleware (10MB)
	router.MaxMultipartMemory = 10 << 20 // 10 MB
}

// corsMiddleware adds CORS headers
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}
