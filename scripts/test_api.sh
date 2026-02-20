#!/bin/bash

# Test script for Multi-Engine OCR API

set -e

API_URL="${API_URL:-http://localhost:8080}"
ENGINE="${ENGINE:-tesseract}"

echo "╔════════════════════════════════════════════════════════╗"
echo "║            OCR API Test Script (v2.0)                  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Testing API at: $API_URL"
echo "Engine: $ENGINE"
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "─────────────────────────────────────────────────────────"
response=$(curl -s -w "\n%{http_code}" "$API_URL/health")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "  Health check passed"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo "  Health check failed (HTTP $http_code)"
    echo "$body"
fi
echo ""

# Test 2: Model info
echo "Test 2: Model Information"
echo "─────────────────────────────────────────────────────────"
response=$(curl -s -w "\n%{http_code}" "$API_URL/api/v1/model/info")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "  Model info retrieved"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo "  Model info failed (HTTP $http_code)"
    echo "$body"
fi
echo ""

# Test 3: OCR with sample image (all engines)
if [ -n "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE" ]; then
    echo "Test 3: OCR Processing (Multi-Engine Comparison)"
    echo "─────────────────────────────────────────────────────────"
    echo "Image: $TEST_IMAGE"
    echo ""

    for eng in tesseract easyocr paddleocr ensemble; do
        echo "  --- Engine: $eng ---"
        response=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -F "image=@$TEST_IMAGE" \
            "$API_URL/api/v1/ocr?engine=$eng" 2>/dev/null)

        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | head -n-1)

        if [ "$http_code" -eq 200 ]; then
            confidence=$(echo "$body" | jq -r '.result.confidence // "N/A"' 2>/dev/null)
            text=$(echo "$body" | jq -r '.result.text // "N/A"' 2>/dev/null)
            proc_time=$(echo "$body" | jq -r '.processing_time_ms // "N/A"' 2>/dev/null)
            echo "  Confidence: $confidence"
            echo "  Time: ${proc_time}ms"
            echo "  Text: $(echo "$text" | head -3)"
        else
            echo "  Failed (HTTP $http_code)"
        fi
        echo ""
    done
else
    echo "Test 3: OCR Processing (Skipped)"
    echo "─────────────────────────────────────────────────────────"
    echo "  No test image provided. Set TEST_IMAGE env var to test."
    echo "  Example: TEST_IMAGE=./sample.jpg ./scripts/test_api.sh"
    echo ""
fi

# Test 4: OCR with JSON/base64
echo "Test 4: OCR with JSON/Base64 (engine=$ENGINE)"
echo "─────────────────────────────────────────────────────────"

TEST_BASE64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

response=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"data:image/png;base64,$TEST_BASE64\"}" \
    "$API_URL/api/v1/ocr/json?engine=$ENGINE")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "  JSON/Base64 OCR completed"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo "  JSON/Base64 OCR failed (HTTP $http_code)"
    echo "$body"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "Tests completed!"
echo ""
echo "Usage examples:"
echo "  # Test all engines with an image:"
echo "  TEST_IMAGE=./sample.jpg ./scripts/test_api.sh"
echo ""
echo "  # Test a specific engine:"
echo "  ENGINE=ensemble TEST_IMAGE=./sample.jpg ./scripts/test_api.sh"
echo ""
echo "  # Test against different server:"
echo "  API_URL=http://server:8080 ./scripts/test_api.sh"
echo "════════════════════════════════════════════════════════"
