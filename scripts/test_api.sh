#!/bin/bash

# Test script for OCR API

set -e

API_URL="${API_URL:-http://localhost:8080}"

echo "╔════════════════════════════════════════════════════════╗"
echo "║            OCR API Test Script                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Testing API at: $API_URL"
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "─────────────────────────────────────────────────────────"
response=$(curl -s -w "\n%{http_code}" "$API_URL/health")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "✓ Health check passed"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo "✗ Health check failed (HTTP $http_code)"
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
    echo "✓ Model info retrieved"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo "✗ Model info failed (HTTP $http_code)"
    echo "$body"
fi
echo ""

# Test 3: OCR with sample image (if image file is provided)
if [ -n "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE" ]; then
    echo "Test 3: OCR Processing"
    echo "─────────────────────────────────────────────────────────"
    echo "Image: $TEST_IMAGE"

    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -F "image=@$TEST_IMAGE" \
        -F "enable_deskew=true" \
        "$API_URL/api/v1/ocr")

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" -eq 200 ]; then
        echo "✓ OCR processing completed"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        echo "✗ OCR processing failed (HTTP $http_code)"
        echo "$body"
    fi
    echo ""
else
    echo "Test 3: OCR Processing (Skipped)"
    echo "─────────────────────────────────────────────────────────"
    echo "ℹ No test image provided. Set TEST_IMAGE environment variable to test."
    echo "Example: TEST_IMAGE=./sample.jpg ./scripts/test_api.sh"
    echo ""
fi

# Test 4: OCR with JSON/base64 (example)
echo "Test 4: OCR with JSON/Base64"
echo "─────────────────────────────────────────────────────────"

# Create a simple test image (1x1 white pixel as base64)
TEST_BASE64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

response=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"data:image/png;base64,$TEST_BASE64\", \"enable_deskew\": true}" \
    "$API_URL/api/v1/ocr/json")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" -eq 200 ]; then
    echo "✓ JSON/Base64 OCR completed"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo "✗ JSON/Base64 OCR failed (HTTP $http_code)"
    echo "$body"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "Tests completed!"
echo ""
echo "Usage examples:"
echo "  # Test with custom image:"
echo "  TEST_IMAGE=./path/to/image.jpg ./scripts/test_api.sh"
echo ""
echo "  # Test against different server:"
echo "  API_URL=http://localhost:3000 ./scripts/test_api.sh"
echo "════════════════════════════════════════════════════════"
