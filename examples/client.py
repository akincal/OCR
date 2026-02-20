#!/usr/bin/env python3
"""
Example Python client for OCR API
Demonstrates how to use the OCR API from Python
"""

import requests
import base64
import json
import sys
from pathlib import Path


class OCRClient:
    """Simple client for OCR API"""

    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self):
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_model_info(self):
        """Get model information"""
        response = self.session.get(f"{self.base_url}/api/v1/model/info")
        response.raise_for_status()
        return response.json()

    def ocr_from_file(self, image_path, enable_deskew=True, return_processed=False):
        """
        Perform OCR on an image file

        Args:
            image_path: Path to image file
            enable_deskew: Whether to enable deskewing
            return_processed: Whether to return processed image

        Returns:
            OCR result dictionary
        """
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "enable_deskew": str(enable_deskew).lower(),
                "return_processed": str(return_processed).lower(),
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/ocr", files=files, data=data
            )
            response.raise_for_status()
            return response.json()

    def ocr_from_base64(self, image_base64, enable_deskew=True, return_processed=False):
        """
        Perform OCR on base64 encoded image

        Args:
            image_base64: Base64 encoded image string
            enable_deskew: Whether to enable deskewing
            return_processed: Whether to return processed image

        Returns:
            OCR result dictionary
        """
        payload = {
            "image": image_base64,
            "enable_deskew": enable_deskew,
            "return_processed": return_processed,
        }

        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/json",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def ocr_batch(self, image_paths):
        """
        Perform batch OCR on multiple images

        Args:
            image_paths: List of paths to image files

        Returns:
            Batch OCR result dictionary
        """
        files = []
        for path in image_paths:
            files.append(("images", open(path, "rb")))

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/ocr/batch", files=files
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close all file handles
            for _, f in files:
                f.close()


def main():
    """Example usage"""
    print("╔════════════════════════════════════════════════════════╗")
    print("║           OCR API Python Client Example               ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    # Initialize client
    client = OCRClient("http://localhost:8080")

    # Health check
    print("1. Health Check")
    print("─" * 60)
    try:
        health = client.health_check()
        print(f"✓ API is healthy")
        print(f"  Status: {health.get('status')}")
        print(f"  Version: {health.get('version')}")
        print()
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        print()
        return

    # Model info
    print("2. Model Information")
    print("─" * 60)
    try:
        info = client.get_model_info()
        print(f"✓ Model info retrieved")
        print(json.dumps(info, indent=2))
        print()
    except Exception as e:
        print(f"✗ Failed to get model info: {e}")
        print()

    # OCR from file
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

        if not Path(image_path).exists():
            print(f"✗ Image file not found: {image_path}")
            return

        print(f"3. OCR Processing")
        print("─" * 60)
        print(f"Image: {image_path}")

        try:
            result = client.ocr_from_file(
                image_path, enable_deskew=True, return_processed=False
            )

            if result.get("success"):
                print(f"✓ OCR completed")
                print(f"  Text: {result['result']['text']}")
                print(f"  Confidence: {result['result']['confidence']:.2%}")
                print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
            else:
                print(f"✗ OCR failed: {result.get('error')}")
            print()
        except Exception as e:
            print(f"✗ OCR processing failed: {e}")
            print()

        # Example: Convert image to base64 and use JSON endpoint
        print("4. OCR with Base64 Encoding")
        print("─" * 60)
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"

            result = client.ocr_from_base64(image_base64, enable_deskew=True)

            if result.get("success"):
                print(f"✓ OCR completed (base64)")
                print(f"  Text: {result['result']['text']}")
                print(f"  Confidence: {result['result']['confidence']:.2%}")
            else:
                print(f"✗ OCR failed: {result.get('error')}")
            print()
        except Exception as e:
            print(f"✗ Base64 OCR failed: {e}")
            print()

    else:
        print("3. OCR Processing (Skipped)")
        print("─" * 60)
        print("ℹ No image provided")
        print()
        print("Usage: python client.py <image_path>")
        print("Example: python client.py ./sample.jpg")
        print()

    print("═" * 60)
    print("Example completed!")
    print()


if __name__ == "__main__":
    main()
