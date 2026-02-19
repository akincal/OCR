#!/usr/bin/env python3
"""
OCR Inference Script — Tesseract Engine
With document detection, perspective correction, and post-processing.

Pipeline:
  1. Document/paper region detection (removes background noise)
  2. Perspective correction (fixes tilted papers)
  3. Image enhancement (contrast, sharpness, denoise)
  4. OCR via Tesseract (supports Turkish + English)
  5. Post-processing (spelling correction, confidence filtering)
"""

import sys
import os
import json
import argparse
import io
import math
import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import warnings
warnings.filterwarnings("ignore")

# Global state
_tesseract_verified = False


# ---------------------------------------------------------------------------
# Tesseract verification
# ---------------------------------------------------------------------------

def verify_tesseract():
    """Verify Tesseract is installed and Turkish language data is available."""
    global _tesseract_verified
    if _tesseract_verified:
        return

    import pytesseract
    # Check tesseract binary
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[PythonOCR] Tesseract version: {version}", file=sys.stderr, flush=True)
    except Exception as e:
        raise RuntimeError(f"Tesseract not found: {e}")

    # Check available languages
    langs = pytesseract.get_languages()
    print(f"[PythonOCR] Available languages: {langs}", file=sys.stderr, flush=True)
    if "tur" not in langs:
        print("[PythonOCR] WARNING: Turkish (tur) language data not installed!", file=sys.stderr, flush=True)

    _tesseract_verified = True


# ---------------------------------------------------------------------------
# 1) Document / paper region detection
# ---------------------------------------------------------------------------

def detect_document_region(image):
    """
    Detect the largest rectangular paper/document region in the image.
    Crops out background (keyboard, desk, etc.) automatically.
    Returns the cropped + perspective-corrected document image.
    """
    import numpy as np
    import cv2

    img_array = np.array(image)
    orig = img_array.copy()
    h, w = img_array.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, False

    # Find the largest contour that could be a document
    best_contour = None
    best_area = 0
    min_area = h * w * 0.05  # at least 5% of image

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # We want 4-sided polygons (documents are rectangular)
        if len(approx) == 4 and area > best_area:
            best_contour = approx
            best_area = area

    if best_contour is None:
        # Fallback: use the largest contour's bounding rect
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < min_area:
            return image, False

        # Use rotated bounding rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        best_contour = np.int32(box).reshape(4, 1, 2)

    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = best_contour.reshape(4, 2).astype(np.float32)
    ordered = order_points(pts)

    # Apply perspective transform
    corrected = four_point_transform(orig, ordered)

    from PIL import Image as PILImage
    return PILImage.fromarray(corrected), True


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    import numpy as np

    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff, bottom-left has largest diff
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(image, pts):
    """Apply perspective transform to get a top-down view of the document."""
    import numpy as np
    import cv2

    tl, tr, br, bl = pts

    # Compute new dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


# ---------------------------------------------------------------------------
# 2) Image preprocessing / enhancement
# ---------------------------------------------------------------------------

def preprocess_image(image):
    """Enhanced preprocessing pipeline."""
    from PIL import ImageEnhance, ImageOps, ImageFilter

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = ImageOps.exif_transpose(image)

    # Resize if too large
    max_dim = 2048
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, resample=3)

    # Denoise
    image = image.filter(ImageFilter.MedianFilter(size=3))

    # Auto-contrast (stretches histogram)
    image = ImageOps.autocontrast(image, cutoff=1)

    # Enhance contrast
    image = ImageEnhance.Contrast(image).enhance(1.6)

    # Enhance sharpness
    image = ImageEnhance.Sharpness(image).enhance(1.5)

    return image


def ensure_correct_orientation(image):
    """
    Check if the image is upside-down by comparing text density
    in the top vs bottom half. If bottom-heavy, rotate 180.
    """
    import numpy as np

    img_array = np.array(image.convert("L"))
    h = img_array.shape[0]

    # Binarize (dark pixels = text)
    binary = (img_array < 128).astype(np.uint8)

    top_density = np.sum(binary[:h // 3])
    bottom_density = np.sum(binary[2 * h // 3:])

    # If significantly more text at bottom, might be upside down
    # But this is a weak heuristic, so only rotate if very skewed
    if bottom_density > top_density * 3:
        return image.rotate(180, expand=True)

    return image


# ---------------------------------------------------------------------------
# 3) Post-processing
# ---------------------------------------------------------------------------

# Common Turkish words / city names for fuzzy matching
TURKISH_DICTIONARY = {
    # Cities
    "istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya",
    "gaziantep", "mersin", "diyarbakir", "kayseri", "eskisehir", "trabzon",
    "samsun", "denizli", "malatya", "erzurum", "van", "batman", "elazig",
    # Common company names
    "havelsan", "aselsan", "roketsan", "tusas", "tai", "baykar",
    # Currency
    "tl", "lira", "kurus",
    # Common words
    "fatura", "siparis", "toplam", "tarih", "numara", "adet", "birim",
    "fiyat", "tutar", "miktar", "adres", "telefon", "firma", "musteri",
    "urun", "hizmet", "kdv", "iskonto", "vade", "odeme", "banka",
}


def post_process_text(text, min_confidence=0.15):
    """
    Apply post-processing corrections to OCR output.
    - Fix common OCR character substitutions
    - Filter out low-confidence noise
    - Apply fuzzy dictionary matching
    """
    if not text:
        return text

    # Common OCR character substitutions
    char_fixes = {
        "0": "O",  # context-dependent, applied selectively
        "|": "l",
        "!": "l",
        "{}": "",
        "[]": "",
    }

    lines = text.split("\n")
    corrected_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Apply word-level fuzzy matching
        words = line.split()
        corrected_words = []

        for word in words:
            corrected = fuzzy_match_word(word.lower(), TURKISH_DICTIONARY)
            if corrected:
                # Preserve original casing style
                if word[0].isupper():
                    corrected = corrected.capitalize()
                elif word.isupper():
                    corrected = corrected.upper()
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)

        corrected_lines.append(" ".join(corrected_words))

    return "\n".join(corrected_lines)


def fuzzy_match_word(word, dictionary, max_distance=2):
    """
    Find the closest match in dictionary using edit distance.
    Only corrects if distance is small enough (likely OCR error).
    """
    if not word or len(word) < 3:
        return None

    clean = word.strip(".,;:!?()[]{}\"'")
    if clean.lower() in dictionary:
        return None  # Already correct

    best_match = None
    best_dist = max_distance + 1

    for dict_word in dictionary:
        # Quick length filter
        if abs(len(clean) - len(dict_word)) > max_distance:
            continue

        dist = levenshtein_distance(clean.lower(), dict_word)
        if dist <= max_distance and dist < best_dist:
            best_dist = dist
            best_match = dict_word

    return best_match


def levenshtein_distance(s1, s2):
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def filter_noise_detections(detections, image_width, image_height):
    """
    Filter out detections that are likely noise:
    - Very low confidence
    - Very small text
    - Single character garbage
    """
    filtered = []

    for det in detections:
        text = det.get("text", "").strip()
        conf = det.get("confidence", 0)

        # Skip empty or single-char detections with low confidence
        if not text:
            continue
        if len(text) <= 1 and conf < 0.5:
            continue
        # Skip very low confidence
        if conf < 0.15:
            continue
        # Skip if text is only special characters
        if all(c in "!@#$%^&*()[]{}|\\/<>~`+=_-" for c in text):
            continue

        filtered.append(det)

    return filtered


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def recognize_tesseract(image_bytes):
    """Full pipeline: detect doc -> preprocess -> Tesseract OCR -> post-process.
    Uses Tesseract (pure C++ engine) — works on any CPU, no PyTorch needed."""
    import numpy as np
    import pytesseract
    from PIL import Image

    # Validate image size (reject files > 20MB to prevent OOM)
    if len(image_bytes) > 20 * 1024 * 1024:
        return {
            "success": False,
            "error": f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 20MB.",
        }

    image = Image.open(io.BytesIO(image_bytes))

    # Limit image dimensions to prevent memory issues
    max_pixels = 4096
    if max(image.size) > max_pixels:
        ratio = max_pixels / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        print(f"[PythonOCR] Resizing image from {image.size} to {new_size}", file=sys.stderr, flush=True)
        image = image.resize(new_size, resample=3)

    # Step 1: Detect and crop document region
    doc_image, detected = detect_document_region(image)

    # Step 2: Preprocess
    doc_image = preprocess_image(doc_image)

    # Step 3: Fix orientation
    doc_image = ensure_correct_orientation(doc_image)

    # Step 4: Run Tesseract OCR (Turkish + English)
    # Use --oem 3 (default LSTM engine) and --psm 6 (assume uniform block of text)
    custom_config = r"--oem 3 --psm 6"
    lang = "tur+eng"

    # Get detailed data with bounding boxes and confidence
    try:
        data = pytesseract.image_to_data(
            doc_image, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        # Fallback: try English only
        print(f"[PythonOCR] Tesseract tur+eng failed: {e}, trying eng only", file=sys.stderr, flush=True)
        data = pytesseract.image_to_data(
            doc_image, lang="eng", config=custom_config, output_type=pytesseract.Output.DICT
        )

    # Parse Tesseract output into lines
    n = len(data["text"])
    lines_dict = {}  # line_num -> list of (text, conf)
    detections = []

    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if not text or conf < 0:
            continue

        line_num = data["line_num"][i]
        block_num = data["block_num"][i]
        key = (block_num, line_num)

        if key not in lines_dict:
            lines_dict[key] = []

        lines_dict[key].append({
            "text": text,
            "confidence": conf / 100.0,
            "bbox": [[data["left"][i], data["top"][i]],
                     [data["left"][i] + data["width"][i], data["top"][i]],
                     [data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]],
                     [data["left"][i], data["top"][i] + data["height"][i]]],
        })

        detections.append({
            "text": text,
            "confidence": round(conf / 100.0, 4),
            "bbox": [[data["left"][i], data["top"][i]],
                     [data["left"][i] + data["width"][i], data["top"][i]],
                     [data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]],
                     [data["left"][i], data["top"][i] + data["height"][i]]],
        })

    # Filter noise detections
    img_array = np.array(doc_image)
    detections = filter_noise_detections(
        detections, img_array.shape[1], img_array.shape[0]
    )

    # Build text from lines
    line_texts = []
    all_confidences = []

    for key in sorted(lines_dict.keys()):
        words = lines_dict[key]
        line_text = " ".join(w["text"] for w in words)
        line_conf = sum(w["confidence"] for w in words) / len(words)
        if line_text.strip():
            line_texts.append(line_text)
            all_confidences.append(line_conf)

    full_text = "\n".join(line_texts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    # Step 5: Post-process (fuzzy spelling correction)
    corrected_text = post_process_text(full_text)

    return {
        "success": True,
        "text": corrected_text,
        "raw_text": full_text,
        "confidence": round(avg_confidence, 4),
        "lines": len(line_texts),
        "detections": detections,
        "document_detected": detected,
        "engine": "tesseract",
    }


def recognize_image(image_bytes, engine="tesseract"):
    """Run OCR using Tesseract. No subprocess needed — Tesseract is a safe C++ engine."""
    import traceback
    try:
        print(f"[PythonOCR] recognize_image called: engine=tesseract, size={len(image_bytes)} bytes",
              file=sys.stderr, flush=True)
        t0 = time.time()
        result = recognize_tesseract(image_bytes)
        elapsed = time.time() - t0
        print(f"[PythonOCR] OCR completed in {elapsed:.1f}s, success={result.get('success')}",
              file=sys.stderr, flush=True)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[PythonOCR] ERROR in recognize_image: {e}\n{tb}", file=sys.stderr, flush=True)
        return {"success": False, "error": str(e)}


def warmup_ocr():
    """Run a small dummy image through Tesseract to verify it works."""
    from PIL import Image, ImageDraw
    import pytesseract

    print("[PythonOCR] Running Tesseract warm-up...", file=sys.stderr, flush=True)
    try:
        img = Image.new("RGB", (200, 50), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Test 123", fill=(0, 0, 0))
        text = pytesseract.image_to_string(img, lang="eng").strip()
        print(f"[PythonOCR] Warm-up result: '{text}' — Tesseract is working!", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PythonOCR] WARNING: Tesseract warm-up failed: {e}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(port=5555):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    import urllib.parse
    import traceback
    import threading

    print("[PythonOCR] Verifying Tesseract installation...", file=sys.stderr, flush=True)
    verify_tesseract()
    print("[PythonOCR] Tesseract verified!", file=sys.stderr, flush=True)

    # Pre-warm: run a dummy inference to verify everything works
    warmup_ocr()

    class OCRHandler(BaseHTTPRequestHandler):
        # Increase socket timeout for large requests
        timeout = 300  # 5 minutes

        def log_message(self, fmt, *args):
            print(f"[PythonOCR] {fmt % args}", file=sys.stderr, flush=True)

        def do_POST(self):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                self.log_message("POST %s  Content-Length=%d", self.path, content_length)

                if content_length == 0:
                    self._send_json({"success": False, "error": "Empty request body"})
                    return

                body = self.rfile.read(content_length)
                self.log_message("Body read OK (%d bytes), starting OCR...", len(body))

                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                engine = params.get("engine", ["tesseract"])[0]

                if parsed.path == "/ocr":
                    result = recognize_image(body, engine=engine)
                elif parsed.path == "/health":
                    result = {"status": "healthy", "engine": "tesseract"}
                else:
                    result = {"success": False, "error": "Unknown endpoint"}

                self.log_message("OCR done. success=%s", result.get("success", "?"))
                self._send_json(result)

            except Exception as e:
                tb = traceback.format_exc()
                self.log_message("ERROR in do_POST: %s\n%s", str(e), tb)
                try:
                    self._send_json({"success": False, "error": str(e)})
                except Exception:
                    pass

        def _send_json(self, data):
            response = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response)
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"status": "healthy", "engine": "tesseract"})
            else:
                self._send_json({"error": "Not found"})

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadedHTTPServer(("0.0.0.0", port), OCRHandler)
    print(f"OCR inference server running on http://0.0.0.0:{port} (threaded)", file=sys.stderr, flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="OCR Inference")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--engine", default="tesseract", choices=["tesseract"])
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    if args.server:
        run_server(args.port)
        return

    try:
        if args.image:
            with open(args.image, "rb") as f:
                result = recognize_image(f.read(), engine=args.engine)
        else:
            result = {"success": False, "error": "No image provided"}
    except Exception as e:
        result = {"success": False, "error": str(e)}

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
