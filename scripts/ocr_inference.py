#!/usr/bin/env python3
"""
OCR Inference Script - Dual Engine (EasyOCR + TrOCR)
With document detection, perspective correction, and post-processing.

Pipeline:
  1. Document/paper region detection (removes background noise)
  2. Perspective correction (fixes tilted papers)
  3. Image enhancement (contrast, sharpness, denoise)
  4. OCR (EasyOCR or TrOCR)
  5. Post-processing (spelling correction, confidence filtering)
"""

import sys
import os
import json
import argparse
import io
import math
import time

# Disable NNPACK — causes crashes on unsupported hardware (Docker/VM CPUs)
os.environ["NNPACK_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"  # Limit OpenMP threads to avoid memory spikes
os.environ["MKL_NUM_THREADS"] = "2"
import warnings
warnings.filterwarnings("ignore")

# Global model caches
_easyocr_reader = None
_trocr_processor = None
_trocr_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_easyocr(langs=None):
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    import easyocr

    # Disable NNPACK in PyTorch to prevent crashes on unsupported CPUs
    try:
        import torch
        if hasattr(torch.backends, "nnpack"):
            torch.backends.nnpack.enabled = False
            print("[PythonOCR] NNPACK disabled", file=sys.stderr, flush=True)
    except Exception:
        pass

    if langs is None:
        langs = ["tr", "en"]

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    os.makedirs(cache_dir, exist_ok=True)

    _easyocr_reader = easyocr.Reader(
        langs, gpu=False, model_storage_directory=cache_dir,
    )
    return _easyocr_reader


def load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is not None and _trocr_model is not None:
        return _trocr_processor, _trocr_model

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    model_name = "microsoft/trocr-base-handwritten"
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    _trocr_processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
    _trocr_model.eval()
    return _trocr_processor, _trocr_model


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

def recognize_easyocr(image_bytes):
    """Full pipeline: detect doc -> preprocess -> OCR -> post-process"""
    import numpy as np
    from PIL import Image

    # Validate image size (reject files > 20MB to prevent OOM)
    if len(image_bytes) > 20 * 1024 * 1024:
        return {
            "success": False,
            "error": f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 20MB.",
        }

    image = Image.open(io.BytesIO(image_bytes))

    # Limit image dimensions to prevent memory issues on CPU
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

    # Convert to numpy array for EasyOCR
    img_array = np.array(doc_image)

    reader = load_easyocr()

    # Step 4: Run OCR with optimized parameters
    results = reader.readtext(
        img_array,
        detail=1,
        paragraph=False,
        contrast_ths=0.3,      # lower = detect more low-contrast text
        adjust_contrast=0.7,   # auto adjust contrast
        text_threshold=0.6,    # confidence to consider as text
        low_text=0.3,          # text low-bound score
        width_ths=0.7,         # max horizontal distance to merge boxes
    )

    if not results:
        return {
            "success": True,
            "text": "",
            "confidence": 0.0,
            "lines": 0,
            "detections": [],
            "document_detected": detected,
        }

    # Sort by vertical then horizontal position
    results.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))

    # Build detections list
    detections = []
    for bbox, text, conf in results:
        detections.append({
            "text": text.strip(),
            "confidence": round(float(conf), 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    # Step 5: Filter noise
    detections = filter_noise_detections(
        detections, img_array.shape[1], img_array.shape[0]
    )

    # Group into lines
    lines = group_into_lines(detections)

    # Build text
    line_texts = []
    all_confidences = []

    for line in lines:
        line_text = " ".join(d["text"] for d in line)
        line_conf = sum(d["confidence"] for d in line) / len(line)
        line_texts.append(line_text)
        all_confidences.append(line_conf)

    full_text = "\n".join(line_texts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    # Step 6: Post-process (fuzzy spelling correction)
    corrected_text = post_process_text(full_text)

    return {
        "success": True,
        "text": corrected_text,
        "raw_text": full_text,
        "confidence": round(avg_confidence, 4),
        "lines": len(line_texts),
        "detections": detections,
        "document_detected": detected,
    }


def group_into_lines(detections):
    """Group detections into lines based on vertical proximity."""
    if not detections:
        return []

    lines = []
    current_line = [detections[0]]
    last_y = get_center_y(detections[0])
    line_threshold = 25

    for det in detections[1:]:
        y = get_center_y(det)

        if abs(y - last_y) < line_threshold:
            current_line.append(det)
        else:
            current_line.sort(key=lambda d: d["bbox"][0][0])
            lines.append(current_line)
            current_line = [det]

        last_y = y

    if current_line:
        current_line.sort(key=lambda d: d["bbox"][0][0])
        lines.append(current_line)

    return lines


def get_center_y(det):
    return sum(p[1] for p in det["bbox"]) / len(det["bbox"])


def recognize_trocr(image_bytes):
    """TrOCR with document detection pipeline."""
    import torch
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))

    # Detect and crop document region
    doc_image, detected = detect_document_region(image)
    doc_image = preprocess_image(doc_image)

    processor, model = load_trocr()

    pixel_values = processor(images=doc_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_length=128,
            num_beams=4,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
    text = post_process_text(text)

    confidence = 0.0
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        confidence = min(1.0, max(0.0, math.exp(outputs.sequences_scores[0].item())))

    return {
        "success": True,
        "text": text,
        "confidence": round(confidence, 4),
        "lines": 1,
        "document_detected": detected,
    }


def recognize_tesseract(image_bytes):
    """Tesseract OCR - Pure C++ engine, no PyTorch/AVX required."""
    import numpy as np
    from PIL import Image
    import pytesseract

    # Validate image size
    if len(image_bytes) > 20 * 1024 * 1024:
        return {
            "success": False,
            "error": f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 20MB.",
        }

    image = Image.open(io.BytesIO(image_bytes))

    # Limit image dimensions
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

    # Tesseract OCR with Turkish + English
    # --oem 3: Use LSTM OCR Engine Mode (most accurate)
    # --psm 6: Assume a single uniform block of text
    # -l tur+eng: Enable Turkish and English language support
    custom_config = r'--oem 3 --psm 6 -l tur+eng'
    
    # Get detailed data with bounding boxes and confidence
    data = pytesseract.image_to_data(doc_image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Build detections
    detections = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        # Tesseract uses -1 for invalid/no confidence value
        if not text or conf < 0:
            continue
            
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        detections.append({
            "text": text,
            "confidence": conf / 100.0,  # Convert to 0-1 range
            "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        })
    
    # Filter noise
    detections = filter_noise_detections(detections, doc_image.width, doc_image.height)
    
    # Group into lines
    lines = group_into_lines(detections)
    
    # Build text
    line_texts = []
    all_confidences = []
    
    for line in lines:
        line_text = " ".join(d["text"] for d in line)
        line_conf = sum(d["confidence"] for d in line) / len(line) if line else 0
        line_texts.append(line_text)
        all_confidences.append(line_conf)
    
    full_text = "\n".join(line_texts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    # Post-process
    corrected_text = post_process_text(full_text)
    
    return {
        "success": True,
        "text": corrected_text,
        "raw_text": full_text,
        "confidence": round(avg_confidence, 4),
        "lines": len(line_texts),
        "detections": detections,
        "document_detected": detected,
    }


def _recognize_image_impl(image_bytes, engine="tesseract"):
    """Internal OCR implementation — runs inside a subprocess for crash isolation."""
    import traceback
    try:
        print(f"[PythonOCR] _recognize_image_impl: engine={engine}, size={len(image_bytes)} bytes",
              file=sys.stderr, flush=True)
        t0 = time.time()
        if engine == "trocr":
            result = recognize_trocr(image_bytes)
        elif engine == "easyocr":
            result = recognize_easyocr(image_bytes)
        else:  # Default to tesseract
            result = recognize_tesseract(image_bytes)
        elapsed = time.time() - t0
        print(f"[PythonOCR] OCR completed in {elapsed:.1f}s, success={result.get('success')}",
              file=sys.stderr, flush=True)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[PythonOCR] ERROR in _recognize_image_impl: {e}\n{tb}", file=sys.stderr, flush=True)
        return {"success": False, "error": str(e)}


def _ocr_subprocess_worker(image_bytes, engine, result_queue):
    """Worker function that runs in a forked child process."""
    try:
        import torch
        torch.set_num_threads(1)  # Safe single-threaded mode after fork
    except Exception:
        pass
    try:
        result = _recognize_image_impl(image_bytes, engine)
        result_queue.put(result)
    except Exception as e:
        result_queue.put({"success": False, "error": str(e)})


def recognize_image(image_bytes, engine="tesseract"):
    """Crash-safe OCR wrapper: runs inference in a forked subprocess.

    If the OCR code triggers a SIGILL/SIGSEGV (e.g. CPU doesn't support
    required instructions), only the child process dies. The HTTP server
    parent stays alive and returns a proper error response.
    """
    import multiprocessing

    print(f"[PythonOCR] recognize_image called: engine={engine}, size={len(image_bytes)} bytes",
          file=sys.stderr, flush=True)

    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_ocr_subprocess_worker,
        args=(image_bytes, engine, result_queue),
    )
    proc.start()
    proc.join(timeout=180)  # 3 minute timeout

    if proc.is_alive():
        proc.kill()
        proc.join()
        print("[PythonOCR] OCR subprocess timed out (180s), killed.", file=sys.stderr, flush=True)
        return {"success": False, "error": "OCR timed out after 180 seconds"}

    if proc.exitcode != 0:
        sig = -proc.exitcode if proc.exitcode < 0 else proc.exitcode
        print(f"[PythonOCR] OCR subprocess crashed with exit code {proc.exitcode} (signal {sig})",
              file=sys.stderr, flush=True)
        return {
            "success": False,
            "error": f"OCR process crashed (signal {sig}). "
                     f"Server CPU may not support required instructions.",
        }

    try:
        result = result_queue.get_nowait()
        return result
    except Exception:
        return {"success": False, "error": "OCR subprocess produced no result"}


def warmup_ocr(engine="tesseract"):
    """Run a realistic-sized dummy image through the OCR pipeline.
    For PyTorch-based engines (EasyOCR), uses a 640x480 image to trigger ALL code paths
    including CRAFT detector convolutions that use NNPACK on larger images.
    For Tesseract, warmup is skipped as it doesn't use PyTorch/NNPACK.
    If NNPACK crashes, it happens here at startup instead of on first request."""
    import numpy as np
    from PIL import Image, ImageDraw

    if engine == "tesseract":
        print("[PythonOCR] Tesseract engine - no warmup required", file=sys.stderr, flush=True)
        return

    print("[PythonOCR] Running warm-up inference (640x480)...", file=sys.stderr, flush=True)
    try:
        # Create a realistically-sized image with text-like patterns
        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw several lines of dark rectangles (simulates text lines)
        for row in range(5):
            y = 50 + row * 60
            for col in range(8):
                x = 30 + col * 70
                draw.rectangle([x, y, x + 50, y + 20], fill=(0, 0, 0))

        img_array = np.array(img)
        reader = load_easyocr()
        _ = reader.readtext(img_array, detail=0)
        print("[PythonOCR] Warm-up inference completed successfully!", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PythonOCR] WARNING: Warm-up inference failed: {e}", file=sys.stderr, flush=True)
        print("[PythonOCR] OCR requests may crash. Check CPU compatibility.", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(port=5555):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    import urllib.parse
    import traceback
    import threading

    print("[PythonOCR] Starting OCR server with Tesseract engine (default)...", file=sys.stderr, flush=True)

    # Pre-warm: run a dummy inference to fully initialize PyTorch/NNPACK/OpenCV
    # This catches segfaults at startup instead of on the first real request
    # For Tesseract, this is skipped
    warmup_ocr(engine="tesseract")

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
                    result = {"status": "healthy", "engine": "tesseract (default), easyocr, trocr"}
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
                self._send_json({"status": "healthy", "engine": "tesseract (default), easyocr, trocr"})
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
    parser.add_argument("--engine", default="tesseract", choices=["tesseract", "easyocr", "trocr"])
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
