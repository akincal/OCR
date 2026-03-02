#!/usr/bin/env python3
"""
OCR Inference Script - Multi-Engine (Tesseract + EasyOCR + TrOCR + PaddleOCR + Ensemble)
With document detection, perspective correction, and advanced preprocessing.

Pipeline:
  1. Document/paper region detection (removes background noise)
  2. Perspective correction (fixes tilted papers)
  3. Image quality analysis (adaptive parameter selection)
  4. Advanced preprocessing (deskew, binarization, CLAHE, denoising)
  5. Orientation detection (Tesseract OSD + heuristic)
  6. OCR (Tesseract / EasyOCR / TrOCR / PaddleOCR / Ensemble)
  7. Post-processing (structural regex fixes only)
"""

import sys
import os
import json
import argparse
import io
import math
import time
import re

os.environ["NNPACK_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import warnings
warnings.filterwarnings("ignore")

# Global model caches
_easyocr_reader = None
_trocr_processor = None
_trocr_model = None
_paddleocr_engine = None
# Paddle detection-only engine for TrOCR pipeline (no cls/rec models loaded)
_paddle_det_only_engine = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_easyocr(langs=None):
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    import easyocr

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

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )

    # Keep TrOCR behavior aligned with the minimal HF flow:
    # load the model directly from Hugging Face unless explicitly overridden.
    model_name = os.getenv("TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten")
    print(f"[PythonOCR] Loading TrOCR model: {model_name}", file=sys.stderr, flush=True)

    _trocr_processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    # use_safetensors=True avoids torch.load vulnerability (CVE-2025-32434)
    # which blocks loading with torch < 2.6 even when weights_only=True.
    _trocr_model = VisionEncoderDecoderModel.from_pretrained(
        model_name, cache_dir=cache_dir, use_safetensors=True
    )
    _trocr_model.eval()
    return _trocr_processor, _trocr_model


def load_paddleocr():
    global _paddleocr_engine
    if _paddleocr_engine is not None:
        return _paddleocr_engine

    from paddleocr import PaddleOCR

    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )

    # Fine-tuned Turkish recognition model takes priority when present.
    # Place the exported model in models/tr_PP-OCRv4_rec_infer/ and the
    # character dictionary in models/tr_ppocr_dict.txt after Colab training.
    custom_rec_dir  = os.path.join(models_dir, "tr_PP-OCRv4_rec_infer")
    custom_dict     = os.path.join(models_dir, "tr_ppocr_dict.txt")
    use_custom_model = (
        os.path.isdir(custom_rec_dir)
        and os.path.isfile(custom_dict)
    )

    kwargs = dict(
        use_angle_cls=True,
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.2,
        det_db_box_thresh=0.4,
        det_db_unclip_ratio=2.0,
        rec_batch_num=6,
    )

    if use_custom_model:
        # Fine-tuned Turkish model: supply rec model + dict directly.
        # Detection still uses the default PP-OCRv4 en detector.
        print("[PythonOCR] Loading fine-tuned Turkish PP-OCRv4 rec model", file=sys.stderr, flush=True)
        kwargs["rec_model_dir"]       = custom_rec_dir
        kwargs["rec_char_dict_path"]  = custom_dict
        kwargs["lang"]                = "en"           # keeps en detector (PP-OCRv4)
        kwargs["ocr_version"]         = "PP-OCRv4"
    else:
        # Fallback: standard PP-OCRv4 English model.
        # "en" alphabet covers Turkish (same Latin base) better than "tr" (PP-OCRv3).
        kwargs["lang"]       = "en"
        kwargs["ocr_version"] = "PP-OCRv4"

    _paddleocr_engine = PaddleOCR(**kwargs)
    return _paddleocr_engine


def load_paddle_det_only():
    """
    Load PaddleOCR with detection only (no angle cls, no recognition).
    Used by TrOCR pipeline for text region detection; text extraction is done by TrOCR.
    Avoids loading ch_ppocr_mobile_v2.0_cls_infer and rec models.
    """
    global _paddle_det_only_engine
    if _paddle_det_only_engine is not None:
        return _paddle_det_only_engine

    from paddleocr import PaddleOCR

    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )

    # Detection-only: same det params as full engine, no cls (no missing .tar), no rec needed.
    kwargs = dict(
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.2,
        det_db_box_thresh=0.4,
        det_db_unclip_ratio=2.0,
        lang="en",
        ocr_version="PP-OCRv4",
    )

    print("[PythonOCR] Loading PaddleOCR (detection only) for TrOCR pipeline", file=sys.stderr, flush=True)
    _paddle_det_only_engine = PaddleOCR(**kwargs)
    return _paddle_det_only_engine


# ---------------------------------------------------------------------------
# Image quality analysis
# ---------------------------------------------------------------------------

def analyze_image_quality(image):
    """Analyze image quality metrics to drive adaptive preprocessing."""
    import numpy as np
    import cv2

    gray = np.array(image.convert("L"))

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast_score = float(gray.std())

    dy = np.abs(np.diff(gray.astype(np.float32), axis=0))
    noise_score = float(np.mean(dy))

    return {
        "blur": blur_score,
        "contrast": contrast_score,
        "noise": noise_score,
    }


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

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, False

    best_contour = None
    best_area = 0
    min_area = h * w * 0.05

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_contour = approx
            best_area = area

    if best_contour is None:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < min_area:
            return image, False

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        best_contour = np.int32(box).reshape(4, 1, 2)

    pts = best_contour.reshape(4, 2).astype(np.float32)
    ordered = order_points(pts)
    corrected = four_point_transform(orig, ordered)

    from PIL import Image as PILImage
    return PILImage.fromarray(corrected), True


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    import numpy as np

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(image, pts):
    """Apply perspective transform to get a top-down view of the document."""
    import numpy as np
    import cv2

    tl, tr, br, bl = pts

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
# 2) Advanced image preprocessing / enhancement
# ---------------------------------------------------------------------------

def deskew_image(image):
    """Correct skew angle using minAreaRect on text pixels."""
    import numpy as np
    import cv2
    import sys

    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # If a near-full-width horizontal stroke is present (e.g. underlines),
    # minAreaRect can over-estimate skew on otherwise straight images.
    h, w = gray.shape[:2]
    row_ink = np.sum(binary > 0, axis=1)
    strong_rows = row_ink > (0.6 * w)
    if np.any(strong_rows):
        strong_count = int(np.count_nonzero(strong_rows))
        if strong_count <= max(3, int(0.01 * h)):
            print("[PythonOCR] Deskew skipped: horizontal-line dominant rows detected",
                  file=sys.stderr, flush=True)
            return image

    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Conservative deskew for mostly-straight mobile captures.
    if abs(angle) < 0.8 or abs(angle) > 5:
        print(f"[PythonOCR] Deskew skipped: estimated angle={angle:.2f}°", file=sys.stderr, flush=True)
        return image

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_array, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    from PIL import Image as PILImage
    print(f"[PythonOCR] Deskew applied: angle={angle:.2f}°", file=sys.stderr, flush=True)
    return PILImage.fromarray(rotated)


def adaptive_binarize(image):
    """Apply adaptive thresholding for cleaner text extraction."""
    import numpy as np
    import cv2

    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    from PIL import Image as PILImage
    return PILImage.fromarray(binary)


def preprocess_image(image, quality=None):
    """Advanced preprocessing pipeline with adaptive parameters."""
    import numpy as np
    import cv2
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = ImageOps.exif_transpose(image)

    # Upscale small images for better OCR accuracy
    min_dim = 1200
    if max(image.size) < min_dim:
        ratio = min_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, resample=Image.LANCZOS)

    max_dim = 2048
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, resample=3)

    if quality is None:
        quality = analyze_image_quality(image)

    # Adaptive denoising based on noise level
    if quality["noise"] > 30:
        img_array = np.array(image)
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        from PIL import Image as PILImage
        image = PILImage.fromarray(denoised)
    else:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    # CLAHE for low-contrast images
    if quality["contrast"] < 40:
        img_array = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)
        from PIL import Image as PILImage
        image = PILImage.fromarray(enhanced).convert("RGB")
    else:
        image = ImageOps.autocontrast(image, cutoff=1)

    # Keep enhancement mild to avoid over-processing clean handwritten inputs.
    contrast_factor = 1.3 if quality["contrast"] < 40 else 1.15
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    # Adaptive sharpness
    sharpness_factor = 1.4 if quality["blur"] < 100 else 1.15
    image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

    return image


def ensure_correct_orientation(image):
    """
    Multi-strategy orientation detection:
    1. Try Tesseract OSD for reliable angle detection (0/90/180/270)
    2. Fall back to text density heuristic
    """
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(image, config="--psm 0 -l tur+eng")
        for line in osd.split('\n'):
            if 'Rotate' in line:
                angle = int(line.split(':')[1].strip())
                if angle != 0:
                    print(f"[PythonOCR] OSD detected rotation: {angle}°", file=sys.stderr, flush=True)
                    return image.rotate(angle, expand=True)
        return image
    except Exception:
        pass

    import numpy as np
    img_array = np.array(image.convert("L"))
    h = img_array.shape[0]
    binary = (img_array < 128).astype(np.uint8)

    top_density = np.sum(binary[:h // 3])
    bottom_density = np.sum(binary[2 * h // 3:])

    if bottom_density > top_density * 3:
        return image.rotate(180, expand=True)

    return image


# ---------------------------------------------------------------------------
# 3) Post-processing
# ---------------------------------------------------------------------------


def structural_corrections(text):
    """Fix structural OCR patterns common in Turkish documents."""
    text = re.sub(r'(\d)[lI]\.', r'\g<1>1.', text)
    text = re.sub(r'(?i)\bkdv\b', 'KDV', text)
    text = re.sub(r'(?i)\b[tT][lL]\b', 'TL', text)
    text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)
    text = re.sub(r'O(\d{2,})', r'0\g<1>', text)
    text = re.sub(r'(\d)[oO](\d)', r'\g<1>0\g<2>', text)
    text = re.sub(r'[|]', 'l', text)
    text = re.sub(r'\{|\}', '', text)
    text = re.sub(r'\[|\]', '', text)
    text = re.sub(r'(?i)\bT[.,]C[.,]\b', 'T.C.', text)
    text = re.sub(r'(?i)\bno[.:]\s*', 'No: ', text)
    return text



def post_process_text(text):
    """Post-processing pipeline: structural regex corrections only."""
    if not text:
        return text

    return structural_corrections(text)


# Engine-specific confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "tesseract": {"min_conf": 0.30, "single_char_conf": 0.60},
    "easyocr":   {"min_conf": 0.20, "single_char_conf": 0.50},
    "trocr":     {"min_conf": 0.10, "single_char_conf": 0.40},
    "paddleocr": {"min_conf": 0.25, "single_char_conf": 0.55},
}


def filter_noise_detections(detections, image_width, image_height, engine="tesseract"):
    """
    Filter out noise detections with engine-calibrated thresholds.
    """
    t = CONFIDENCE_THRESHOLDS.get(engine, CONFIDENCE_THRESHOLDS["tesseract"])
    filtered = []

    for det in detections:
        text = det.get("text", "").strip()
        conf = det.get("confidence", 0)

        if not text:
            continue
        if len(text) <= 1 and conf < t["single_char_conf"]:
            continue
        if conf < t["min_conf"]:
            continue
        if all(c in "!@#$%^&*()[]{}|\\/<>~`+=_-" for c in text):
            continue

        filtered.append(det)

    return filtered


# ---------------------------------------------------------------------------
# Adaptive Tesseract PSM selection
# ---------------------------------------------------------------------------

def get_optimal_psm(image):
    """Select the best Tesseract PSM mode based on image content analysis."""
    import numpy as np
    import cv2

    img_array = np.array(image.convert("L"))
    binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 100]

    h_proj = np.sum(binary, axis=1)
    peak_threshold = h_proj.max() * 0.1 if h_proj.max() > 0 else 0
    text_rows = np.sum(h_proj > peak_threshold)

    if len(large_contours) <= 3 and text_rows < img_array.shape[0] * 0.2:
        return 7  # Single text line
    elif text_rows > img_array.shape[0] * 0.6:
        return 3  # Fully automatic segmentation
    else:
        return 6  # Uniform text block (default)


# ---------------------------------------------------------------------------
# Line segmentation for TrOCR
# ---------------------------------------------------------------------------

def segment_text_lines(image):
    """Segment image into individual text lines using horizontal projection."""
    import numpy as np
    import cv2

    img_array = np.array(image.convert("L"))
    binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    h_proj = np.sum(binary, axis=1)
    # Use a slightly stricter projection threshold to reduce noisy micro-lines.
    threshold = h_proj.max() * 0.08 if h_proj.max() > 0 else 0

    in_line = False
    lines = []
    start = 0

    for i, val in enumerate(h_proj):
        if not in_line and val > threshold:
            in_line = True
            start = i
        elif in_line and val <= threshold:
            in_line = False
            # Add a bit more margin to avoid cutting ascenders/descenders.
            padding = 10
            lines.append((max(0, start - padding), min(len(h_proj), i + padding)))

    if in_line:
        lines.append((max(0, start - 10), len(h_proj)))

    # Filter out tiny fragments that usually come from noise after preprocessing.
    min_height = 12
    lines = [(y1, y2) for y1, y2 in lines if (y2 - y1) >= min_height]

    # Remove non-text horizontal strokes (e.g. underlines) and tiny noisy bands.
    filtered = []
    img_w = img_array.shape[1]
    for y1, y2 in lines:
        band = binary[y1:y2, :]
        band_h = max(1, y2 - y1)
        ink_pixels = int(np.count_nonzero(band))
        ink_ratio = ink_pixels / float(band.size)
        row_ink = np.sum(band > 0, axis=1)
        active_rows = int(np.count_nonzero(row_ink > (0.05 * img_w)))
        max_row_ink = int(row_ink.max()) if row_ink.size else 0

        # Likely an underline: very long stroke concentrated in very few rows.
        if max_row_ink > 0.70 * img_w and active_rows <= max(3, int(0.20 * band_h)):
            continue
        # Very light short bands are usually noise.
        if band_h < 20 and ink_ratio < 0.01:
            continue

        filtered.append((y1, y2))

    lines = filtered

    # Merge adjacent fragments when a tiny gap splits the same text line.
    if lines:
        merged = [lines[0]]
        for y1, y2 in lines[1:]:
            py1, py2 = merged[-1]
            gap = y1 - py2
            prev_h = py2 - py1
            curr_h = y2 - y1
            if gap <= 2 or (gap <= 6 and (prev_h < 22 or curr_h < 22)):
                merged[-1] = (py1, y2)
            else:
                merged.append((y1, y2))
        lines = merged

    if not lines:
        lines = [(0, img_array.shape[0])]

    return lines


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def recognize_easyocr(image_bytes):
    """Full pipeline: detect doc -> preprocess -> OCR -> post-process"""
    import numpy as np
    from PIL import Image

    if len(image_bytes) > 20 * 1024 * 1024:
        return {
            "success": False,
            "error": f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 20MB.",
        }

    image = Image.open(io.BytesIO(image_bytes))

    max_pixels = 4096
    if max(image.size) > max_pixels:
        ratio = max_pixels / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        print(f"[PythonOCR] Resizing image from {image.size} to {new_size}", file=sys.stderr, flush=True)
        image = image.resize(new_size, resample=3)

    doc_image, detected = detect_document_region(image)
    quality = analyze_image_quality(doc_image)
    doc_image = deskew_image(doc_image)
    doc_image = preprocess_image(doc_image, quality=quality)
    doc_image = ensure_correct_orientation(doc_image)

    img_array = np.array(doc_image)

    reader = load_easyocr()

    results = reader.readtext(
        img_array,
        detail=1,
        paragraph=False,
        contrast_ths=0.3,
        adjust_contrast=0.7,
        text_threshold=0.6,
        low_text=0.3,
        width_ths=0.7,
    )

    if not results:
        return {
            "success": True,
            "text": "",
            "confidence": 0.0,
            "lines": 0,
            "detections": [],
            "document_detected": detected,
            "engine": "easyocr",
        }

    results.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))

    detections = []
    for bbox, text, conf in results:
        detections.append({
            "text": text.strip(),
            "confidence": round(float(conf), 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    detections = filter_noise_detections(
        detections, img_array.shape[1], img_array.shape[0], engine="easyocr"
    )

    lines = group_into_lines(detections)

    line_texts = []
    all_confidences = []

    for line in lines:
        line_text = " ".join(d["text"] for d in line)
        line_conf = sum(d["confidence"] for d in line) / len(line)
        line_texts.append(line_text)
        all_confidences.append(line_conf)

    full_text = "\n".join(line_texts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    corrected_text = post_process_text(full_text)

    return {
        "success": True,
        "text": corrected_text,
        "raw_text": full_text,
        "confidence": round(avg_confidence, 4),
        "lines": len(line_texts),
        "detections": detections,
        "document_detected": detected,
        "engine": "easyocr",
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


def detect_line_regions_with_paddle(image):
    """
    Detect text boxes with PaddleOCR detector only (no cls/rec).
    Returns line regions as (x1, y1, x2, y2), top-to-bottom.
    Used by TrOCR pipeline; recognition is done by TrOCR.
    """
    import numpy as np

    engine = load_paddle_det_only()
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    boxes = []

    try:
        # Use low-level detector directly because some PaddleOCR versions
        # can fail on `ocr(..., rec=False)` with numpy truth-value errors.
        dt_boxes, _ = engine.text_detector(img_array)
        if dt_boxes is None:
            dt_boxes = []

        for polygon in dt_boxes:
            xs = [int(p[0]) for p in polygon]
            ys = [int(p[1]) for p in polygon]
            x1 = max(0, min(xs))
            x2 = min(img_w - 1, max(xs))
            y1 = max(0, min(ys))
            y2 = min(img_h - 1, max(ys))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"[PythonOCR] Paddle det failed: {e}", file=sys.stderr, flush=True)
        boxes = []

    if not boxes:
        # Fallback: horizontal projection line segmentation so TrOCR still gets per-line crops
        try:
            line_bands = segment_text_lines(image)
            regions = [(0, y1, img_w - 1, y2) for y1, y2 in line_bands]
            if regions:
                print("[PythonOCR] Using projection-based line regions (Paddle det had no boxes)", file=sys.stderr, flush=True)
                return regions
        except Exception:
            pass
        return [(0, 0, img_w - 1, img_h - 1)]

    boxes.sort(key=lambda b: ((b[1] + b[3]) / 2.0, b[0]))
    heights = [max(1, b[3] - b[1]) for b in boxes]
    median_h = float(np.median(heights)) if heights else 20.0
    line_threshold = max(8.0, median_h * 0.6)

    lines = []
    current = [boxes[0]]
    current_cy = (boxes[0][1] + boxes[0][3]) / 2.0

    for b in boxes[1:]:
        cy = (b[1] + b[3]) / 2.0
        if abs(cy - current_cy) <= line_threshold:
            current.append(b)
            current_cy = sum((x[1] + x[3]) / 2.0 for x in current) / len(current)
        else:
            lines.append(current)
            current = [b]
            current_cy = cy
    if current:
        lines.append(current)

    regions = []
    for line_boxes in lines:
        x1 = min(b[0] for b in line_boxes)
        y1 = min(b[1] for b in line_boxes)
        x2 = max(b[2] for b in line_boxes)
        y2 = max(b[3] for b in line_boxes)

        pad_x = 8
        pad_y = max(8, int(0.2 * (y2 - y1 + 1)))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(img_w - 1, x2 + pad_x)
        ry2 = min(img_h - 1, y2 + pad_y)

        region_h = ry2 - ry1
        region_w = rx2 - rx1
        if region_h < 10:
            continue
        if region_w > 0.75 * img_w and region_h < max(18, int(0.03 * img_h)):
            continue
        regions.append((rx1, ry1, rx2, ry2))

    if not regions:
        return [(0, 0, img_w - 1, img_h - 1)]
    return regions


def _prepare_crop_for_trocr(crop):
    """
    Prepare a line crop for TrOCR:
    1. Upscale if height < 100px so thin strokes are readable.
    2. Pad to square with white background so the processor doesn't
       stretch the aspect ratio when resizing to 384x384.
    """
    from PIL import Image

    w, h = crop.size

    # Step 1: upscale short crops
    min_h = 100
    if h < min_h:
        scale = min_h / h
        new_w = max(1, int(w * scale))
        crop = crop.resize((new_w, min_h), Image.LANCZOS)
        w, h = crop.size

    # Step 2: pad to square with white background
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    padded.paste(crop, (pad_x, pad_y))

    return padded


def recognize_trocr(image_bytes):
    """PaddleOCR text detection + TrOCR recognition for multi-line images."""
    import torch
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    processor, model = load_trocr()
    line_regions = detect_line_regions_with_paddle(image)

    all_texts = []
    all_confidences = []
    for x1, y1, x2, y2 in line_regions:
        line_img = _prepare_crop_for_trocr(image.crop((x1, y1, x2, y2)))
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_length=128,
                num_beams=5,
                return_dict_in_generate=True,
                output_scores=True,
                length_penalty=1.0,
            )

        text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        if text:
            all_texts.append(text)
        conf = 0.0
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            conf = min(1.0, max(0.0, math.exp(outputs.sequences_scores[0].item())))
        all_confidences.append(conf)

    full_text = "\n".join(all_texts)
    confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return {
        "success": True,
        "text": full_text,
        "raw_text": full_text,
        "confidence": round(confidence, 4),
        "lines": len(all_texts),
        "document_detected": False,
        "engine": "trocr",
    }


def recognize_tesseract(image_bytes):
    """Tesseract OCR with adaptive PSM and advanced preprocessing."""
    import numpy as np
    from PIL import Image
    import pytesseract

    if len(image_bytes) > 20 * 1024 * 1024:
        return {
            "success": False,
            "error": f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 20MB.",
        }

    image = Image.open(io.BytesIO(image_bytes))

    max_pixels = 4096
    if max(image.size) > max_pixels:
        ratio = max_pixels / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        print(f"[PythonOCR] Resizing image from {image.size} to {new_size}", file=sys.stderr, flush=True)
        image = image.resize(new_size, resample=3)

    doc_image, detected = detect_document_region(image)
    quality = analyze_image_quality(doc_image)
    doc_image = deskew_image(doc_image)
    doc_image = preprocess_image(doc_image, quality=quality)
    doc_image = ensure_correct_orientation(doc_image)

    psm = get_optimal_psm(doc_image)
    custom_config = f'--oem 3 --psm {psm} -l tur+eng'
    print(f"[PythonOCR] Tesseract using PSM {psm}", file=sys.stderr, flush=True)

    # Run on both the enhanced color image and a binarized version, pick best
    data_color = pytesseract.image_to_data(doc_image, config=custom_config, output_type=pytesseract.Output.DICT)

    bin_image = adaptive_binarize(doc_image)
    data_binary = pytesseract.image_to_data(bin_image, config=custom_config, output_type=pytesseract.Output.DICT)

    def _build_detections(data):
        dets = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if not text or conf < 0:
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            dets.append({
                "text": text,
                "confidence": conf / 100.0,
                "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            })
        return dets

    dets_color = _build_detections(data_color)
    dets_binary = _build_detections(data_binary)

    avg_color = sum(d["confidence"] for d in dets_color) / max(len(dets_color), 1)
    avg_binary = sum(d["confidence"] for d in dets_binary) / max(len(dets_binary), 1)

    detections = dets_binary if avg_binary > avg_color else dets_color

    detections = filter_noise_detections(detections, doc_image.width, doc_image.height, engine="tesseract")

    lines = group_into_lines(detections)

    line_texts = []
    all_confidences = []

    for line in lines:
        line_text = " ".join(d["text"] for d in line)
        line_conf = sum(d["confidence"] for d in line) / len(line) if line else 0
        line_texts.append(line_text)
        all_confidences.append(line_conf)

    full_text = "\n".join(line_texts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

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


def recognize_paddleocr(image_bytes):
    """PaddleOCR engine with built-in angle classification and detection."""
    import numpy as np
    from PIL import Image

    if len(image_bytes) > 20 * 1024 * 1024:
        return {
            "success": False,
            "error": f"Image too large ({len(image_bytes) // 1024 // 1024}MB). Max 20MB.",
        }

    image = Image.open(io.BytesIO(image_bytes))

    max_pixels = 4096
    if max(image.size) > max_pixels:
        ratio = max_pixels / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, resample=3)

    doc_image, detected = detect_document_region(image)
    quality = analyze_image_quality(doc_image)
    doc_image = deskew_image(doc_image)
    doc_image = preprocess_image(doc_image, quality=quality)

    img_array = np.array(doc_image)
    if len(img_array.shape) == 2:
        import cv2
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    ocr = load_paddleocr()
    result = ocr.ocr(img_array, cls=True)

    if not result or not result[0]:
        return {
            "success": True,
            "text": "",
            "confidence": 0.0,
            "lines": 0,
            "detections": [],
            "document_detected": detected,
            "engine": "paddleocr",
        }

    detections = []
    for line in result[0]:
        bbox, (text, conf) = line[0], line[1]
        detections.append({
            "text": text.strip(),
            "confidence": round(float(conf), 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    detections = filter_noise_detections(
        detections, img_array.shape[1], img_array.shape[0], engine="paddleocr"
    )

    lines = group_into_lines(detections)

    line_texts = []
    all_confidences = []

    for line in lines:
        line_text = " ".join(d["text"] for d in line)
        line_conf = sum(d["confidence"] for d in line) / len(line)
        line_texts.append(line_text)
        all_confidences.append(line_conf)

    full_text = "\n".join(line_texts)
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    corrected_text = post_process_text(full_text)

    return {
        "success": True,
        "text": corrected_text,
        "raw_text": full_text,
        "confidence": round(avg_confidence, 4),
        "lines": len(line_texts),
        "detections": detections,
        "document_detected": detected,
        "engine": "paddleocr",
    }


# ---------------------------------------------------------------------------
# Ensemble engine
# ---------------------------------------------------------------------------

def recognize_ensemble(image_bytes):
    """
    Run multiple OCR engines and merge results line-by-line,
    picking the highest-confidence output per line.
    """
    print("[PythonOCR] Ensemble: running Tesseract + EasyOCR...", file=sys.stderr, flush=True)

    result_tesseract = recognize_tesseract(image_bytes)
    result_easyocr = recognize_easyocr(image_bytes)

    results = []
    if result_tesseract.get("success"):
        results.append(result_tesseract)
    if result_easyocr.get("success"):
        results.append(result_easyocr)

    try:
        result_paddle = recognize_paddleocr(image_bytes)
        if result_paddle.get("success"):
            results.append(result_paddle)
    except Exception as e:
        print(f"[PythonOCR] Ensemble: PaddleOCR skipped ({e})", file=sys.stderr, flush=True)

    if not results:
        return {"success": False, "error": "All ensemble engines failed"}

    best = max(results, key=lambda r: r.get("confidence", 0))

    if len(results) >= 2:
        merged = _merge_results_line_by_line(results)
        if merged:
            return merged

    best["engine"] = "ensemble"
    return best


def _merge_results_line_by_line(results):
    """Merge multiple engine results by selecting best lines."""
    try:
        all_line_groups = []
        for r in results:
            dets = r.get("detections", [])
            if dets:
                all_line_groups.append((group_into_lines(dets), r.get("engine", "unknown")))

        if not all_line_groups:
            return None

        best_result = max(results, key=lambda r: len(r.get("detections", [])))
        best_lines = group_into_lines(best_result.get("detections", []))

        merged_line_texts = []
        merged_confidences = []

        for i, line in enumerate(best_lines):
            best_text = " ".join(d["text"] for d in line)
            best_conf = sum(d["confidence"] for d in line) / max(len(line), 1)

            for line_groups, engine_name in all_line_groups:
                if i < len(line_groups):
                    other_line = line_groups[i]
                    other_conf = sum(d["confidence"] for d in other_line) / max(len(other_line), 1)
                    if other_conf > best_conf:
                        best_text = " ".join(d["text"] for d in other_line)
                        best_conf = other_conf

            merged_line_texts.append(best_text)
            merged_confidences.append(best_conf)

        full_text = "\n".join(merged_line_texts)
        avg_conf = sum(merged_confidences) / len(merged_confidences) if merged_confidences else 0.0

        corrected_text = post_process_text(full_text)

        return {
            "success": True,
            "text": corrected_text,
            "raw_text": full_text,
            "confidence": round(avg_conf, 4),
            "lines": len(merged_line_texts),
            "document_detected": best_result.get("document_detected", False),
            "engine": "ensemble",
        }
    except Exception as e:
        print(f"[PythonOCR] Ensemble merge failed: {e}", file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------
# Dispatch & crash isolation
# ---------------------------------------------------------------------------

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
        elif engine == "paddleocr":
            result = recognize_paddleocr(image_bytes)
        elif engine == "ensemble":
            result = recognize_ensemble(image_bytes)
        else:
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
        torch.set_num_threads(1)
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

    # TrOCR can be slow on CPU with line-by-line generation.
    if engine == "ensemble":
        timeout = 300
    elif engine == "trocr":
        timeout = 420
    else:
        timeout = 180

    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_ocr_subprocess_worker,
        args=(image_bytes, engine, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        print(f"[PythonOCR] OCR subprocess timed out ({timeout}s), killed.", file=sys.stderr, flush=True)
        return {"success": False, "error": f"OCR timed out after {timeout} seconds"}

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
    """Run a realistic-sized dummy image through the OCR pipeline."""
    import numpy as np
    from PIL import Image, ImageDraw

    if engine == "tesseract":
        print("[PythonOCR] Tesseract engine - no warmup required", file=sys.stderr, flush=True)
        return

    print("[PythonOCR] Running warm-up inference (640x480)...", file=sys.stderr, flush=True)
    try:
        img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
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


def preload_trocr_and_paddle_det():
    """
    Load Paddle (detection-only) and TrOCR at startup so the service is ready
    with fine-tuned weights before accepting requests (no lazy loading for this pipeline).
    Forked OCR workers inherit these loaded models from the server process.
    """
    from PIL import Image

    model_name = os.getenv("TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten")
    print(f"[PythonOCR] Preloading Paddle (det) + TrOCR at startup (TrOCR: {model_name})...",
          file=sys.stderr, flush=True)

    try:
        load_trocr()
        print("[PythonOCR] TrOCR model loaded.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PythonOCR] WARNING: TrOCR preload failed: {e}", file=sys.stderr, flush=True)
        raise

    try:
        load_paddle_det_only()
        print("[PythonOCR] PaddleOCR (detection only) loaded.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PythonOCR] WARNING: Paddle det preload failed: {e}", file=sys.stderr, flush=True)
        raise

    # Sanity check: run one minimal inference so first request does not pay cold cost
    try:
        small = Image.new("RGB", (200, 60), color=(255, 255, 255))
        buf = io.BytesIO()
        small.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        _ = recognize_trocr(image_bytes)
        print("[PythonOCR] Paddle+TrOCR pipeline warm-up inference OK.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PythonOCR] WARNING: Paddle+TrOCR warm-up inference failed: {e}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(port=5555):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    import urllib.parse
    import traceback
    import threading

    print("[PythonOCR] Starting OCR server (engines: tesseract, easyocr, trocr, paddleocr, ensemble)...",
          file=sys.stderr, flush=True)

    warmup_ocr(engine="tesseract")

    # Embed Paddle (det) + TrOCR at startup so service is ready with fine-tuned model
    print("[PythonOCR] Loading Paddle (det) + TrOCR at startup (eager load)...", file=sys.stderr, flush=True)
    try:
        preload_trocr_and_paddle_det()
        print("[PythonOCR] Paddle + TrOCR ready at startup.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PythonOCR] Startup preload failed: {e}. TrOCR requests may fail or be slow on first use.",
              file=sys.stderr, flush=True)

    class OCRHandler(BaseHTTPRequestHandler):

    class OCRHandler(BaseHTTPRequestHandler):
        timeout = 300

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
                    result = {
                        "status": "healthy",
                        "engine": "tesseract (default), easyocr, trocr, paddleocr, ensemble",
                    }
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
                self._send_json({
                    "status": "healthy",
                    "engine": "tesseract (default), easyocr, trocr, paddleocr, ensemble",
                })
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
    parser.add_argument("--engine", default="tesseract",
                        choices=["tesseract", "easyocr", "trocr", "paddleocr", "ensemble"])
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
