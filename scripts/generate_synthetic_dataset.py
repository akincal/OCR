#!/usr/bin/env python3
"""
Synthetic Turkish OCR Dataset Generator

Generates labeled line-image pairs for fine-tuning TrOCR on Turkish text.

Features:
  - Rich Turkish text corpus (documents, invoices, general text, addresses)
  - Multiple fonts, sizes, and styles
  - Realistic augmentation: noise, blur, skew, brightness, stains
  - Direct compatibility with train_trocr_turkish.py

Usage:
  python scripts/generate_synthetic_dataset.py \
      --output_dir ./training_data \
      --count 5000

  # Quick test with 100 samples:
  python scripts/generate_synthetic_dataset.py --output_dir ./training_data --count 100
"""

import os
import sys
import json
import random
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np


# ---------------------------------------------------------------------------
# Turkish text corpus
# ---------------------------------------------------------------------------

CORPUS_GENERAL = [
    # Günlük ifadeler
    "Merhaba, nasılsınız?", "İyi günler dilerim.", "Teşekkür ederim.",
    "Lütfen dikkat ediniz.", "Bu belge önemlidir.", "Tarih ve saat belirtiniz.",
    "İmzalı ve mühürlü suret.", "Aslına uygundur.", "Onaylı fotokopi.",
    "Bilgilerinize saygılarımla sunarım.",
    # Resmi yazışmalar
    "Sayın Yetkili,", "Gereği için arz ederim.", "İlgi yazınız hakkında,",
    "Ekte sunulan belgeler incelenmiştir.", "Tarafımıza ulaşan başvurunuz,",
    "Konu: Bilgi talebi hakkında.", "Tarih: 15 Ocak 2024",
    "Sayı: 2024/001", "Ek: 3 adet belge", "Dağıtım: Gereği",
    # Günlük metinler
    "Türkiye Cumhuriyeti Devleti", "Ankara, Türkiye'nin başkentidir.",
    "İstanbul Boğazı çok güzeldir.", "Ege kıyıları mavi berraklığındadır.",
    "Karadeniz'de fındık hasadı başladı.", "Akdeniz iklimi yumuşaktır.",
    "Toroslar'da kar yağışı bekleniyor.", "Marmara Bölgesi sanayi merkezi.",
]

CORPUS_INVOICE = [
    # Fatura başlıkları
    "FATURA", "İRSALİYE", "FATURA NO:", "TARİH:", "VADE TARİHİ:",
    "MÜŞTERİ ADI:", "MÜŞTERİ KODU:", "SİPARİŞ NO:", "VERGİ NO:",
    "TC KİMLİK NO:", "FATURA ADRESİ:", "TESLİMAT ADRESİ:",
    # Tablo başlıkları
    "ÜRÜN ADI", "MİKTAR", "BİRİM FİYAT", "TUTAR", "KDV ORANI",
    "KDV TUTARI", "TOPLAM TUTAR", "GENEL TOPLAM", "ÖDENECEK TUTAR",
    "İSKONTO", "NET TUTAR", "BRÜT TUTAR",
    # Ürün satırları
    "Ofis Malzemesi 100 adet x 2,50 TL = 250,00 TL",
    "A4 Kağıt (500'lü Paket) 5 kutu 45,00 TL 225,00 TL",
    "Kalem Seti 10 adet 8,00 TL 80,00 TL %18 KDV",
    "Dosya Dolabı 1 adet 850,00 TL 850,00 TL",
    "Yazıcı Kartuşu HP 2 adet 120,00 TL 240,00 TL",
    "Zımba Makinesi 3 adet 35,00 TL 105,00 TL",
    # Toplamlar
    "Ara Toplam: 1.750,00 TL", "KDV (%18): 315,00 TL",
    "Genel Toplam: 2.065,00 TL", "Ödeme Vadesi: 30 gün",
    "Banka: Ziraat Bankası", "IBAN: TR12 0001 0012 3456 7890 12",
    "Ödeme Şekli: Havale/EFT",
]

CORPUS_ADDRESS = [
    "Atatürk Caddesi No: 42 Daire: 3",
    "Cumhuriyet Mahallesi, İnönü Sokak No: 7",
    "Bağcılar, İstanbul 34200",
    "Kadıköy / İSTANBUL",
    "Çankaya / ANKARA 06100",
    "Konak / İZMİR 35250",
    "Osmangazi / BURSA 16000",
    "Muratpaşa / ANTALYA 07100",
    "Seyhan / ADANA 01120",
    "Selçuklu / KONYA 42060",
    "Tel: 0212 555 12 34", "Fax: 0212 555 12 35",
    "GSM: 0532 123 45 67", "E-posta: info@sirket.com.tr",
    "Web: www.sirket.com.tr",
]

CORPUS_PERSONAL = [
    "Ad Soyad: Mehmet Yılmaz", "Ad Soyad: Fatma Kaya",
    "Ad Soyad: Ahmet Demir", "Ad Soyad: Ayşe Çelik",
    "Ad Soyad: Mustafa Şahin", "Ad Soyad: Zeynep Güneş",
    "TC: 12345678901", "Doğum Tarihi: 01.01.1985",
    "Meslek: Mühendis", "Meslek: Öğretmen",
    "Baba Adı: Ali", "Anne Adı: Hatice",
    "Medeni Hal: Evli", "Uyruk: T.C.",
    "Pasaport No: U123456",
]

CORPUS_NUMERIC = [
    "1.250,00 TL", "3.456,78 TL", "15.000,00 TL",
    "250.000,00 TL", "%18 KDV", "%8 KDV",
    "01.01.2024", "15/03/2024", "2024-06-30",
    "Saat: 09:30", "Saat: 14:00-17:00",
    "Sıra No: 001", "Sıra No: 142",
    "Madde 1:", "Madde 2.3:", "Paragraf 4:",
    "m²", "kg", "lt", "adet", "kutu", "paket",
]

CORPUS_LEGAL = [
    "T.C. YARGITAY KARARI", "T.C. DANIŞTAY",
    "TÜRK CEZA KANUNU MADDE 125",
    "İş bu sözleşme taraflarca okunmuş ve imzalanmıştır.",
    "Yukarıda belirtilen koşullar kabul edilmiştir.",
    "Taraflar arasında akdedilen bu sözleşme,",
    "Sözleşme tarihi: 20 Şubat 2024",
    "Sözleşme süresi: 1 (bir) yıl",
    "Fesih bildirimi 30 gün önceden yapılmalıdır.",
    "Uyuşmazlıklarda İstanbul Mahkemeleri yetkilidir.",
    "Noter tasdikli suret geçerlidir.",
    "Vekaletname ile temsil edilmektedir.",
]

CORPUS_MIXED_CHARS = [
    # Türkçe özel karakter yoğun
    "Şişli'deki şirkete gönderilmiştir.",
    "Öğrencilerin ödevi tamamlanmıştır.",
    "Çalışanların ücreti ödenmiştir.",
    "Güneşli bir günde İstanbul'daydım.",
    "Kış aylarında ısınma giderleri artar.",
    "Müdürlüğe başvuru formu doldurulacak.",
    "İçişleri Bakanlığı'nın açıklaması.",
    "Şirketin hissedarları toplantıda buluştu.",
    "Türkçe öğrenmek çok keyiflidir.",
    "Çalışkanlık başarının anahtarıdır.",
    "Özgürlük ve eşitlik temel değerlerimizdir.",
    "Küçük işletmeler ekonominin bel kemiğidir.",
    "İğne ipliğe döndü deyimi yaygın kullanılır.",
    "Şükran duymak mutluluğu artırır.",
]

ALL_TEXTS = (
    CORPUS_GENERAL * 3
    + CORPUS_INVOICE * 4
    + CORPUS_ADDRESS * 2
    + CORPUS_PERSONAL * 2
    + CORPUS_NUMERIC * 3
    + CORPUS_LEGAL * 2
    + CORPUS_MIXED_CHARS * 4
)


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

KNOWN_FONT_PATHS = [
    # Liberation (genellikle Linux'ta mevcut)
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    # DejaVu
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    # Ubuntu
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-I.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    # Freefont
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    # Noto
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    # macOS fallback
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def discover_fonts():
    available = [p for p in KNOWN_FONT_PATHS if os.path.exists(p)]
    if not available:
        print("[WARNING] No TTF fonts found. Using PIL default font. Quality will be limited.", file=sys.stderr)
    else:
        print(f"[Dataset] Found {len(available)} fonts.", file=sys.stderr)
    return available


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------

def get_font(font_path, size):
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def render_text_line(text, font_path, font_size, bg_color, text_color):
    """Render a single text line onto a white/near-white background."""
    font = get_font(font_path, font_size)

    # Measure text size
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x = random.randint(10, 30)
    padding_y = random.randint(6, 18)
    img_w = max(text_w + 2 * padding_x, 100)
    img_h = max(text_h + 2 * padding_y, 30)

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((padding_x, padding_y), text, fill=text_color, font=font)

    return img


def augment_image(img):
    """Apply random realistic augmentations."""
    aug = random.choices(
        ["none", "blur", "noise", "skew", "brightness", "stain", "combined"],
        weights=[10, 15, 15, 10, 15, 10, 25],
        k=1,
    )[0]

    if aug == "none":
        return img

    if aug in ("blur", "combined"):
        radius = random.uniform(0.3, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if aug in ("noise", "combined"):
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if aug in ("skew", "combined"):
        angle = random.uniform(-3.0, 3.0)
        img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)

    if aug in ("brightness", "combined"):
        factor = random.uniform(0.75, 1.25)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if aug in ("stain",):
        arr = np.array(img)
        h, w = arr.shape[:2]
        for _ in range(random.randint(1, 4)):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            r = random.randint(5, 25)
            color_delta = random.randint(-40, -10)
            for y in range(max(0, cy - r), min(h, cy + r)):
                for x in range(max(0, cx - r), min(w, cx + r)):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        arr[y, x] = np.clip(arr[y, x].astype(np.int16) + color_delta, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    return img


def vary_colors():
    """Return (bg_color, text_color) with realistic paper/ink variations."""
    style = random.choices(
        ["clean", "aged", "gray_bg", "dark_ink", "faint"],
        weights=[40, 20, 15, 15, 10],
        k=1,
    )[0]

    if style == "clean":
        bg = (255, 255, 255)
        text = (0, 0, 0)
    elif style == "aged":
        bg = (random.randint(235, 250), random.randint(225, 240), random.randint(200, 220))
        text = (random.randint(20, 60), random.randint(10, 50), random.randint(0, 40))
    elif style == "gray_bg":
        v = random.randint(240, 252)
        bg = (v, v, v)
        text = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
    elif style == "dark_ink":
        bg = (255, 255, 255)
        text = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))
    else:  # faint
        bg = (255, 255, 255)
        v = random.randint(80, 140)
        text = (v, v, v)

    return bg, text


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_dataset(output_dir, count, fonts):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = {}
    generated = 0
    skipped = 0

    font_sizes = [18, 20, 22, 24, 26, 28, 32, 36]

    print(f"[Dataset] Generating {count} samples → {output_dir}")

    texts = ALL_TEXTS.copy()
    # Ensure enough variety by cycling
    while len(texts) < count:
        texts += ALL_TEXTS

    random.shuffle(texts)
    selected_texts = texts[:count]

    for i, text in enumerate(selected_texts):
        text = text.strip()
        if not text:
            skipped += 1
            continue

        font_path = random.choice(fonts) if fonts else None
        font_size = random.choice(font_sizes)
        bg_color, text_color = vary_colors()

        try:
            img = render_text_line(text, font_path, font_size, bg_color, text_color)
            img = augment_image(img)

            filename = f"line_{i:06d}.png"
            img.save(output_path / filename)
            labels[filename] = text
            generated += 1

            if (i + 1) % 500 == 0:
                print(f"[Dataset] {i + 1}/{count} generated...", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[Dataset] Skipped sample {i}: {e}", file=sys.stderr)
            skipped += 1

    labels_path = output_path / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n[Dataset] Done!")
    print(f"  Generated : {generated}")
    print(f"  Skipped   : {skipped}")
    print(f"  Labels    : {labels_path}")
    print(f"\nTo start training:")
    print(f"  python scripts/train_trocr_turkish.py train \\")
    print(f"    --data_dir {output_dir} \\")
    print(f"    --output_dir ./models/trocr-turkish \\")
    print(f"    --base_model microsoft/trocr-base-handwritten \\")
    print(f"    --epochs 10 --batch_size 8")


def preview_samples(output_dir, n=5):
    """Print a few label samples for quick sanity check."""
    labels_path = Path(output_dir) / "labels.json"
    if not labels_path.exists():
        print("No labels.json found.")
        return
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    print(f"\nSample labels ({min(n, len(labels))}/{len(labels)}):")
    for fname, text in list(labels.items())[:n]:
        print(f"  {fname}: {text}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic Turkish OCR Dataset Generator")
    parser.add_argument("--output_dir", default="./training_data",
                        help="Output directory for images and labels.json")
    parser.add_argument("--count", type=int, default=5000,
                        help="Number of line images to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--preview", action="store_true",
                        help="Print sample labels after generation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fonts = discover_fonts()
    generate_dataset(args.output_dir, args.count, fonts)

    if args.preview:
        preview_samples(args.output_dir)


if __name__ == "__main__":
    main()
