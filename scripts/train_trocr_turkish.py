#!/usr/bin/env python3
"""
TrOCR Fine-tuning Script for Turkish OCR

This script fine-tunes microsoft/trocr-base-printed on a Turkish text dataset
for significantly improved Turkish character recognition (ç, ş, ğ, ü, ö, ı, İ).

Usage:
  1. Prepare your dataset:
     - Create a directory with line-cropped images and a labels.json file
     - labels.json format: {"image_filename.png": "ground truth text", ...}

  2. Run training:
     python scripts/train_trocr_turkish.py \
       --data_dir ./training_data \
       --output_dir ./models/trocr-turkish \
       --epochs 10 \
       --batch_size 8

  3. The fine-tuned model will be automatically used by ocr_inference.py
     when placed in ./models/trocr-turkish/

Dataset preparation tips:
  - Crop individual text lines from documents (not full pages)
  - Aim for at least 5,000+ line-image pairs for good results
  - Include variety: different fonts, sizes, quality levels
  - Include Turkish-specific characters in ground truth
  - Recommended image size: width 300-800px, height 30-60px
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)


class TurkishOCRDataset(Dataset):
    """Dataset for Turkish OCR line images with ground truth labels."""

    def __init__(self, data_dir, processor, max_target_length=128):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_target_length = max_target_length

        labels_path = self.data_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.json not found in {data_dir}. "
                "Create it with format: {\"image.png\": \"ground truth text\", ...}"
            )

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        self.samples = []
        for img_name, text in self.labels.items():
            img_path = self.data_dir / img_name
            if img_path.exists() and text.strip():
                self.samples.append((str(img_path), text.strip()))

        if not self.samples:
            raise ValueError(f"No valid image-label pairs found in {data_dir}")

        print(f"[Training] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def compute_cer(pred_str, label_str):
    """Compute Character Error Rate."""
    if not label_str:
        return 0.0 if not pred_str else 1.0

    d = [[0] * (len(label_str) + 1) for _ in range(len(pred_str) + 1)]
    for i in range(len(pred_str) + 1):
        d[i][0] = i
    for j in range(len(label_str) + 1):
        d[0][j] = j
    for i in range(1, len(pred_str) + 1):
        for j in range(1, len(label_str) + 1):
            cost = 0 if pred_str[i - 1] == label_str[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(pred_str)][len(label_str)] / len(label_str)


def train(args):
    print("=" * 60)
    print("TrOCR Turkish Fine-tuning")
    print("=" * 60)

    base_model = args.base_model
    print(f"[Training] Base model: {base_model}")
    print(f"[Training] Data directory: {args.data_dir}")
    print(f"[Training] Output directory: {args.output_dir}")

    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    print("[Training] Loading dataset...")
    full_dataset = TurkishOCRDataset(args.data_dir, processor, max_target_length=128)

    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[Training] Train: {train_size}, Validation: {val_size}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=min(500, train_size // args.batch_size),
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )

    print("[Training] Starting training...")
    trainer.train()

    print(f"[Training] Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("[Training] Running evaluation on validation set...")
    predictions = trainer.predict(val_dataset)

    pred_ids = predictions.predictions
    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    total_cer = sum(compute_cer(p, l) for p, l in zip(pred_strs, label_strs))
    avg_cer = total_cer / len(pred_strs) if pred_strs else 0

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"  Average CER: {avg_cer:.4f} ({(1 - avg_cer) * 100:.1f}% character accuracy)")
    print(f"  Model saved to: {args.output_dir}")
    print(f"{'=' * 60}")

    print("\nSample predictions:")
    for i in range(min(5, len(pred_strs))):
        print(f"  Label: {label_strs[i]}")
        print(f"  Pred:  {pred_strs[i]}")
        print()


def create_sample_dataset(output_dir):
    """Create a small sample dataset for testing the training pipeline."""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)
    labels = {}

    sample_texts = [
        "Turkiye Cumhuriyeti",
        "Istanbul Buyuksehir Belediyesi",
        "Fatura Tarihi: 01.01.2024",
        "Toplam Tutar: 1.250,00 TL",
        "Vergi Dairesi: Kadikoy",
        "Musteri Numarasi: 12345",
        "Adres: Ankara Caddesi No: 42",
        "Telefon: 0212 555 1234",
        "KDV Orani: %18",
        "Odeme Vadesi: 30 Gun",
    ]

    for i, text in enumerate(sample_texts):
        img = Image.new("RGB", (600, 50), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)

        filename = f"sample_{i:04d}.png"
        img.save(os.path.join(output_dir, filename))
        labels[filename] = text

    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"[Sample] Created {len(labels)} sample images in {output_dir}")
    print("[Sample] This is a minimal dataset for pipeline testing only.")
    print("[Sample] For real training, prepare 5,000+ labeled line images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for Turkish OCR")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", required=True, help="Directory with images + labels.json")
    train_parser.add_argument("--output_dir", default="./models/trocr-turkish", help="Where to save the model")
    train_parser.add_argument("--base_model", default="microsoft/trocr-base-printed",
                              help="Base model to fine-tune")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--learning_rate", type=float, default=5e-5)

    sample_parser = subparsers.add_parser("create-sample", help="Create sample dataset for testing")
    sample_parser.add_argument("--output_dir", default="./training_data", help="Output directory")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "create-sample":
        create_sample_dataset(args.output_dir)
    else:
        parser.print_help()
