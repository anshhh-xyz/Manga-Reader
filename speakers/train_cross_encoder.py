# speakers/train_cross_encoder.py
"""
Fine-tune DeBERTa-v3 Pairwise Cross-Encoder for Manga Speaker Attribution
==========================================================================
Trains on 12,368 dialogue pairs from development sequences:
  Input:  [CLS] [P{page_a}] [PAN {panel_a}] text_a [SEP] [P{page_b}] [PAN {panel_b}] text_b [SEP]
  Output: Logits for [different_speaker (0), same_speaker (1)]

Validation split is strictly grouped by sequence_id (using official val split_ids)
to prevent data leakage across pages.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "cache" / "cross_encoder_dataset.json"
WEIGHTS_DIR = ROOT / "weights" / "deberta_speaker_cross_encoder"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
BATCH_SIZE = 16
MAX_LEN = 96
LR = 2e-5
EPOCHS = 2
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class MangaPairDataset(Dataset):
    def __init__(self, pairs: List[Dict], tokenizer, max_len: int = MAX_LEN):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        text_a = item["text_a"]
        text_b = item["text_b"]
        label = item["label"]

        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "seq_id": item["seq_id"],
        }


def load_dataset_splits():
    from evaluation.common import split_ids

    val_sids = set(split_ids("val"))
    all_data = json.load(open(DATA_PATH, encoding="utf-8"))

    train_pairs = [d for d in all_data if d["seq_id"] not in val_sids]
    val_pairs = [d for d in all_data if d["seq_id"] in val_sids]

    print(f"Total dataset: {len(all_data)} pairs")
    print(f"  Train: {len(train_pairs)} pairs ({len(set(d['seq_id'] for d in train_pairs))} sequences)")
    print(f"  Val  : {len(val_pairs)} pairs ({len(set(d['seq_id'] for d in val_pairs))} sequences)")

    return train_pairs, val_pairs


def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)

    auc = roc_auc_score(all_labels, all_probs)
    return auc, np.array(all_probs), np.array(all_labels)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Initialize with 2 labels for binary same-speaker classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_pairs, val_pairs = load_dataset_splits()
    train_ds = MangaPairDataset(train_pairs, tokenizer)
    val_ds = MangaPairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initial zero-shot evaluation
    val_auc_init, _, _ = evaluate_model(model, val_loader, device)
    print(f"\nInitial Val Pairwise AUC (untrained head): {val_auc_init:.4f}")

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    best_auc = val_auc_init

    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    print(f"\nStarting training: {EPOCHS} epochs, {total_steps} total steps...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if step % 150 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation after epoch
        val_auc, _, _ = evaluate_model(model, val_loader, device)
        print(f"\n>>> Epoch {epoch} Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            print(f"  New best AUC! Saving model to: {WEIGHTS_DIR}")
            model.save_pretrained(WEIGHTS_DIR)
            tokenizer.save_pretrained(WEIGHTS_DIR)

    print(f"\nTraining Complete. Best Validation Pairwise AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
