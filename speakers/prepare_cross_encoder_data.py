# speakers/prepare_cross_encoder_data.py
"""
Pairwise Dataset Generator for Cross-Encoder Training
======================================================
Extracts 12,368 dialogue pairs from aligned ground-truth manga sequences.
Formats each pair with lightweight structural layout tokens:
  [P{page}] [PAN {panel}] {text}

Saves to: outputs/cache/cross_encoder_dataset.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "cache" / "cross_encoder_dataset.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def format_utterance(box: Dict) -> str:
    pg = box.get("page", 0)
    pan = box.get("panel", -1)
    pan_str = f"[PAN {pan}]" if pan >= 0 else "[PAN ?]"
    text = (box.get("text") or "").strip()
    return f"[P{pg}] {pan_str} {text}"


def build_pair_dataset():
    from speakers.adapters import load_sequences

    print("Loading aligned ground-truth sequences...")
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences.")

    pairs_data: List[Dict] = []
    same_count = 0
    diff_count = 0

    for sid, boxes in sequences.items():
        # Keep dialogue/story boxes only
        story = [b for b in boxes if b.get("tag") != "narration"]
        n = len(story)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                sa = story[i].get("speaker")
                sb = story[j].get("speaker")
                if sa in (None, "UNKNOWN") or sb in (None, "UNKNOWN"):
                    continue

                label = 1 if sa == sb else 0
                if label == 1:
                    same_count += 1
                else:
                    diff_count += 1

                pairs_data.append({
                    "seq_id": sid,
                    "i": i,
                    "j": j,
                    "text_a": format_utterance(story[i]),
                    "text_b": format_utterance(story[j]),
                    "raw_text_a": (story[i].get("text") or "").strip(),
                    "raw_text_b": (story[j].get("text") or "").strip(),
                    "page_a": story[i].get("page", 0),
                    "page_b": story[j].get("page", 0),
                    "panel_a": story[i].get("panel", -1),
                    "panel_b": story[j].get("panel", -1),
                    "label": label,
                })

    print(f"\nExtracted {len(pairs_data)} valid dialogue pairs:")
    print(f"  Same-speaker (1): {same_count} ({100*same_count/len(pairs_data):.1f}%)")
    print(f"  Diff-speaker (0): {diff_count} ({100*diff_count/len(pairs_data):.1f}%)")

    OUT_PATH.write_text(json.dumps(pairs_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset to: {OUT_PATH}")
    return pairs_data


if __name__ == "__main__":
    build_pair_dataset()
