# speakers/small_lm_diarizer.py
"""
Small LM Dialogue Diarizer
===========================
Uses Qwen2.5-0.5B-Instruct (plain causal LM, ~1GB, text-only) to assign speakers
as a lightweight alternative to the 3B VL model.

The hypothesis: smaller but faster => more consistent JSON output with less
hallucination drift, better for structured-output tasks like speaker labelling.

Fallback: Phase 5 Pairwise XGBoost if JSON parsing fails.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_SMALL = ROOT / "weights" / "qwen_small_diarizer"
HUB_ID_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"

_PROCESSOR = None
_MODEL = None


def _load_model():
    """Lazy-load small Qwen2.5 causal LM."""
    global _PROCESSOR, _MODEL
    if _MODEL is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = str(WEIGHTS_SMALL) if WEIGHTS_SMALL.exists() else HUB_ID_SMALL
        print(f"[SmallLMDiarizer] Loading model from: {model_path}")

        _PROCESSOR = AutoTokenizer.from_pretrained(model_path)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        _MODEL.eval()
        print("[SmallLMDiarizer] Model loaded.")
    return _PROCESSOR, _MODEL


PROMPT_TEMPLATE = """You are a manga dialogue diarizer. Given numbered utterances from a manga sequence, assign each to its speaker.

Speakers: "char1", "char2", "char3", "char4", "NARRATION"

Rules:
1. Alternate speakers in back-and-forth dialogue (char1 → char2 → char1...).
2. Wide caption boxes describing scenes/time/place → "NARRATION".
3. Keep character labels consistent across all pages.
4. When a third person enters, use char3, etc.
5. Multiple consecutive lines in the same panel by the same speaker stay the same label.

EXAMPLE:
[Page 1 | Panel 0] Box 0: "Hey, you okay?"
[Page 1 | Panel 0] Box 1: "Yeah... just tired."
[Page 1 | Panel 1] Box 2: "THREE MONTHS LATER."
[Page 2 | Panel 0] Box 3: "Don't forget your lunch!"
[Page 2 | Panel 0] Box 4: "Thanks mom."

OUTPUT:
[{"id":0,"speaker":"char1"},{"id":1,"speaker":"char2"},{"id":2,"speaker":"NARRATION"},{"id":3,"speaker":"char3"},{"id":4,"speaker":"char2"}]

Now diarize:
{transcript}

Respond ONLY with the JSON array (no explanation):"""


def _build_transcript(boxes: List[Dict]) -> str:
    lines = []
    for i, b in enumerate(boxes):
        pg = b.get("page", 0) + 1
        pnl = b.get("panel", 0)
        t = b.get("text", "").strip()
        lines.append(f'[Page {pg} | Panel {pnl}] Box {i}: "{t}"')
    return "\n".join(lines)


def _parse_json(text: str, n_boxes: int) -> Optional[List[str]]:
    """Extract JSON array from LLM response."""
    # Try to find a JSON array anywhere in the output
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if not isinstance(data, list):
            return None

        mapping = {}
        valid_spkrs = {"char1", "char2", "char3", "char4", "NARRATION"}
        for idx_counter, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            idx = item.get("id", item.get("box", item.get("index", idx_counter)))
            spk = item.get("speaker", item.get("label", item.get("character", "char1")))
            spk_str = str(spk).strip()
            # Normalize
            if spk_str.lower() in ("narr", "narration", "narrator"):
                spk_str = "NARRATION"
            elif not spk_str.startswith("char") and spk_str not in valid_spkrs:
                digits = re.findall(r"\d+", spk_str)
                spk_str = f"char{digits[0]}" if digits else "char1"
            if spk_str in valid_spkrs:
                try:
                    mapping[int(idx)] = spk_str
                except (ValueError, TypeError):
                    mapping[idx_counter] = spk_str

        if len(mapping) >= n_boxes * 0.70:
            result = []
            last = "char1"
            for i in range(n_boxes):
                s = mapping.get(i, last)
                result.append(s)
                last = s
            return result
    except Exception as e:
        print(f"[SmallLMDiarizer] JSON parse error: {e}")
    return None


def small_lm_assign_speakers(
    boxes: List[Dict],
    page_sizes: Optional[List[Tuple[float, float]]] = None,
) -> List[str]:
    """
    Assign speakers using Qwen2.5-0.5B-Instruct with XGBoost fallback.
    """
    if not boxes:
        return []
    if len(boxes) == 1:
        return ["NARRATION"] if boxes[0].get("tag") == "narration" else ["char1"]

    try:
        tokenizer, model = _load_model()
        transcript = _build_transcript(boxes)
        prompt = PROMPT_TEMPLATE.replace("{transcript}", transcript)

        # Use chat template for instruction following
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to("cuda:0")

        max_new = min(512, len(boxes) * 25 + 64)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )
        output_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        parsed = _parse_json(response, len(boxes))
        if parsed and len(parsed) == len(boxes):
            return parsed
        else:
            print(f"[SmallLMDiarizer] JSON parse failed, response snippet: {response[:200]}")

    except Exception as e:
        print(f"[SmallLMDiarizer] Error: {e}. Falling back to XGBoost.")

    # Fallback
    from speakers.enhanced_spatial_clusterer import cluster_assign
    return cluster_assign(boxes, page_sizes)
