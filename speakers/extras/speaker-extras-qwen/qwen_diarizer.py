# speakers/qwen_diarizer.py
"""
Qwen2.5 Conversational Dialogue Diarizer
========================================
Uses Qwen2.5-VL-3B-Instruct to solve manga speaker attribution as a
conversational turn-taking problem.

Architecture:
1. Ordered utterance boxes (with page, panel, text) -> transcript prompt.
2. Few-shot conversational in-context guidance (turn alternation, multi-speaker clustering, narration).
3. Qwen2.5-VL generation -> JSON mapping of utterance index -> speaker label.
4. Automatic fallback to Phase 5 Pairwise XGBoost if JSON parsing or generation fails.

Model Weights:
    weights/qwen_diarizer (symlinked to local cache)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_QWEN = ROOT / "weights" / "qwen_diarizer"
FALLBACK_HUB_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_PROCESSOR = None
_MODEL = None


def _load_qwen():
    """Lazy-load Qwen2.5-VL model and processor."""
    global _PROCESSOR, _MODEL
    if _MODEL is None:
        model_path = str(WEIGHTS_QWEN) if WEIGHTS_QWEN.exists() else FALLBACK_HUB_ID
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        _PROCESSOR = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
        )
        _MODEL.eval()
    return _PROCESSOR, _MODEL


PROMPT_TEMPLATE = """You are an expert manga dialogue diarization engine.
Your task is to assign each numbered dialogue utterance in this 3-page sequence to its speaker ("char1", "char2", "char3", "char4", or "NARRATION").

Rules:
1. Turn-taking: In conversation, questions and answers usually alternate between speakers (char1 asks, char2 replies).
2. Multi-speaker: When new characters speak or are addressed by name, assign a new label (char3, char4). Do not lump all characters into char1!
3. Stacked bubbles: Adjacent dialogue lines inside the same panel with matching tone belong to the same character.
4. "NARRATION": Expository captions describing facts, time, or third-person exposition.
5. Consistent identity: Maintain the same character labels across all 3 pages (e.g., if the protagonist is char1 on Page 1, they are still char1 on Page 3).

--- FEW-SHOT EXAMPLE ---
Transcript:
[Page 1 | Panel 0] Box 0: "Hey Hachiken, are you feeling better?"
[Page 1 | Panel 0] Box 1: "Yeah. Komaba came by earlier."
[Page 1 | Panel 1] Box 2: "Oh, I see. He came to get his things."
[Page 1 | Panel 1] Box 3: "Why did he have to drop out to work?!"
[Page 2 | Panel 0] Box 4: "SHUCHI'IN ACADEMY WAS FOUNDED IN 1920."
[Page 3 | Panel 0] Box 5: "Good afternoon, boys! Did you finish your chores?"
[Page 3 | Panel 0] Box 6: "Yes, Mrs. Komaba."

Output JSON:
[
  {"id": 0, "speaker": "char1"},
  {"id": 1, "speaker": "char2"},
  {"id": 2, "speaker": "char1"},
  {"id": 3, "speaker": "char1"},
  {"id": 4, "speaker": "NARRATION"},
  {"id": 5, "speaker": "char3"},
  {"id": 6, "speaker": "char2"}
]
--- END EXAMPLE ---

Now diarize this sequence:
{transcript}

Respond ONLY with the JSON array:
"""


def _build_transcript(boxes: List[Dict]) -> str:
    lines = []
    for i, b in enumerate(boxes):
        pg = b.get("page", 0) + 1
        pnl = b.get("panel", 0)
        t = b.get("text", "").strip()
        lines.append(f'[Page {pg} | Panel {pnl}] Box {i}: "{t}"')
    return "\n".join(lines)


def _parse_qwen_json(text: str, n_boxes: int) -> Optional[List[str]]:
    """Extract and validate JSON array of speakers from LLM response."""
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
            # Normalize speaker name
            spk_str = str(spk).strip()
            if spk_str.lower() in ("narr", "narration"):
                spk_str = "NARRATION"
            elif not spk_str.startswith("char") and spk_str not in valid_spkrs:
                # e.g., "1" -> "char1"
                digits = re.findall(r"\d+", spk_str)
                spk_str = f"char{digits[0]}" if digits else "char1"
            if spk_str in valid_spkrs:
                try:
                    mapping[int(idx)] = spk_str
                except (ValueError, TypeError):
                    mapping[idx_counter] = spk_str

        # Check coverage
        if len(mapping) >= n_boxes * 0.70:
            result = []
            last = "char1"
            for i in range(n_boxes):
                s = mapping.get(i, last)
                result.append(s)
                last = s
            return result
    except Exception as exc:
        print(f"[QwenParser Debug] parse error: {exc}")
    return None


def qwen_assign_speakers(
    boxes: List[Dict],
    page_sizes: Optional[List[Tuple[float, float]]] = None,
) -> List[str]:
    """
    Assign speaker labels to dialogue boxes using Qwen2.5-VL with Phase 5 fallback.

    Args:
        boxes: List of dicts with 'page', 'panel', 'bbox', 'text', 'tag'.
        page_sizes: Optional list of (W, H) per page.

    Returns:
        List of speaker strings (e.g. ['char1', 'char2', ...]).
    """
    if not boxes:
        return []

    # Fast path for single box
    if len(boxes) == 1:
        return ["NARRATION"] if boxes[0].get("tag") == "narration" else ["char1"]

    # Try Qwen2.5 conversational diarization
    try:
        processor, model = _load_qwen()
        transcript = _build_transcript(boxes)
        prompt_text = PROMPT_TEMPLATE.replace("{transcript}", transcript)

        messages = [{"role": "user", "content": prompt_text}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=min(1024, len(boxes) * 35 + 100),
                do_sample=False,
            )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
            response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        parsed_labels = _parse_qwen_json(response, len(boxes))
        if parsed_labels and len(parsed_labels) == len(boxes):
            return parsed_labels

    except Exception as exc:
        print(f"[QwenDiarizer Warning] LLM generation error ({exc}). Falling back to XGBoost.")

    # Graceful Fallback: Phase 5 Pairwise XGBoost
    from speakers.enhanced_spatial_clusterer import cluster_assign
    return cluster_assign(boxes, page_sizes)
