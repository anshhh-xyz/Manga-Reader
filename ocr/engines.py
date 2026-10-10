# ocr/engines.py
from dataclasses import dataclass
from typing import List
from PIL import Image


@dataclass
class OCRBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0


def _poly_to_box(poly, text, conf=1.0) -> OCRBox:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return OCRBox(text, min(xs), min(ys), max(xs), max(ys), float(conf))


class OCREngine:
    name = "base"

    def read(self, image_path: str) -> List[OCRBox]:
        raise NotImplementedError


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self):
        import easyocr, torch
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def read(self, image_path):
        return [_poly_to_box(poly, text, conf)
                for poly, text, conf in self.reader.readtext(image_path)]


class FlorenceEngine(OCREngine):
    """Florence-2 <OCR_WITH_REGION>: text + quadrilateral per region."""
    name = "florence"

    def __init__(self, model_id="microsoft/Florence-2-large"):
        # Compatibility fixes for newer transformers
        from transformers.configuration_utils import PretrainedConfig
        PretrainedConfig.forced_bos_token_id = None
        from transformers import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
            PreTrainedTokenizerBase.additional_special_tokens = property(
                lambda self: [t for t in self.all_special_tokens if t not in [
                    self.bos_token, self.eos_token, self.unk_token, self.sep_token,
                    self.pad_token, self.cls_token, self.mask_token
                ]]
            )

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.dtype, trust_remote_code=True,
            attn_implementation="eager").to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def read(self, image_path):
        task = "<OCR_WITH_REGION>"
        img = Image.open(image_path).convert("RGB")
        inputs = self.proc(text=task, images=img, return_tensors="pt").to(self.dev, self.dtype)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3, do_sample=False)
        raw = self.proc.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.proc.post_process_generation(raw, task=task, image_size=img.size)[task]
        boxes = []
        for text, quad in zip(parsed["labels"], parsed["quad_boxes"]):
            text = text.replace("</s>", "").replace("<pad>", "").strip()
            if text:
                poly = [(quad[i], quad[i + 1]) for i in range(0, 8, 2)]
                boxes.append(_poly_to_box(poly, text))
        return boxes


ENGINES = {"easyocr": EasyOCREngine, "florence": FlorenceEngine}
