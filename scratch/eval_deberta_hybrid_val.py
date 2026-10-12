# scratch/eval_deberta_hybrid_val.py
"""
Evaluate DeBERTa-v3 Cross-Encoder + Spatial XGBoost Hybrid on Validation Set
==============================================================================
Tests hybrid fusion:
  P_hybrid = alpha * P_xgb + (1 - alpha) * P_deberta
across alpha in [0.0 (pure DeBERTa), 0.3, 0.5, 0.7, 1.0 (pure XGBoost)].
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[4]
if not (ROOT / "pipeline").exists():
    ROOT = Path(r"d:\Projects\BYTE\Manga Task Reader AML-2")
sys.path.insert(0, str(ROOT))

from evaluation.common import split_ids, image_paths, score as scorer
from pipeline.inference_yolo import predict_sequence_hybrid_yolo
from speakers.enhanced_spatial_clusterer import extract_pair_features, load_model, cluster_probabilities, relabel
import speakers.enhanced_spatial_clusterer as esc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = ROOT / "weights" / "deberta_speaker_cross_encoder"

_TOKENIZER = None
_DEBERTA_MODEL = None

def get_deberta():
    global _TOKENIZER, _DEBERTA_MODEL
    if _DEBERTA_MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
        _DEBERTA_MODEL = AutoModelForSequenceClassification.from_pretrained(str(WEIGHTS_DIR))
        _DEBERTA_MODEL.to(DEVICE)
        _DEBERTA_MODEL.eval()
    return _TOKENIZER, _DEBERTA_MODEL

def compute_deberta_probs(boxes, pairs):
    if not pairs:
        return np.array([])
    tokenizer, model = get_deberta()
    
    texts_a = []
    texts_b = []
    for i, j in pairs:
        pa, pka = boxes[i].get("page", 0), boxes[i].get("panel", -1)
        pb, pkb = boxes[j].get("page", 0), boxes[j].get("panel", -1)
        pan_a = f"[PAN {pka}]" if pka >= 0 else "[PAN ?]"
        pan_b = f"[PAN {pkb}]" if pkb >= 0 else "[PAN ?]"
        ta = f"[P{pa}] {pan_a} {(boxes[i].get('text') or '').strip()}"
        tb = f"[P{pb}] {pan_b} {(boxes[j].get('text') or '').strip()}"
        texts_a.append(ta)
        texts_b.append(tb)
        
    probs_list = []
    batch_size = 32
    with torch.no_grad():
        for k in range(0, len(texts_a), batch_size):
            ba = texts_a[k:k+batch_size]
            bb = texts_b[k:k+batch_size]
            enc = tokenizer(ba, bb, truncation=True, max_length=96, padding=True, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.extend(p)
            
    return np.array(probs_list)

def make_hybrid_cluster_assign(alpha: float):
    def hybrid_cluster_assign(boxes, page_sizes=None, cfg=None, **kwargs):
        xgb_model, base_cfg = load_model()
        run_cfg = base_cfg.copy()
        if cfg:
            run_cfg.update(cfg)
            
        story_idx = [k for k, b in enumerate(boxes) if b.get("tag") != "narration"]
        labels = ["NARRATION"] * len(boxes)
        if not story_idx:
            return labels
        if len(story_idx) == 1:
            labels[story_idx[0]] = "char1"
            return labels
            
        sub_boxes = [boxes[k] for k in story_idx]
        if page_sizes is None:
            page_sizes = [(1000.0, 1500.0)] * 3
            
        X, pairs = extract_pair_features(sub_boxes, page_sizes)
        if X.shape[1] != xgb_model.n_features_in_:
            X = X[:, :xgb_model.n_features_in_]
            
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        
        if alpha < 1.0:
            p_deb = compute_deberta_probs(sub_boxes, pairs)
            p_final = alpha * p_xgb + (1.0 - alpha) * p_deb
        else:
            p_final = p_xgb
            
        P = np.eye(len(sub_boxes))
        for (i, j), p in zip(pairs, p_final):
            P[i, j] = P[j, i] = p
            
        assigned = relabel(
            cluster_probabilities(
                P,
                t=float(run_cfg.get("threshold", 0.50)),
                merge_singletons=bool(run_cfg.get("merge_singletons", True)),
                max_k=int(run_cfg.get("max_clusters", 4)),
            )
        )
        for k, name in zip(story_idx, assigned):
            labels[k] = name
        return labels
    return hybrid_cluster_assign

def evaluate_alpha(alpha: float, val_ids, all_refs):
    orig_assign = esc.cluster_assign
    esc.cluster_assign = make_hybrid_cluster_assign(alpha)
    
    preds = {}
    try:
        for sid in val_ids:
            preds[sid] = predict_sequence_hybrid_yolo(sid, postprocess=True, crop_recovery=True)
    finally:
        esc.cluster_assign = orig_assign
        
    refs = {i: all_refs[i] for i in val_ids}
    report = scorer.evaluate(refs, preds)
    macro = report["macro"]
    return macro

if __name__ == "__main__":
    val_ids = split_ids("val")
    all_refs = scorer.load_jsonl(ROOT / "dataset" / "development" / "labels.jsonl", reference=True)
    
    print("=== DeBERTa-v3 Cross-Encoder Hybrid Sweep on Val Set ===")
    alphas = [1.0, 0.7, 0.5, 0.3, 0.0]
    
    results = {}
    for a in alphas:
        print(f"\nEvaluating alpha = {a:.1f} (XGB: {a*100:.0f}%, DeBERTa: {(1-a)*100:.0f}%)...")
        macro = evaluate_alpha(a, val_ids, all_refs)
        results[a] = macro
        print(f"  balanced_joint_f1:           {macro['balanced_joint_f1']:.4f}")
        print(f"  speaker_accuracy_on_matched: {macro['speaker_accuracy_on_matched']:.4f}")
        print(f"  joint_f1:                    {macro['joint_f1']:.4f}")
        print(f"  text_order_score:            {macro['text_order_score']:.4f}")
        
    print("\n" + "="*70)
    print(f"{'Alpha':<8} {'XGB/DeB':<12} {'Balanced F1':<14} {'Spk Acc':<12} {'Joint F1':<10} {'Text Order':<10}")
    print("="*70)
    for a in alphas:
        m = results[a]
        name = f"{int(a*100)}/{int((1-a)*100)}"
        print(f"{a:<8.1f} {name:<12} {m['balanced_joint_f1']:<14.4f} {m['speaker_accuracy_on_matched']:<12.4f} {m['joint_f1']:<10.4f} {m['text_order_score']:<10.4f}")
    print("="*70)
