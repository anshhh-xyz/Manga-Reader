# speakers/character_memory.py


"""cluster() converts pairwise probabilities into distances and uses agglomerative clustering to group boxes likely belonging to the same speaker; optionally merges singleton clusters.
relabel() names the resulting clusters char1, char2, etc., based on their first appearance.
labels_from_P() assigns NARRATION to narration boxes and character IDs to the remaining boxes.
assign_speakers() creates pairwise features, gets same-speaker probabilities from the trained classifier, builds a probability matrix, and calls the labeling logic.
Overall: This file converts pairwise speaker predictions into consistent character labels across the manga sequence."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from .adapters import TAG_NARR
from .pair_features import seq_pair_features, prob_matrix

def cluster(P, t, merge_singletons=True, max_k=None):
    n = len(P)
    if n == 1:
        return np.zeros(1, dtype=int)
    D = np.clip(1 - P, 0, 1); np.fill_diagonal(D, 0)
    lab = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                  linkage="average", distance_threshold=t).fit_predict(D)
    if max_k and len(set(lab)) > max_k and n > max_k:      # hard cap on speaker count
        lab = AgglomerativeClustering(n_clusters=max_k, metric="precomputed",
                                      linkage="average").fit_predict(D)
    if merge_singletons:
        sizes = np.bincount(lab)
        big = [c for c in range(len(sizes)) if sizes[c] >= 2]
        new = lab.copy()
        if big:
            for c in np.where(sizes == 1)[0]:
                i = int(np.where(lab == c)[0][0])
                new[i] = max(big, key=lambda k: P[i, lab == k].mean())
        lab = new
    return lab

def relabel(lab):
    names, out = {}, []
    for c in lab:                                  # first appearance -> char1, char2, ...
        names.setdefault(c, f"char{len(names) + 1}")
        out.append(names[c])
    return out

def labels_from_P(boxes, P, cfg):
    story = [k for k, b in enumerate(boxes) if b["tag"] != TAG_NARR]
    labels = ["NARRATION"] * len(boxes)
    if story:
        names = relabel(cluster(P, cfg["threshold"], cfg["merge_singletons"], cfg.get("max_clusters")))
        for k, nm in zip(story, names):
            labels[k] = nm
    return labels

def assign_speakers(boxes, emb, sizes, model, cfg):
    """Inference entry point: used by pipeline/inference.py (Step 8)."""
    story = [k for k, b in enumerate(boxes) if b["tag"] != TAG_NARR]
    if not story:
        return ["NARRATION"] * len(boxes)
    sub = [boxes[k] for k in story]
    X, _, pairs = seq_pair_features(sub, emb[story], sizes)
    P = prob_matrix(len(sub), pairs, model.predict_proba(X)[:, 1]) if pairs else np.ones((1, 1))
    return labels_from_P(boxes, P, cfg)