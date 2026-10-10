
import pandas as pd

CLASS_MAP = {
    "non-story": 0,
    "dialogue": 1,
    "narration": 2,
}

def load_aligned(path):
    df = pd.read_json(path, orient="records")

    # Rename columns to the names expected by the filter
    df = df.rename(columns={
        "sequence_id": "seq_id",
        "ocr_text": "text",
        "order_index": "order_idx",
    })

    # Convert polygon points into bounding-box coordinates
    if "poly" in df.columns:
        df["x0"] = df["poly"].apply(lambda p: min(pt[0] for pt in p))
        df["y0"] = df["poly"].apply(lambda p: min(pt[1] for pt in p))
        df["x1"] = df["poly"].apply(lambda p: max(pt[0] for pt in p))
        df["y1"] = df["poly"].apply(lambda p: max(pt[1] for pt in p))

    required = [
        "seq_id", "page", "x0", "y0", "x1", "y1",
        "text", "conf", "tier", "ambiguous", "tag", "order_idx"
    ]

    missing = [c for c in required if c not in df.columns]
    assert not missing, (
        f"Missing columns: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )

    df["ambiguous"] = df["ambiguous"].fillna(False).astype(bool)
    df["conf"] = df["conf"].fillna(0.0)
    df["text"] = df["text"].fillna("").astype(str)

    return df.reset_index(drop=True)


def labels(df):
    return df["tag"].map(CLASS_MAP).fillna(-1).astype(int).to_numpy()


def training_mask(df):
    """High-tier examples + none-tier negatives; exclude ambiguous rows."""
    ok_tier = df["tier"].isin(["high", "none"])
    return (
        ok_tier & ~df["ambiguous"]
    ).to_numpy() & (labels(df) >= 0)
