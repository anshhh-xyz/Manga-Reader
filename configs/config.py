from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUTPUTS = ROOT / "outputs"

SEQUENCES_JSON = DATASET / "sequences.json"
DEV_REFERENCES = DATASET / "development" / "labels.jsonl"  
SAMPLE_SUBMISSION = DATASET / "sample_submission.jsonl"
SCORER = DATASET / "score.py"

SPLITS_JSON = DATASET / "splits.json"
REF_TRAIN = DATASET / "ref_train.jsonl"
REF_VAL = DATASET / "ref_val.jsonl"

VAL_FRACTION = 0.2