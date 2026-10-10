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

ALIGNED_PATH      = "outputs/alignment/aligned_boxes.json"
FILTER_MODEL_PATH = "outputs/filtering/filter_model.joblib"
OOF_PATH          = "outputs/filtering/filter_oof.pkl"
PANELS_PATH       = "outputs/panels_dev.json"
READING_RTL       = True   # placeholder, measured in Step 6
FILTER_THRESHOLD  = 0.5    # placeholder, tuned in Step 4
PAGE_GATE         = 0.0    # placeholder, tuned in Step 4