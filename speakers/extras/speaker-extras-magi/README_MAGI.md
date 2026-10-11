# Magi speakers for the Manga Task

Replaces the weak layout-based speaker model with **Magi v2** (`ragavsachdeva/magiv2`), used as-is.

**What Magi does here:** for each page it finds text boxes and character boxes, links each text box to
the character who says it, and gives characters an identity that is consistent across the pages you pass
in one call. We match *your* OCR balloon boxes to Magi's text boxes, read off the linked character, and
turn character identities into `char1`, `char2`, ... Your OCR, filter and reading order are untouched.

**Licence / citation:** the model card allows personal, research, non-commercial and not-for-profit use.
Cite the model page and the papers it lists in your README, and state clearly that Magi is external work
and that you used it unmodified (your own contribution: matching, label policy, tuning, evaluation).

## Install (once)

Put the `speakers_magi/` folder in your repo root, next to `speakers/`. Then:

    pip install transformers pillow numpy einops shapely timm scipy

(you already have torch + CUDA). If loading the model fails with "No module named X", `pip install X`.

## Steps (run from the repo root)

1. **Self-test the logic** (no GPU, no model): `python -m speakers_magi.selftest`
   Expect: `all 8 checks passed`.
2. **Probe Magi on one sequence**: `python -m speakers_magi.probe`
   First run downloads the model. It prints what Magi returns and writes `outputs/magi/probe_*.png`.
   Open the PNGs: red = text boxes, blue = characters, green = text->character links. If the boxes sit on
   the balloons and characters, you are good. If not, or it crashes, paste the output to Claude.
3. **Run Magi on the dev set** (cached, safe to stop and re-run): `python -m speakers_magi.run_magi --split dev`
4. **Score it and tune** (one command, ~24 runs of your scorer): `python -m speakers_magi.grid`
   Prints every setting's `balanced_joint_f1`, saves the winner to `outputs/magi/best_cfg.json`.
   Compare with: floor 0.2485, your old layout model 0.3418, gold-speaker control 0.6639.
   (One setting only: `python -m speakers_magi.run_oracle_magi --score magi_try`)
5. **Use it on the test set** (only after you are happy with dev):
   `python -m speakers_magi.run_magi --split test`, then in `pipeline/inference.py`:

       from speakers_magi.pipeline_hook import magi_labels
       labels = magi_labels(seq_id, boxes, split="test")   # boxes = filtered, ordered story boxes

   `boxes` need `"page"` (0-based), `"bbox"` ([x1,y1,x2,y2] original pixels) and `"tag"`
   (narration boxes must carry the same tag string as `TAG_NARR` in `speakers/adapter.py`).

## Settings the grid searches (what they mean)

- `unsure`: balloons Magi can't link to a character. `prev` copies the previous speaker, `common` uses the
  most frequent one. (Extra labels are penalised by the scorer, so we never invent a new one.)
- `maxk`: keep only the K most frequent characters per sequence (0 = no cap).
- `scope`: `global` = a Magi character id means the same speaker on all 3 pages; `page` = ids are
  page-local (safer if Magi's ids turn out not to be chapter-wide).
- `nearest`: when there is no text->character link, use the nearest character box.

## Things that may need a one-line edit

- `common.py -> page_paths`: pages are sorted file names. The page order must match the `page` index of
  your aligned boxes (same assumption as `speakers/adapter.py`).
- `run_magi.py -> call_magi`: tries the model-card call `do_chapter_wide_prediction` with an empty
  character bank, then `predict_detections_and_associations`. The probe tells you which one worked.
- Result keys used: `texts`, `characters`, `text_character_associations`, and `character_names` or
  `character_cluster_labels`. If the probe shows different names, change `core.py -> normalize_pages`.

## Honest limits

Written and tested with a stand-in model; the real Magi has not been run by the author of this package
(the sandbox it was built in cannot reach Hugging Face). Step 2 is the first real check.
