# Greek Squeezes

OCR pipeline for the TROGS / ICDAR 2026 Greek Squeezes task. The active workflow is a
dual-light TrOCR line-recognizer family plus a character-posterior lattice/ranker correction layer.

## Current Result

The most recent frozen production run evaluated the deployed charpost pipeline on the separate
`papadimas/trogs-26-test-images` test split:

| Dataset | Images | Squeezes | Rows | Characters | Errors | CER |
|---|---:|---:|---:|---:|---:|---:|
| `trogs26-test` | 100 | 50 | 605 | 7,919 | 651 | 0.0822073494 |

Run summary:
`https://huggingface.co/buckets/papadimas/greek_squeezes/tree/data/prod_runs/trogs26-test/summary.json`

The underlying frozen `SqueezeTrOCR` line-model reference in `data/results/line_frozen_eval.json` records:

| Split | Setting | CER |
|---|---|---:|
| validation | train-only LM | 0.0358126722 |
| test | train-only LM | 0.0320499480 |
| test | train+validation LM | 0.0310093652 |

## Active Files

- `greek_squeezes.ipynb` / `greek_squeezes.md`: cache-guarded notebook workflow.
- `squeeze_runtime/`: runtime modules used by the notebook and Modal.
- `modal_app.py`: Modal runner for artifact building, upload, status, and production eval.
- `report/`: final English report source and rendered PDF.
- `data/`: ignored local artifact tree; sync or build this before running the notebook end-to-end.
- `old/`: archived research notebooks, reports, scripts, and logs. Do not use for active claims.

## Model Stack

The line recognizer is `SqueezeTrOCR`: tile4 dual-light TrOCR using
`microsoft/trocr-large-printed`, real crops only, 6000 optimizer steps, batch 96, phase registration,
channel swap during training, and mono dropout 0.15.

Dual-light packing:

- `R = Rotation1`
- `G = phase-registered Rotation2`
- `B = min(R, G)`

The charpost layer trains tile1 dual-light classifiers, caches character logits, builds row candidates
with order 6/10/12 character LMs, and fits a ridge ranker over visual, length, and LM features.

Clean OOF is required: each held-out fold is excluded from the fold TrOCR checkpoint, the charpost
classifier, and the fold character LMs. Validation/test/prod data are report-only after the recipe is
frozen.

## Setup

Python 3.10-3.12 is recommended for the ML stack.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch wheel appropriate for the host CUDA/MPS setup before or after
installing the requirements.

## Sync Artifacts

The full artifact tree is large and is not tracked by git. Sync the Hugging Face bucket into the repo
root when you want the notebook to pass cache guards without rebuilding everything:

```bash
hf auth login
hf sync hf://buckets/papadimas/greek_squeezes .
```

Expected local layout:

```text
data/line_recognizer/all_train
data/line_recognizer/oof_fold0 ... oof_fold4
data/splits/oof_folds.json
data/character_posterior/
data/results/
```

## Run Locally

Open `greek_squeezes.ipynb` and run all cells. With synced artifacts, the notebook validates and skips
completed stages. Without artifacts, it can rebuild missing pieces only if `RUN_TRAINING_IF_MISSING`
is true, but the full charpost stack is intended for a GPU/Modal run.

## Modal

Create the Hugging Face secret once:

```bash
uvx modal secret create huggingface-token HF_TOKEN=hf_...
```

Prepare inputs:

```bash
uvx modal run modal_app.py --prepare
```

Build reusable charpost artifacts and upload them:

```bash
uvx modal run modal_app.py --run --upload
```

Run the complete pipeline from line-recognizer training through final charpost decodes:

```bash
uvx modal run --detach modal_app.py --full --bg --upload-reusable
```

`--full` trains or reuses `data/line_recognizer/all_train` and the five
`data/line_recognizer/oof_fold*` checkpoints before starting charpost. Use `--skip-line-training`
only when you want the older behavior that requires those checkpoints to already exist.

Run only the initial `SqueezeTrOCR` line-recognizer stage:

```bash
uvx modal run --detach modal_app.py --line-only --bg
```

Check remote artifact status:

```bash
uvx modal run modal_app.py --status
```

Run the frozen pipeline on the external TROGS-26 test images:

```bash
uvx modal run modal_app.py --prod \
  --prod-source papadimas/trogs-26-test-images \
  --prod-source-split test \
  --prod-name trogs26-test
```

Upload a completed production result:

```bash
uvx modal run modal_app.py --upload-prod-result-only --prod-name trogs26-test
```

## Report

The report source is `report/report.tex`; the rendered PDF is `report/report.pdf`.

```bash
cd report
lualatex report.tex
lualatex report.tex
```
