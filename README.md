# Greek Squeezes

OCR pipeline for the TROGS / ICDAR 2026 Greek Squeezes task. The active workflow is a
dual-light TrOCR line-recognizer family plus a character-posterior lattice/ranker correction layer.

## Two reproducible recipes

Two charpost recipes are kept public and reproducible. They share the dual-light line recognizer,
the OOF fold classifiers/logits, and the all-train charpost classifier; they differ in the
character count language model(s), the candidate-table feature stamp, and the row selector.

| Recipe | Git branch | Bucket namespace | Prod slug | Internal CER (errs) | Official v2 |
|---|---|---|---|---|---|
| **PPM unified release** | `main` | `ppm_unified/` | `trogs26-test-ppm` | 0.0838 (664) | `all_cer` 0.0786, `opt_cer` 0.0801, `cer` 0.1166 |
| **Frozen three-order** — pre-simplification | `frozen/three-order` (tag `recipe-frozen-three-order`) | `data/` | `trogs26-test` | 0.0822 (651) | `all_cer` 0.0771, `opt_cer` 0.0784, `cer` 0.1145 |

Both were evaluated on the separate `papadimas/trogs-26-test-images` test split (100 images, 50
squeezes, 605 rows, 7,919 characters). The two recipes differ by thirteen errors in 7,919
characters; this external comparison is not used for model selection.

- Simplification prod summary:
  `https://huggingface.co/buckets/papadimas/greek_squeezes/tree/ppm_unified/data/prod_runs/trogs26-test-ppm/summary.json`
- Frozen prod summary:
  `https://huggingface.co/buckets/papadimas/greek_squeezes/tree/data/prod_runs/trogs26-test/summary.json`

The first simplification replaced the frozen order-6/10/12 count-LM ranker features (`lm6_sum` /
`lm10_sum` / `lm12_sum`, feature version `lattice_length_delta_v2`) with a single order-12
character count model scored by adaptive-order PPM-C (`ppm_sum`, feature version `ppm_unified_v1`).
The active selector tunes `visual_sum + lambda * ppm_sum` directly on nested OOF corpus CER
(feature version `ppm_interpolation_v2`). Modal selected `lambda=0.84` in every outer fold, with
nested OOF CER 0.049770 (2,672/53,687); internal validation/test CER are 0.022929 and 0.022207.
The rescored external production set scores 0.083849 (664/7,919), official v2 `all_cer` 0.0786.
`ORDERS` and `FEATURE_VERSION` are set in `modal_app.py` / `squeeze_runtime/artifact_checks.py`
and differ per branch. See `report/report.tex` Section 2 for the metric convention and the
reconciliation of internal CER with the official v2 score.

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

## Model Stack

The line recognizer is `SqueezeTrOCR`: tile4 dual-light TrOCR using
`microsoft/trocr-large-printed`, real crops only, 6000 optimizer steps, batch 96, phase registration,
channel swap during training, and mono dropout 0.15.

Dual-light packing:

- `R = Rotation1`
- `G = phase-registered Rotation2`
- `B = min(R, G)`

The charpost layer trains tile1 dual-light classifiers, caches character logits, builds row candidates
with a single order-12 character count model scored by adaptive-order PPM-C (ppm_unified recipe, a
post-freeze simplification of the earlier order-6/10/12 recipe -- see `report/report.tex` Section 2),
and selects rows with the OOF-tuned score `visual_sum + 0.84 * ppm_sum`.

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

The full artifact tree is large and is not tracked by git. Each recipe lives under its own bucket
namespace (see the table above); sync the **matching** namespace into the local `data/` tree that
the runtime reads (`squeeze_runtime/runtime_config.py` resolves `data/`).

```bash
hf auth login
# Simplification (ppm_unified) — branch main:
hf sync hf://buckets/papadimas/greek_squeezes/ppm_unified/data ./data
# Frozen three-order — branch frozen/three-order:
hf sync hf://buckets/papadimas/greek_squeezes/data ./data
```

Syncing the whole bucket root (`hf sync hf://buckets/papadimas/greek_squeezes .`) pulls both
namespaces plus `archives/`; use the targeted sync above when you want one recipe's artifacts in
`data/`.

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

Section 8 of the notebook walks through the charpost decode mechanism step by step on a real
row from the cached decode and regenerates every figure used by `report/report.tex` into
`report/figs/` (builders in `squeeze_runtime/report_figs.py`). Every builder there reads cached
`data/` artifacts and is skipped individually when its inputs have not been synced or built yet.

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

`--upload` syncs the volume's `/data/` tree into the namespace named by `BUCKET_URI` in
`modal_app.py`, which differs per branch: `main` writes `bucket/ppm_unified/data/`, and
`frozen/three-order` writes `bucket/data/`. Run it from a checkout of the recipe whose artifacts
you want published, and it will only touch that recipe's namespace.

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

Run the pipeline on the external TROGS-26 test images. Use the prod slug of the recipe you are
running — `trogs26-test-ppm` for the simplification (branch `main`), `trogs26-test` for the frozen
three-order recipe (branch `frozen/three-order`):

```bash
uvx modal run modal_app.py --prod \
  --prod-source papadimas/trogs-26-test-images \
  --prod-source-split test \
  --prod-name trogs26-test-ppm   # simplification; use trogs26-test for the frozen recipe
```

Re-running `--prod` with the same `--prod-name` reuses the cached decode for that slug.

The prod summary includes an `official_v2` block: the three official
TROGS-26 scores (`cer` on confident letters only, `all_cer` on all letters,
`opt_cer` masked) computed by the verbatim `contest_evaluation_v2.py`, plus
confidence-section coverage and a v1-vs-v2 row-grouping agreement check.
Re-running `--prod` on a finished run reuses the cached decode. To rescore a
finished run without touching the decode (no GPU, no frozen-artifact gate):

```bash
uvx modal run modal_app.py --prod-score-only --prod-name <slug>
```

To score locally over synced data:

```bash
python squeeze_runtime/official_scoring.py \
  --pred data/prod_runs/trogs26-test/transcripts \
  --gt   data/prod_inputs/<slug>/Annotations/Annotations
```

Note: transcript files are written without a trailing newline — the v2
reference is `'\n'.join(rows)`, so a trailing newline would score as one
insertion per squeeze.

Upload a completed production result:

```bash
uvx modal run modal_app.py --upload-prod-result-only --prod-name trogs26-test
```

`--upload-prod-result-only` writes to `<BUCKET_URI>/data/prod_runs/{slug}`, i.e. the namespace of
the checked-out branch (`ppm_unified/data/prod_runs/` on `main`, `data/prod_runs/` on
`frozen/three-order`), so use the slug that matches the branch.

## Notebook on Modal

`modal_notebook.py` runs the notebook itself on Modal against the prepared
`greek-squeezes-data` volume (no local GPU or artifact sync needed):

```bash
uvx modal run --detach modal_notebook.py::run_all   # headless Run All
uvx modal run --detach modal_notebook.py::jupyter   # interactive, via SSH tunnel
```

`run_all` executes every cell and persists the executed notebook plus every
regenerated report figure to the volume. Recover them locally and refresh the
repo's figures with:

```bash
uvx modal volume get greek-squeezes-data /notebook_run ./notebook_run
cp notebook_run/figs/*.png report/figs/
```

Both entrypoints bake `squeeze_runtime/` and `greek_squeezes.ipynb` from the
local repo at launch, so run them from a checkout that contains the version you
want executed.

## Report

The report source is `report/report.tex`; the rendered PDF is `report/report.pdf`.

```bash
cd report
lualatex report.tex
lualatex report.tex
```
