---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Greek Squeezes Charpost

This notebook is the active cache-guarded recipe for the final Greek Squeezes OCR stack.
It expects artifacts under `data/` and validates each artifact before reusing it. If an
artifact is missing and `RUN_TRAINING_IF_MISSING = True`, the notebook rebuilds it through
the same runtime modules used by the Modal runner.

The deployed system has two stages:

- `SqueezeTrOCR` dual-light TrOCR line recognizers. The all-train checkpoint is used for final
  deployment; five fold-excluded checkpoints provide clean OOF evidence for ranker fitting.
- A character-posterior (`charpost`) correction layer. It trains tile1 dual-light character
  classifiers, caches per-position logits, proposes row candidates from visual lattices plus
  character LMs, and fits a ridge row ranker.

Dual-light means the two raking-light scans are treated as one observation. Each paired crop is
phase-registered and packed into TrOCR's normal RGB input:

- `R = Rotation1`
- `G = phase-registered Rotation2`
- `B = min(R, G)`, a dark-stroke union channel

Training uses channel swap and mono dropout for robustness. Evaluation is deterministic:
`DUAL_LIGHT=1`, `DUAL_MODE=min`, phase registration on, channel swap off, mono dropout off.

Active artifacts:

- `line_recognizer/all_train`
- `line_recognizer/oof_fold0` ... `line_recognizer/oof_fold4`
- `splits/oof_folds.json`
- `character_posterior/image_cache/`
- `character_posterior/oof_f0` ... `character_posterior/oof_f4`
- `character_posterior/oof_ranker/candidates.json`
- `character_posterior/oof_ranker/ranker_scores.json`
- `character_posterior/all_train`
- `character_posterior/all_train__val_decode.json`
- `character_posterior/all_train__test_decode.json`

Clean OOF is required for claims: the held-out fold is excluded from the fold TrOCR checkpoint,
the charpost classifier, and the fold character LMs. Validation/test/prod decodes are report-only
after this recipe is frozen.

## 0. Setup

This cell defines the artifact roots, fixed hyperparameters, and deterministic dual-light
evaluation environment.

```python
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

REPO = Path.cwd()
RUNTIME_DIR = REPO / 'squeeze_runtime'
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))
DATA = REPO / 'data'
WORK = DATA
LINE_DIR = DATA / 'line_recognizer'
CHARPOST_DIR = DATA / 'character_posterior'
SPLITS_DIR = DATA / 'splits'

FOLDS = tuple(range(5))
TILE = 4
FOLDS_JSON = SPLITS_DIR / 'oof_folds.json'

LINE_ALL_TRAIN = LINE_DIR / 'all_train'
LINE_OOF_FOLD = {fold: LINE_DIR / f'oof_fold{fold}' for fold in FOLDS}
CHARPOST_ALL_TRAIN_TAG = 'all_train'
CHARPOST_OOF_PREFIX = 'oof'
CHARPOST_ORDERS = (6, 10, 12)
CHARPOST_ORDER_ARG = ','.join(str(order) for order in CHARPOST_ORDERS)
IMAGE_CACHE_DIR = CHARPOST_DIR / 'image_cache'
RANKER_DIR = CHARPOST_DIR / 'oof_ranker'
CANDIDATES_JSON = RANKER_DIR / 'candidates.json'
RANKER_JSON = RANKER_DIR / 'ranker_scores.json'

RUN_TRAINING_IF_MISSING = True

RANKER_METHODS = 'ridge'
CHARPOST_RANKER_FEATURES = 'visual_sum,mean_char_logprob,length,length_delta,lm6_sum,lm10_sum,lm12_sum'

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TORCH_SHARE_STRATEGY', 'file_system')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

DUAL_EVAL_ENV = {
    'DUAL_LIGHT': '1',
    'DUAL_MODE': 'min',
    'DUAL_DIVERGENT': 'both',
    'DUAL_REGISTER': '1',
    'DUAL_REGISTER_METHOD': 'phase',
    'DUAL_REGISTER_MAX_SHIFT': '0.15',
    'DUAL_CHANNEL_SWAP': '0',
    'DUAL_MONO_DROPOUT': '0',
}

def apply_dual_eval_env() -> None:
    os.environ.update(DUAL_EVAL_ENV)

apply_dual_eval_env()

for path in (DATA, LINE_DIR, CHARPOST_DIR, SPLITS_DIR, RANKER_DIR):
    path.mkdir(parents=True, exist_ok=True)

print('repo       :', REPO)
print('data root :', DATA)
```

## 1. Runtime Modules

Runtime code lives in `squeeze_runtime/` so the notebook only orchestrates and audits the
pipeline. The import cell reloads the modules, points their mutable artifact roots at this
repo's `data/` directory, and checks that the expected Python packages are installed.

```python

import importlib
import importlib.util

_REQUIRED_IMPORTS = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'PIL': 'pillow',
    'cv2': 'opencv-python-headless',
    'torch': 'torch',
    'transformers': 'transformers',
    'tokenizers': 'tokenizers',
    'sklearn': 'scikit-learn',
}
_missing = [pkg for mod, pkg in _REQUIRED_IMPORTS.items() if importlib.util.find_spec(mod) is None]
if _missing:
    raise ImportError('Missing Python packages: ' + ', '.join(_missing))

if not RUNTIME_DIR.exists():
    raise FileNotFoundError(f'missing runtime package: {RUNTIME_DIR}')
_runtime_dir = str(RUNTIME_DIR)
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)


def _load_runtime_module(name: str) -> types.ModuleType:
    module = importlib.import_module(name)
    return importlib.reload(module)

# Dependency order matters for imports inside the runtime modules.
DL = _load_runtime_module('duallight')
PC = _load_runtime_module('prep_cache')
LF = _load_runtime_module('line_folds')
R = _load_runtime_module('rerank')
CL = _load_runtime_module('char_lattice')
CR = _load_runtime_module('charpost_ranker')

# Point imported modules at the local artifact tree.
R.WORK = str(WORK.resolve())

CL.WORK = WORK
CL.CHARPOST_DIR = CHARPOST_DIR
CL.DEFAULT_FOLDS = FOLDS_JSON

CR.WORK = WORK
CR.CHARPOST_DIR = CHARPOST_DIR
CR.OOF_DIR = RANKER_DIR

print('runtime imported from', RUNTIME_DIR)
print('runtime modules:', ', '.join(['duallight', 'prep_cache', 'line_folds', 'rerank', 'char_lattice', 'charpost_ranker']))
```

## 2. Pipeline Configuration

The active charpost recipe uses orders 6, 10, and 12 character LMs, a ridge scorer, and the
`lattice_length_delta_v2` feature schema. Candidate generation uses the defaults enforced by
`ArtifactChecks`: proposal weights `0,0.25,0.5,0.75,1.0`, beam 256, and char-top-k 8.

```python

from artifact_checks import ArtifactChecks, PipelineConfig
from charpost_pipeline import CharpostPipeline

PIPELINE_CONFIG = PipelineConfig(
    repo=REPO,
    charpost_dir=CHARPOST_DIR,
    folds=FOLDS,
    tile=TILE,
    folds_json=FOLDS_JSON,
    line_all_train=LINE_ALL_TRAIN,
    line_oof_fold=LINE_OOF_FOLD,
    charpost_all_train_tag=CHARPOST_ALL_TRAIN_TAG,
    charpost_oof_prefix=CHARPOST_OOF_PREFIX,
    charpost_orders=CHARPOST_ORDERS,
    image_cache_dir=IMAGE_CACHE_DIR,
    ranker_dir=RANKER_DIR,
    candidates_json=CANDIDATES_JSON,
    ranker_json=RANKER_JSON,
    ranker_methods=RANKER_METHODS,
    charpost_ranker_features=CHARPOST_RANKER_FEATURES,
    rank_fold_seed=LF.RANK_FOLD_SEED,
)
ARTIFACTS = ArtifactChecks(PIPELINE_CONFIG)
PIPELINE = CharpostPipeline(
    cfg=PIPELINE_CONFIG,
    artifacts=ARTIFACTS,
    rerank=R,
    line_folds=LF,
    char_lattice=CL,
    charpost_ranker=CR,
    run_training_if_missing=RUN_TRAINING_IF_MISSING,
)

status = ARTIFACTS.status
require_artifact_layout = ARTIFACTS.require_artifact_layout
require_artifact_layout()
print(json.dumps(status(), indent=2, sort_keys=True))
```

## 3. Line Recognizer Checkpoints

The line recognizers are prerequisites for charpost. The final all-train checkpoint is the
deployable visual model; the five OOF checkpoints initialize fold-clean character-posterior
models. The expected training recipe is dual-light tile4 `microsoft/trocr-large-printed`,
real crops only, 6000 optimizer steps, batch 96, phase registration, channel swap during
training, and mono dropout 0.15.

```python

# Verify the cached line recognizers.
PIPELINE.ensure_line_models()
```

## 4. Fold-Clean OOF Charpost Artifacts

This stage builds or validates the dual-light image cache, five fold-excluded charpost
classifiers, fold logits, and fold-excluded character LMs. These are the only artifacts used
to fit the row ranker, so validation/test rows are not used for ranker selection.

```python

PIPELINE.ensure_oof_artifacts()
```

## 5. Candidate Table and Ridge Ranker

The ranker stage converts held-out fold lattices into row candidates, computes visual and LM
features, and fits the final ridge scorer. The selected feature set is:
`visual_sum, mean_char_logprob, length, length_delta, lm6_sum, lm10_sum, lm12_sum`.

```python

PIPELINE.ensure_ranker()
```

## 6. Final All-Train Decode

After OOF selection is frozen, the notebook trains or validates the final all-train charpost
classifier, train-only character LMs, validation logits, test logits, and final decodes.
The same frozen artifacts are used by `modal.py --prod` for external datasets such as
`papadimas/trogs-26-test-images`.

```python

PIPELINE.ensure_decodes()
```

## 7. Summary

The summary prints artifact readiness, OOF ranker quality, and any available validation/test
decode scores. The most recent external production run reported 50 squeezes, 100 images,
605 scored rows, 7,919 characters, 651 errors, and CER 0.0822073494 on
`papadimas/trogs-26-test-images`.

```python

PIPELINE.summarize()
```
