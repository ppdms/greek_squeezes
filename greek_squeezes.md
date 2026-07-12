---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Greek Squeezes Charpost

This notebook is the recipe for the Greek Squeezes OCR stack.
It expects artifacts under `data/` and validates each artifact before reusing it. If an
artifact is missing and `RUN_TRAINING_IF_MISSING = True`, it rebuilds it through
the same runtime modules used by the Modal runner.

The deployed system has two stages:

- `SqueezeTrOCR` dual-light TrOCR line recognizers. The all-train checkpoint is used for final
  deployment; five fold-excluded checkpoints provide clean OOF evidence for selector tuning.
- A character-posterior (`charpost`) correction layer. It trains tile1 dual-light character
  classifiers, caches per-position logits, proposes row candidates from visual lattices plus
  character LMs, and tunes one visual--PPM interpolation weight.

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

## 0. Setup

This cell defines the artifact roots, fixed hyperparameters, and deterministic dual-light
evaluation environment.


## Artifact bootstrap

Two supported environments, detected by one check:

* **Modal** (`modal_notebook.py::jupyter` / `::run_all`): the prepared volume is
  mounted at `/mnt/greek-squeezes-data` and the repo is baked at
  `$MODAL_REPO_ROOT`. If the volume is empty, the HF bucket is synced *into the
  volume*, so the download persists across sessions.
* **Local**: run from the repo root; artifacts live in `./data`.

Set `FORCE_REDOWNLOAD = True` to re-sync even if artifacts look present. On
Modal, `HF_TOKEN` is injected by the `huggingface-token` secret; running
locally, set the `HF_TOKEN` variable in the cell below (or export `HF_TOKEN`
in your shell, or run `hf auth login` first) -- don't commit a notebook with a
real token pasted into it.

```python
# --- Artifact bootstrap -------------------------------------------------
# Modal: prepared volume at /mnt/greek-squeezes-data, repo at $MODAL_REPO_ROOT.
# Local: repo root is the cwd, artifacts in ./data.
FORCE_REDOWNLOAD = False
HF_BUCKET_URI = 'hf://buckets/papadimas/greek_squeezes/ppm_unified'
# For local runs: set your token here instead of exporting HF_TOKEN (Modal
# injects HF_TOKEN via the huggingface-token secret). Don't commit a real one.
HF_TOKEN = ''

import os, sys, tempfile, warnings
from pathlib import Path
warnings.filterwarnings('ignore', message='.*IProgress not found.*')

_VOLUME = Path('/mnt/greek-squeezes-data')          # mounted by modal_notebook.py
_data_root = _VOLUME if _VOLUME.exists() else Path.cwd()
_data = _data_root / 'data'
_data.mkdir(parents=True, exist_ok=True)

_repo = Path(os.environ.get('MODAL_REPO_ROOT') or Path.cwd())
if not (_repo / 'squeeze_runtime' / '__init__.py').is_file():
    raise FileNotFoundError(f'squeeze_runtime/ not found under {_repo}; run from the repo root')
os.environ['REPO_ROOT'] = str(_repo)
os.environ['ARTIFACT_ROOT'] = str(_data_root)
print('repo root     ->', _repo)
print('artifact root ->', _data_root)


def _artifacts_present(root: Path) -> bool:
    sentinel_files = [
        root / 'splits' / 'oof_folds.json',
        root / 'line_recognizer' / 'all_train' / 'config.json',
    ]
    missing = [str(p.relative_to(root)) for p in sentinel_files if not p.is_file()]
    if missing:
        print('missing artifacts:', ', '.join(missing))
    return not missing


def _hf_sync_cli() -> str:
    # `hf sync` on hf://buckets/... URIs needs huggingface_hub>=1.5, but
    # transformers<5 (required for trocr's slow tokenizer) pins
    # huggingface_hub<1.0, so the CLI in *this* environment never gains the
    # `sync` subcommand. Use an isolated venv just for the CLI (same
    # workaround as modal_app.py).
    import subprocess
    hfcli_dir = Path(tempfile.gettempdir()) / 'greek_squeezes_hfcli_venv'
    hf_bin = hfcli_dir / 'bin' / 'hf'
    if not hf_bin.exists():
        print('installing isolated hf CLI (huggingface_hub>=1.5) ->', hfcli_dir)
        subprocess.run([sys.executable, '-m', 'venv', str(hfcli_dir)], check=True)
        subprocess.run([str(hfcli_dir / 'bin' / 'pip'), 'install', '-q',
                        'huggingface_hub[hf_xet]>=1.5'], check=True)
    return str(hf_bin)


if FORCE_REDOWNLOAD or not _artifacts_present(_data):
    import subprocess
    _tok = HF_TOKEN or os.environ.get('HF_TOKEN', '')
    if not _tok:
        raise RuntimeError(
            'No HF_TOKEN found. Set the HF_TOKEN variable above, export HF_TOKEN in '
            'your shell, run `hf auth login`, or attach the `huggingface-token` Modal secret.'
        )
    os.environ['HF_TOKEN'] = _tok  # the hf CLI subprocess and later cells (raw dataset) read it
    # Sync the bucket's data/ prefix straight into the local data/ dir -- NOT
    # the bucket root onto the repo root, which would overwrite this repo's
    # README.md with the bucket's.
    print('downloading HF bucket ->', _data, '(this is large)')
    subprocess.run([_hf_sync_cli(), 'sync', HF_BUCKET_URI.rstrip('/') + '/data', str(_data)], check=True)
    if not _artifacts_present(_data):
        raise RuntimeError('bucket sync completed but sentinel artifacts are still missing.')
else:
    print('caches already present at', _data_root, '- skipping download')
```

## Raw squeeze dataset

The HF bucket stores trained artifacts, not the source squeeze scans. This cell
fetches the public dataset `papadimas/trogs-greek-squeezes` (~7.4 GB) and lays
it out as `data/Images/*_Rotation{1,2}_300dpi.png` and
`data/Annotations/Annotations/*_letters.txt`, which the pipeline's dataset index
requires. It skips work if the images are already present and falls back to the
original Smith ScholarWorks zips if the HF mirror is unavailable.
Set `DOWNLOAD_DATASET = False` to skip.

```python
# --- Raw squeeze dataset provisioning -------------------------------------
import os, json, glob, tempfile, zipfile, shutil, importlib.util
from pathlib import Path

DOWNLOAD_DATASET = True
HF_DATASET = 'papadimas/trogs-greek-squeezes'

SCHOLARWORKS_FILES = {
    'Annotations.zip': 'https://scholarworks.smith.edu/context/dds_data/article/1017/type/native/viewcontent',
    'Images1.zip': 'https://scholarworks.smith.edu/cgi/viewcontent.cgi?filename=4&article=1017&context=dds_data&type=additional',
    'Images2.zip': 'https://scholarworks.smith.edu/cgi/viewcontent.cgi?filename=1&article=1017&context=dds_data&type=additional',
    'Images3.zip': 'https://scholarworks.smith.edu/cgi/viewcontent.cgi?filename=2&article=1017&context=dds_data&type=additional',
    'Images4.zip': 'https://scholarworks.smith.edu/cgi/viewcontent.cgi?filename=3&article=1017&context=dds_data&type=additional',
}

_DATA_ROOT = Path(os.environ['ARTIFACT_ROOT'])
_DATA = _DATA_ROOT / 'data'


def _raw_data_ready(root: Path) -> bool:
    images = glob.glob(str(root / 'Images' / '*.png'))
    letters = glob.glob(str(root / 'Annotations' / 'Annotations' / '*_letters.txt'))
    return len(images) >= 448 and bool(letters)


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return str(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def _prepare_raw_data(root: Path, token: str) -> None:
    img_dir = root / 'Images'
    ann_dir = root / 'Annotations'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    if _raw_data_ready(root):
        print('raw data ready ->', root)
        return
    try:
        from huggingface_hub import snapshot_download
        dl_dir = Path(tempfile.gettempdir()) / 'hf_trogs_dataset'
        if dl_dir.exists():
            shutil.rmtree(dl_dir, ignore_errors=True)
        print(f'downloading {HF_DATASET} (this is large) -> {dl_dir}')
        root_dl = Path(snapshot_download(
            repo_id=HF_DATASET, repo_type='dataset',
            allow_patterns=['data/**'],
            local_dir=str(dl_dir),
            token=token or None,
        ))
        ann_nested = ann_dir / 'Annotations'
        split_files = {
            'train': ann_dir / 'training_set.txt',
            'validation': ann_dir / 'validation_set.txt',
            'test': ann_dir / 'test_set.txt',
        }
        split_bases = {split: set() for split in split_files}
        n_img = 0
        for split in split_files:
            meta = root_dl / 'data' / split / 'metadata.jsonl'
            with meta.open(encoding='utf-8') as fh:
                for line in fh:
                    rec = json.loads(line)
                    split_bases[split].add(rec['base_id'])
                    _link_or_copy(root_dl / 'data' / split / rec['file_name'],
                                  img_dir / rec['file_name'])
                    ann_path = ann_nested / rec['annotation_file']
                    if not ann_path.exists():
                        ann_path.parent.mkdir(parents=True, exist_ok=True)
                        ann_path.write_text(rec['raw_annotation'].rstrip() + '\n',
                                            encoding='utf-8')
                    n_img += 1
                    if n_img % 50 == 0:
                        print(f'  staged {n_img} images ...')
        for split, path in split_files.items():
            path.write_text('\n'.join(sorted(split_bases[split])) + '\n', encoding='utf-8')
        readme = ann_dir / 'README.txt'
        if not readme.exists():
            readme.write_text(
                'Prepared from Hugging Face dataset papadimas/trogs-greek-squeezes.\n'
                'Original source: https://scholarworks.smith.edu/dds_data/18\n',
                encoding='utf-8')
        shutil.rmtree(dl_dir, ignore_errors=True)
        if _raw_data_ready(root):
            print(f'raw data ready from Hugging Face; images={len(glob.glob(str(img_dir / "*.png")))}')
            return
        print('Hugging Face dataset mirror was incomplete; using ScholarWorks')
    except Exception as exc:
        print(f'Hugging Face dataset preparation failed: {exc}; using ScholarWorks')

    import requests
    for name, url in SCHOLARWORKS_FILES.items():
        dst = Path(tempfile.gettempdir()) / name
        print(f'downloading {name}')
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with dst.open('wb') as fh:
                for chunk in response.iter_content(1 << 20):
                    fh.write(chunk)
        target = ann_dir if name == 'Annotations.zip' else img_dir
        with zipfile.ZipFile(dst) as archive:
            archive.extractall(target)
        dst.unlink()
        print(f'extracted {name}')
    if not _raw_data_ready(root):
        raise RuntimeError('raw data preparation failed; check HF_DATASET / ScholarWorks access')


if DOWNLOAD_DATASET:
    if importlib.util.find_spec('huggingface_hub') is None:
        import subprocess
        subprocess.run(['pip', 'install', '-q', 'huggingface_hub[hf_xet]'], check=True)
    _tok = os.environ.get('HF_TOKEN', '')
    _prepare_raw_data(_DATA, _tok)
else:
    print('DOWNLOAD_DATASET=False -> skipping raw squeeze dataset provisioning')
```

```python
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

REPO = Path(os.environ.get('REPO_ROOT', Path.cwd()))
RUNTIME_DIR = REPO / 'squeeze_runtime'
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))
DATA = Path(os.environ.get('ARTIFACT_ROOT', str(REPO))) / 'data'
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
CHARPOST_ORDERS = (12,)
CHARPOST_ORDER_ARG = ','.join(str(order) for order in CHARPOST_ORDERS)
IMAGE_CACHE_DIR = CHARPOST_DIR / 'image_cache'
RANKER_DIR = CHARPOST_DIR / 'oof_ranker'
CANDIDATES_JSON = RANKER_DIR / 'candidates.json'
RANKER_JSON = RANKER_DIR / 'ranker_scores.json'

RUN_TRAINING_IF_MISSING = True

RANKER_METHODS = 'visual_ppm_interpolation'
CHARPOST_RANKER_FEATURES = 'visual_sum,ppm_sum'

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
print('runtime    :', RUNTIME_DIR)
print('data root :', DATA)
```

## 1. Runtime Modules

Runtime code lives in `squeeze_runtime/` so the notebook only orchestrates and audits the
pipeline. The import cell reloads the modules, checks that the expected Python packages are
installed, and points every module's artifact paths at this tree through
`runtime_config.configure` -- the single place that wiring happens.

```python
import importlib
import importlib.util
import subprocess

# Runtime packages the notebook/pipeline need (import name -> pip package).
_REQUIRED_IMPORTS = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'PIL': 'pillow',
    'cv2': 'opencv-python-headless',
    'torch': 'torch',
    'transformers': 'transformers',
    'tokenizers': 'tokenizers',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'textdistance': 'textdistance',
    'accelerate': 'accelerate',
    'einops': 'einops',
    'timm': 'timm',
    'sentencepiece': 'sentencepiece',
    'requests': 'requests',
}
_missing = [(mod, pkg) for mod, pkg in _REQUIRED_IMPORTS.items()
            if importlib.util.find_spec(mod) is None]
if _missing:
    _pkgs = [pkg for _, pkg in _missing]
    print('installing missing runtime packages:', ', '.join(_pkgs))
    subprocess.run(['pip', 'install', '-q', *_pkgs], check=True)
    still = [mod for mod, _ in _missing if importlib.util.find_spec(mod) is None]
    if still:
        raise ImportError('Missing Python packages after install: ' + ', '.join(still))

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
RF = _load_runtime_module('report_figs')
RC = _load_runtime_module('runtime_config')

# Point every runtime module at this artifact tree (the one sanctioned place).
RC.configure(WORK, figs_dir=REPO / 'report' / 'figs')

print('runtime imported from', RUNTIME_DIR)
print('runtime modules:', ', '.join(['duallight', 'prep_cache', 'line_folds', 'rerank',
                                     'char_lattice', 'charpost_ranker', 'report_figs',
                                     'runtime_config']))
```

## 2. Pipeline Configuration

The active charpost recipe uses a single order-12 character count model scored by adaptive-order
PPM-C -- the one language mechanism, driving both candidate proposals and the selector's LM score --
the OOF-tuned interpolation `visual_sum + lambda * ppm_sum`, and the `ppm_interpolation_v2`
feature schema. Candidate generation uses the defaults
enforced by `ArtifactChecks`: one purely visual beam plus one PPM-C-weighted beam (proposal
weight `1.0`), beam 256, and char-top-k 8.

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
to tune the row selector, so validation/test rows are not used for selector selection.

```python
PIPELINE.ensure_oof_artifacts()
```

## 5. Candidate Table and One-Parameter Selector

The selector stage converts held-out fold lattices into row candidates and tunes the score
`visual_sum + lambda * ppm_sum` directly on corpus CER. Nested fold-heldout tuning selects
`lambda=0.84` in every outer fold; the final all-OOF search selects the same value. The complete
feature set is `visual_sum, ppm_sum`.

```python
PIPELINE.ensure_ranker()
```

## 6. Final All-Train Decode

After OOF selection is frozen, the notebook trains or validates the final all-train charpost
classifier, train-only character LMs, validation logits, test logits, and final decodes.
The same frozen artifacts are used by `modal_app.py --prod` for external datasets such as
`papadimas/trogs-26-test-images`.

```python
PIPELINE.ensure_decodes()
```

## 7. Summary

The summary prints artifact readiness, nested OOF selector quality, and any available
validation/test decode scores. The interpolation selector reaches OOF CER 0.049770
(2,672/53,687), validation CER 0.022929 (93/4,056), and test CER 0.022207 (100/4,503).
The rescored external `papadimas/trogs-26-test-images` production set scores 664/7,919
(CER 0.083849); official v2 `all_cer` is 0.0786.

```python
PIPELINE.summarize()
```

## 8. Mechanism Walkthrough and Report Figures

This section makes the charpost mechanism inspectable on real cached artifacts and regenerates
every figure used by `report/report.tex` into `report/figs/`. The figure builders live in
`squeeze_runtime/report_figs.py`.

`find_example_row` scans the frozen all-train test decode for a real row where the selector
overturns the visual-only argmax and lands on the ground truth (falling back to any
disagreement, then the first row). The same returned row is then reused for the printed
step-by-step narrative and for both lattice figures, so the report's prose and its figures
describe one concrete, reproducible example instead of three independently sampled ones.

```python

# RF was imported and pointed at this tree by runtime_config.configure (Section 1).
from IPython.display import Image as _Image, display as _display

example = RF.maybe(RF.find_example_row)
if example is not None:
    RF.mechanism_walkthrough(rec=example)
    graph_fig = RF.fig_lattice_graph(rec=example)
    _display(_Image(str(graph_fig)))
```

The next cell rebuilds the remaining artifact-backed charpost figures. Each builder validates
its inputs and is skipped (with a message) when the corresponding `data/` artifact has not been
synced or built yet:

- `charpost_lattice_example.png` — the same row as above, but as a full-alphabet lattice
  heatmap with the truth / visual-only / selector-selected paths and the top-scored candidates;
- `charpost_cer_ladder.png` — OOF CER of the visual-only argmax vs the tuned interpolation
  vs the ground-truth-selected candidate-pool oracle, from the frozen candidate table;
- `charpost_selector_tuning.png` — the all-OOF CER curve used to choose `lambda=0.84`;
- `charpost_confusion.png` — aggregate OOF tile1 classifier confusion over the 27-symbol
  charpost alphabet.

```python

if example is not None:
    _path = RF.maybe(RF.fig_lattice_example, rec=example)
    if _path:
        _display(_Image(str(_path)))

for _fig in (RF.fig_cer_ladder, RF.fig_selector_tuning, RF.fig_confusion):
    _path = RF.maybe(_fig)
    if _path:
        _display(_Image(str(_path)))
```

The last cell regenerates the line-model view figures from Section 3 of the report:
the dual-light packing steps (`duallight_steps.png`), the ViT patch grid over packed crops
(`patch_grid.png`), and decoder cross-attention for generated proxy tokens
(`cross_attention.png`). These need the raw squeeze images and — for cross-attention — the
all-train line checkpoint, so they are also cache-guarded.

```python

for _fig in (RF.fig_duallight_steps, RF.fig_patch_grid, RF.fig_cross_attention):
    _path = RF.maybe(_fig)
    if _path:
        _display(_Image(str(_path)))
```
