#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import argparse
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import modal

APP_DIR = Path('/root/app')
VOL_DIR = Path('/root/work')
DATA_DIR = VOL_DIR / 'data'
HF_CACHE = VOL_DIR / 'hf_cache'
SMOKE_DIR = DATA_DIR / '_smoke'
BUCKET_URI = 'hf://buckets/papadimas/greek_squeezes'
HF_DATASET = 'papadimas/trogs-greek-squeezes'
PROD_DEFAULT_SOURCE = 'papadimas/trogs-26-test-images'
GPU_DEFAULT = 'A100'
FOLDS = tuple(range(5))
ORDERS = (6, 10, 12)
PROD_OUTPUT_SPLIT = 'prod'
PROD_INPUT_ROOT = DATA_DIR / 'prod_inputs'
PROD_SOURCE_CACHE = VOL_DIR / 'prod_sources'
PROD_OUTPUT_ROOT = DATA_DIR / 'prod_outputs'
PROD_STAGE_VERSION = 2
SMOKE_FOLDS = (0, 1)
SMOKE_ORDER = 6
SMOKE_ROWS = 2
LINE_BASE_MODEL = 'microsoft/trocr-large-printed'
LINE_PROCESSOR_BASE = 'microsoft/trocr-base-printed'
LINE_TILE = 4
LINE_BATCH = 96
LINE_EVAL_BATCH = 16
LINE_GRAD_ACCUM = 1
LINE_MAX_STEPS = 6000
LINE_EVAL_STEPS = 300
LINE_EVAL_CAP = 400
LINE_SAVE_TOTAL_LIMIT = 5
LINE_LR = 2e-5
LINE_WEIGHT_DECAY = 0.0
LINE_DATALOADER_WORKERS = 6
LINE_DATALOADER_PREFETCH = 4
LOCK_DIR = VOL_DIR / '_locks'
LOCK_HEARTBEAT_SECONDS = 120
LOCK_STALE_SECONDS = 26 * 60 * 60
LOCK_POLL_SECONDS = 20
_VOL_COMMIT_LOCK = threading.Lock()

_ART = 'https://scholarworks.smith.edu/cgi/viewcontent.cgi?filename={}&article=1017&context=dds_data&type=additional'
SCHOLARWORKS_FILES = {
    'Annotations.zip': 'https://scholarworks.smith.edu/context/dds_data/article/1017/type/native/viewcontent',
    'Images1.zip': _ART.format('4'),
    'Images2.zip': _ART.format('1'),
    'Images3.zip': _ART.format('2'),
    'Images4.zip': _ART.format('3'),
}

app = modal.App('greek-squeezes-charpost')
vol = modal.Volume.from_name('greek-squeezes-data', create_if_missing=True)
hf_secret = modal.Secret.from_name('huggingface-token')
JOB_RETRIES = modal.Retries(max_retries=2, initial_delay=15.0, backoff_coefficient=2.0, max_delay=60.0)
ORCHESTRATOR_RETRIES = modal.Retries(max_retries=2, initial_delay=60.0, backoff_coefficient=1.0, max_delay=60.0)

image = (
    modal.Image.debian_slim(
        python_version='3.12',
        force_build=os.environ.get('MODAL_FORCE_BUILD', '').lower() in ('1', 'true', 'yes'),
    )
    .apt_install('libglib2.0-0')
    .pip_install(
        'torch',
        'transformers>=4.41,<5',  # v5 cannot load trocr-large-printed's slow tokenizer
        'accelerate>=0.30',
        'timm>=0.9',
        'einops>=0.7',
        'opencv-python-headless>=4.8',
        'pillow>=10.0',
        'numpy<2.2',
        'pandas>=2.0',
        'scikit-learn>=1.3',
        'matplotlib>=3.7',
        'textdistance>=4.5',
        'requests>=2.31',
        'sentencepiece>=0.1',
        'huggingface_hub[hf_xet]>=0.34,<1.0',  # transformers 4.x pins hub<1.0
    )
    # `hf sync` on hf://buckets/... URIs needs huggingface_hub 1.x, which conflicts
    # with transformers 4.x -- so the CLI lives in its own venv and shadows the 0.x `hf`.
    .run_commands(
        'python -m venv /opt/hfcli'
        " && /opt/hfcli/bin/pip install 'huggingface_hub[cli,hf_xet]>=1.5'"
        ' && ln -sf /opt/hfcli/bin/hf /usr/local/bin/hf',
    )
    .add_local_dir('squeeze_runtime', str(APP_DIR / 'squeeze_runtime'))
)


def _set_env() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TORCH_SHARE_STRATEGY', 'file_system')
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('HF_HOME', str(HF_CACHE))
    os.environ.update({
        'DUAL_LIGHT': '1',
        'DUAL_MODE': 'min',
        'DUAL_DIVERGENT': 'both',
        'DUAL_REGISTER': '1',
        'DUAL_REGISTER_METHOD': 'phase',
        'DUAL_REGISTER_MAX_SHIFT': '0.15',
        'DUAL_CHANNEL_SWAP': '0',
        'DUAL_MONO_DROPOUT': '0',
    })


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _commit_volume(label: str) -> None:
    with _VOL_COMMIT_LOCK:
        vol.commit()
    print(f'volume committed: {label}', flush=True)


def _lock_slug(name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)
    return safe[:180] or 'lock'


def _lock_path(name: str) -> Path:
    return LOCK_DIR / _lock_slug(name)


def _write_lock_file(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _lock_heartbeat(lock_dir: Path, stop: threading.Event) -> None:
    heartbeat = lock_dir / 'heartbeat.json'
    while not stop.wait(LOCK_HEARTBEAT_SECONDS):
        try:
            _write_lock_file(heartbeat, {'pid': os.getpid(), 'time': time.time()})
            _commit_volume(f'lock heartbeat {lock_dir.name}')
        except Exception as exc:
            print(f'warning: lock heartbeat failed for {lock_dir}: {exc}', flush=True)


@contextmanager
def _volume_lock(name: str):
    lock_dir = LOCK_DIR / _lock_slug(name)
    owner = {
        'name': name,
        'pid': os.getpid(),
        'created': time.time(),
        'host': os.uname().nodename if hasattr(os, 'uname') else '',
    }
    wait_logged = 0.0
    while True:
        try:
            LOCK_DIR.mkdir(parents=True, exist_ok=True)
            lock_dir.mkdir()
            _write_lock_file(lock_dir / 'owner.json', owner)
            _write_lock_file(lock_dir / 'heartbeat.json', {'pid': os.getpid(), 'time': time.time()})
            _commit_volume(f'lock acquire {lock_dir.name}')
            stop = threading.Event()
            heartbeat = threading.Thread(target=_lock_heartbeat, args=(lock_dir, stop), daemon=True)
            heartbeat.start()
            print(f'lock acquired: {name}', flush=True)
            try:
                yield
            finally:
                stop.set()
                heartbeat.join(timeout=5)
                shutil.rmtree(lock_dir, ignore_errors=True)
                try:
                    _commit_volume(f'lock release {lock_dir.name}')
                except Exception as exc:
                    print(f'warning: lock release commit failed for {name}: {exc}', flush=True)
                print(f'lock released: {name}', flush=True)
            return
        except FileExistsError:
            try:
                vol.reload()
            except Exception as exc:
                print(f'warning: vol.reload while waiting for lock failed: {exc}', flush=True)
            heartbeat = lock_dir / 'heartbeat.json'
            try:
                age = time.time() - heartbeat.stat().st_mtime
            except OSError:
                age = LOCK_STALE_SECONDS + 1
            if age > LOCK_STALE_SECONDS:
                print(f'breaking stale lock: {name} age={age:.0f}s', flush=True)
                shutil.rmtree(lock_dir, ignore_errors=True)
                try:
                    _commit_volume(f'lock stale remove {lock_dir.name}')
                except Exception as exc:
                    print(f'warning: stale lock removal commit failed for {name}: {exc}', flush=True)
                continue
            now = time.time()
            if now - wait_logged > 120:
                print(f'waiting for lock: {name} age={age:.0f}s', flush=True)
                wait_logged = now
            time.sleep(LOCK_POLL_SECONDS)


def _run_locked(name: str, fn):
    with _volume_lock(name):
        return fn()


def _wait_for_locks_to_clear(names: list[str], *, label: str) -> None:
    names = list(dict.fromkeys(names))
    if not names:
        return
    last_log = 0.0
    while True:
        try:
            vol.reload()
        except Exception as exc:
            print(f'warning: vol.reload while waiting for {label} locks failed: {exc}', flush=True)
        present = [name for name in names if _lock_path(name).exists()]
        if not present:
            return
        now = time.time()
        if now - last_log > 120:
            print(f'waiting for existing {label} locks: {", ".join(present)}', flush=True)
            last_log = now
        time.sleep(LOCK_POLL_SECONDS)


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return str(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def _copy_tree_with_links(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=lambda s, d: _link_or_copy(Path(s), Path(d)))


def _stage_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print('stage skipped missing ->', src, flush=True)
        return
    print('stage copy ->', src, '=>', dst, flush=True)
    _copy_tree_with_links(src, dst)


def _prod_slug(value: str) -> str:
    value = value.strip().removeprefix('hf://datasets/').removeprefix('https://huggingface.co/datasets/')
    value = value.strip('/')
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    return safe.strip('._-')[:140] or 'input'


def _prod_repo_id(source: str) -> str:
    source = source.strip()
    if source.startswith('hf://datasets/'):
        source = source.removeprefix('hf://datasets/')
    if source.startswith('https://huggingface.co/datasets/'):
        source = source.removeprefix('https://huggingface.co/datasets/')
    return source.strip('/')


def _prod_safe_base(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._+:-]+', '_', value.strip().replace('/', '_'))
    return safe.strip('._-')[:180] or 'squeeze'


def _prod_source_root(source: str, revision: str = '') -> Path:
    src = Path(source).expanduser()
    if src.exists():
        return src
    from huggingface_hub import snapshot_download

    repo_id = _prod_repo_id(source)
    local_dir = PROD_SOURCE_CACHE / _prod_slug(repo_id + (f'_{revision}' if revision else ''))
    return Path(snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        revision=revision or None,
        local_dir=str(local_dir),
        allow_patterns=['data/**', 'annotations/**', '*.zip', '*.txt', '*.py', 'README*'],
    ))


def _prod_metadata_path(root: Path, source_split: str) -> Path:
    split = source_split.strip()
    if split and split.lower() != 'auto':
        path = root / 'data' / split / 'metadata.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'prod source split metadata missing: {path}')
        return path
    candidates = sorted((root / 'data').glob('*/metadata.jsonl'))
    if not candidates:
        raise FileNotFoundError(f'no imagefolder metadata.jsonl found under {root / "data"}')
    preferred = [path for path in candidates if path.parent.name == 'test']
    return preferred[0] if preferred else candidates[0]


def _prod_rotation_from_name(name: str) -> str:
    match = re.search(r'_(Rotation[12])_300dpi', Path(name).stem)
    if not match:
        raise ValueError(f'could not infer Rotation1/Rotation2 from {name}')
    return match.group(1)


def _prod_base_from_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r'_Rotation[12]_300dpi$', '', stem)
    return stem.replace('_Merged', '')


def _prod_annotation_lookup(root: Path) -> dict[str, str]:
    import zipfile

    out: dict[str, str] = {}
    ann_dir = root / 'annotations'
    if ann_dir.exists():
        for path in sorted(ann_dir.glob('*_letters.txt')):
            out[path.name] = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    for path in sorted(root.glob('*Annotations*.zip')):
        try:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith('_letters.txt'):
                        out[Path(name).name] = zf.read(name).decode('utf-8-sig').replace('\r\n', '\n')
        except zipfile.BadZipFile:
            continue
    return out


def _prod_record_image_path(root: Path, metadata_path: Path, rec: dict[str, object]) -> Path:
    file_name = str(rec.get('file_name') or rec.get('image') or '')
    if not file_name:
        raise ValueError('prod metadata row is missing file_name')
    candidates = [
        metadata_path.parent / file_name,
        root / file_name,
        root / 'data' / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'prod image missing for {file_name}; tried {candidates}')


def _prod_record_annotation(rec: dict[str, object], annotations: dict[str, str], image_path: Path) -> str:
    raw = rec.get('raw_annotation')
    if isinstance(raw, str) and raw.strip():
        return raw.replace('\r\n', '\n')
    names = []
    if rec.get('annotation_file'):
        names.append(str(rec['annotation_file']))
    names.append(f'{image_path.stem}_letters.txt')
    for name in names:
        raw = annotations.get(Path(name).name)
        if raw:
            return raw
    raise FileNotFoundError(f'prod annotation missing for {image_path.name}; tried {names}')


def _prod_write_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if src.suffix.lower() == '.png':
        _link_or_copy(src, dst)
        return
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        im.save(dst)


def _stage_prod_dataset(
    *,
    source: str,
    source_split: str,
    name: str,
    revision: str,
    force: bool,
) -> dict[str, object]:
    source_root = _prod_source_root(source, revision)
    metadata_path = _prod_metadata_path(source_root, source_split)
    actual_source_split = metadata_path.parent.name
    slug = _prod_slug(name or f'{_prod_repo_id(source)}_{actual_source_split}')
    input_root = PROD_INPUT_ROOT / slug
    manifest_path = input_root / 'prod_manifest.json'
    requested = {
        'stage_version': PROD_STAGE_VERSION,
        'source': source,
        'source_root': str(source_root),
        'source_split': actual_source_split,
        'revision': revision,
        'slug': slug,
    }
    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            if all(existing.get(k) == v for k, v in requested.items()):
                print('prod input ready ->', input_root, flush=True)
                return {**existing, '_rebuilt': False}
        except Exception:
            pass
    if input_root.exists():
        shutil.rmtree(input_root)
    img_dir = input_root / 'Images'
    ann_outer = input_root / 'Annotations'
    ann_dir = ann_outer / 'Annotations'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    annotations = _prod_annotation_lookup(source_root)
    records: list[dict[str, object]] = []
    for line_no, line in enumerate(metadata_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        image_path = _prod_record_image_path(source_root, metadata_path, rec)
        file_name = str(rec.get('file_name') or image_path.name)
        rotation = str(rec.get('rotation') or _prod_rotation_from_name(file_name))
        squeeze = str(
            rec.get('squeeze_id')
            or rec.get('base_id')
            or rec.get('source_base_id')
            or _prod_base_from_name(file_name)
        ).replace('_Merged', '')
        if rotation not in {'Rotation1', 'Rotation2'}:
            raise ValueError(f'bad rotation in prod metadata row {line_no}: {rotation}')
        records.append({
            'squeeze_id': _prod_safe_base(squeeze),
            'rotation': rotation,
            'image_path': str(image_path),
            'annotation': _prod_record_annotation(rec, annotations, image_path),
            'source_file_name': file_name,
        })
    grouped: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        grouped.setdefault(str(rec['squeeze_id']), []).append(rec)
    if not grouped:
        raise RuntimeError(f'prod source has no metadata records: {metadata_path}')

    bases: list[str] = []
    image_count = 0
    view_map: list[dict[str, str]] = []
    for base, views in sorted(grouped.items()):
        by_rotation: dict[str, list[dict[str, object]]] = {'Rotation1': [], 'Rotation2': []}
        for rec in sorted(views, key=lambda item: str(item.get('source_file_name', ''))):
            by_rotation[str(rec['rotation'])].append(rec)
        selected: list[dict[str, object]] = []
        if by_rotation['Rotation1']:
            selected.append(by_rotation['Rotation1'].pop(0))
        elif by_rotation['Rotation2']:
            selected.append(by_rotation['Rotation2'].pop(0))
        if by_rotation['Rotation2']:
            selected.append(by_rotation['Rotation2'].pop(0))
        elif by_rotation['Rotation1']:
            selected.append(by_rotation['Rotation1'].pop(0))
        if len(selected) == 1:
            # The frozen classifier expects a two-view dual_pack input. For
            # single-view prod sources, reuse the one view as both channels.
            selected.append(selected[0])
        canonical = list(enumerate(selected[:2], start=1))
        bases.append(base)
        for rot_num, rec in canonical[:2]:
            image_src = Path(str(rec['image_path']))
            image_dst = img_dir / f'{base}_Rotation{rot_num}_300dpi.png'
            ann_dst = ann_dir / f'{base}_Rotation{rot_num}_300dpi_letters.txt'
            _prod_write_png(image_src, image_dst)
            ann_dst.write_text(str(rec['annotation']).rstrip() + '\n', encoding='utf-8')
            image_count += 1
            view_map.append({
                'base': base,
                'canonical_rotation': f'Rotation{rot_num}',
                'source_file_name': str(rec['source_file_name']),
            })

    (ann_outer / 'training_set.txt').write_text('', encoding='utf-8')
    (ann_outer / 'validation_set.txt').write_text('', encoding='utf-8')
    (ann_outer / 'test_set.txt').write_text('\n'.join(bases) + '\n', encoding='utf-8')
    manifest = {
        **requested,
        'input_root': str(input_root),
        'metadata_path': str(metadata_path),
        'squeezes': len(bases),
        'images': image_count,
        'bases': bases,
        'view_map': view_map,
        'internal_split': 'test',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print(f'prod staged {image_count} images for {len(bases)} squeezes -> {input_root}', flush=True)
    return {**manifest, '_rebuilt': True}


@contextmanager
def _prod_input_context(pipeline, input_root: Path):
    old_env = os.environ.get('SQUEEZE_WORK_DIR')
    old_r_work = pipeline.R.WORK
    old_cl_work = pipeline.CL.WORK
    old_cl_folds = pipeline.CL.DEFAULT_FOLDS
    root = input_root.resolve()
    os.environ['SQUEEZE_WORK_DIR'] = str(root)
    pipeline.R.WORK = str(root)
    pipeline.CL.WORK = root
    pipeline.CL.DEFAULT_FOLDS = root / 'splits' / 'oof_folds.json'
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop('SQUEEZE_WORK_DIR', None)
        else:
            os.environ['SQUEEZE_WORK_DIR'] = old_env
        pipeline.R.WORK = old_r_work
        pipeline.CL.WORK = old_cl_work
        pipeline.CL.DEFAULT_FOLDS = old_cl_folds


def _require_prod_frozen_artifacts(pipeline) -> None:
    model_dir = pipeline.cfg.charpost_dir / pipeline.cfg.charpost_all_train_tag
    missing: list[str] = []
    if not pipeline.artifacts.checkpoint_ready(pipeline.cfg.line_all_train):
        missing.append(f'line checkpoint: {pipeline.cfg.line_all_train}')
    if not pipeline.artifacts.charpost_classifier_ready(model_dir, tag=pipeline.cfg.charpost_all_train_tag, init_ckpt=pipeline.cfg.line_all_train):
        missing.append(f'charpost classifier: {model_dir}')
    if not pipeline.artifacts.ranker_ready(pipeline.cfg.ranker_json):
        missing.append(f'ranker: {pipeline.cfg.ranker_json}')
    for order in pipeline.cfg.charpost_orders:
        lm_path = pipeline.cfg.charpost_dir / f'train_lm_order{order}_train' / 'lm.json'
        if not pipeline.artifacts.final_charpost_lm_ready(lm_path, order):
            missing.append(f'order-{order} LM: {lm_path}')
    if missing:
        raise FileNotFoundError('prod frozen artifacts are missing; run/sync the full pipeline first:\n  ' + '\n  '.join(missing))


def _prod_logits_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            required = {'logits', 'labels', 'bases', 'row_idx', 'chunk_idx', 'row_text', 'alphabet'}
            return required <= set(data.files) and int(data['logits'].shape[0]) > 0
    except Exception:
        return False


def _prod_decode_matches(path: Path, *, tag: str, split: str, shard: int | None = None, shards: int | None = None) -> bool:
    data = _read_json(path)
    if not isinstance(data, dict):
        return False
    if data.get('tag') != tag or data.get('split') != split:
        return False
    if data.get('orders') != list(ORDERS):
        return False
    if data.get('proposal_weights') != [0.0, 0.25, 0.5, 0.75, 1.0]:
        return False
    if data.get('beam') != 256 or data.get('char_topk') != 8:
        return False
    row_shard_id = data.get('row_shard_id')
    row_shard_count = data.get('row_shard_count')
    if shard is not None and int(row_shard_id if row_shard_id is not None else -1) != shard:
        return False
    if shards is not None and int(row_shard_count if row_shard_count is not None else -1) != shards:
        return False
    return bool(data.get('predictions'))


def _write_prod_transcripts(decode_path: Path, out_dir: Path) -> Path:
    data = _read_json(decode_path)
    if not isinstance(data, dict):
        raise RuntimeError(f'cannot write transcripts; bad decode JSON: {decode_path}')
    transcripts = data.get('transcripts') or []
    pred_dir = out_dir / 'transcripts'
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rec in transcripts:
        base = _prod_safe_base(str(rec.get('base') or 'base'))
        text = str(rec.get('text') or '').rstrip() + '\n'
        (pred_dir / f'{base}_transcript.txt').write_text(text, encoding='utf-8')
    return pred_dir


def _prod_summary(*, source: str, manifest: dict[str, object], tag: str, split: str, logits: Path, decode: Path, transcripts_dir: Path) -> dict[str, object]:
    data = _read_json(decode)
    scored = data.get('scored') if isinstance(data, dict) else None
    return {
        'source': source,
        'slug': manifest.get('slug'),
        'input_root': manifest.get('input_root'),
        'squeezes': manifest.get('squeezes'),
        'images': manifest.get('images'),
        'tag': tag,
        'split': split,
        'logits': str(logits),
        'decode': str(decode),
        'transcripts_dir': str(transcripts_dir),
        'scored': scored,
        'predictions': len(data.get('predictions') or []) if isinstance(data, dict) else 0,
        'transcripts': len(data.get('transcripts') or []) if isinstance(data, dict) else 0,
    }


def _prod_result_paths(slug: str) -> dict[str, Path]:
    tag = f'prod_{slug}'
    return {
        'summary': PROD_OUTPUT_ROOT / slug / 'summary.json',
        'transcripts': PROD_OUTPUT_ROOT / slug / 'transcripts',
        'manifest': PROD_INPUT_ROOT / slug / 'prod_manifest.json',
        'decode': DATA_DIR / 'character_posterior' / f'{tag}__{PROD_OUTPUT_SPLIT}_decode.json',
        'logits': DATA_DIR / 'character_posterior' / f'{tag}__{PROD_OUTPUT_SPLIT}_logits.npz',
    }


def _stage_prod_result_for_upload(slug: str, *, include_logits: bool) -> Path:
    slug = _prod_slug(slug)
    paths = _prod_result_paths(slug)
    missing = [
        str(path)
        for key, path in paths.items()
        if key != 'logits' and not path.exists()
    ]
    if include_logits and not paths['logits'].exists():
        missing.append(str(paths['logits']))
    if missing:
        raise FileNotFoundError('prod result files missing before upload:\n  ' + '\n  '.join(missing))

    stage = VOL_DIR / '_prod_result_upload_stage' / slug
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths['summary'], stage / 'summary.json')
    shutil.copy2(paths['manifest'], stage / 'prod_manifest.json')
    shutil.copy2(paths['decode'], stage / 'decode.json')
    if include_logits:
        shutil.copy2(paths['logits'], stage / 'logits.npz')
    _stage_tree(paths['transcripts'], stage / 'transcripts')
    (stage / 'README.md').write_text(
        f'# Prod Run: {slug}\n\n'
        'Frozen TROGS Greek squeezes pipeline output.\n\n'
        '- `summary.json`: scored run summary including CER when ground truth is available.\n'
        '- `decode.json`: row predictions, transcripts, and decode metadata.\n'
        '- `transcripts/`: official submission-style per-squeeze transcript files.\n'
        '- `prod_manifest.json`: source dataset normalization manifest.\n'
        + ('- `logits.npz`: frozen charpost classifier logits used by the decoder.\n' if include_logits else ''),
        encoding='utf-8',
    )
    return stage


def _clear_smoke_path(path: Path) -> None:
    if not path.exists():
        return
    smoke_root = SMOKE_DIR.resolve()
    target = path.resolve()
    if target != smoke_root and smoke_root not in target.parents:
        raise RuntimeError(f'refusing to delete non-smoke path: {path}')
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _unlink_smoke_charpost_outputs() -> None:
    charpost_dir = DATA_DIR / 'character_posterior'
    if not charpost_dir.exists():
        return
    for pattern in ('smoke_*__*_logits.npz', 'smoke_*__*_decode.json'):
        for path in charpost_dir.glob(pattern):
            path.unlink()


def _cleanup_smoke_artifacts() -> None:
    _clear_smoke_path(SMOKE_DIR)
    _unlink_smoke_charpost_outputs()


def _smoke_oof_lm_template() -> str:
    return str(SMOKE_DIR / 'oof_ranker' / 'train_lm_order{order}_exclude_f{fold}' / 'lm.json')


def _smoke_final_lm_template() -> str:
    return str(SMOKE_DIR / 'final_lm_order{order}_train' / 'lm.json')


def _sync_from_bucket(bucket_uri: str) -> None:
    if not bucket_uri:
        print('bucket sync skipped')
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _run(['hf', 'sync', bucket_uri, str(VOL_DIR)])


def _raw_data_ready() -> bool:
    images = list((DATA_DIR / 'Images').glob('*.png'))
    letters = list((DATA_DIR / 'Annotations' / 'Annotations').glob('*_letters.txt'))
    return len(images) >= 448 and bool(letters)


def _prepare_raw_data() -> None:
    import glob
    import tempfile
    import zipfile

    import requests

    img_dir = DATA_DIR / 'Images'
    ann_dir = DATA_DIR / 'Annotations'
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    if _raw_data_ready():
        print('raw data ready ->', DATA_DIR)
        return

    try:
        from huggingface_hub import snapshot_download

        print(f'downloading {HF_DATASET}', flush=True)
        root = Path(snapshot_download(
            repo_id=HF_DATASET,
            repo_type='dataset',
            allow_patterns=['data/**'],
            local_dir=str(VOL_DIR / 'hf_dataset'),
        ))
        ann_nested = ann_dir / 'Annotations'
        split_files = {
            'train': ann_dir / 'training_set.txt',
            'validation': ann_dir / 'validation_set.txt',
            'test': ann_dir / 'test_set.txt',
        }
        split_bases = {split: set() for split in split_files}
        for split in split_files:
            meta = root / 'data' / split / 'metadata.jsonl'
            with meta.open(encoding='utf-8') as fh:
                for line in fh:
                    rec = json.loads(line)
                    split_bases[split].add(rec['base_id'])
                    _link_or_copy(root / 'data' / split / rec['file_name'], img_dir / rec['file_name'])
                    ann_path = ann_nested / rec['annotation_file']
                    if not ann_path.exists():
                        ann_path.parent.mkdir(parents=True, exist_ok=True)
                        ann_path.write_text(rec['raw_annotation'].rstrip() + '\n', encoding='utf-8')
        for split, path in split_files.items():
            path.write_text('\n'.join(sorted(split_bases[split])) + '\n', encoding='utf-8')
        readme = ann_dir / 'README.txt'
        if not readme.exists():
            readme.write_text(
                'Prepared from Hugging Face dataset papadimas/trogs-greek-squeezes.\n'
                'Original source: https://scholarworks.smith.edu/dds_data/18\n',
                encoding='utf-8',
            )
        if _raw_data_ready():
            print(f'raw data ready from Hugging Face; images={len(glob.glob(str(img_dir / "*.png")))}')
            return
        print('Hugging Face dataset mirror was incomplete; using ScholarWorks', flush=True)
    except Exception as exc:
        print(f'Hugging Face dataset preparation failed: {exc}; using ScholarWorks', flush=True)

    for name, url in SCHOLARWORKS_FILES.items():
        dst = Path(tempfile.gettempdir()) / name
        print(f'downloading {name}', flush=True)
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with dst.open('wb') as fh:
                for chunk in response.iter_content(1 << 20):
                    fh.write(chunk)
        target = ann_dir if name == 'Annotations.zip' else img_dir
        with zipfile.ZipFile(dst) as archive:
            archive.extractall(target)
        dst.unlink()
        print(f'extracted {name}', flush=True)
    if not _raw_data_ready():
        raise RuntimeError('raw data preparation failed')


def _build_pipeline():
    import importlib

    _set_env()
    APP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(APP_DIR)
    runtime_dir = APP_DIR / 'squeeze_runtime'
    if str(runtime_dir) not in sys.path:
        sys.path.insert(0, str(runtime_dir))

    def load_module(name: str):
        return importlib.reload(importlib.import_module(name))

    load_module('duallight')
    load_module('prep_cache')
    LF = load_module('line_folds')
    R = load_module('rerank')
    CL = load_module('char_lattice')
    CR = load_module('charpost_ranker')
    artifact_checks = load_module('artifact_checks')
    charpost_pipeline = load_module('charpost_pipeline')
    runtime_config = load_module('runtime_config')

    layout = runtime_config.configure(DATA_DIR)
    for path in (DATA_DIR, layout['line_dir'], layout['charpost_dir'],
                 layout['splits_dir'], layout['ranker_dir']):
        path.mkdir(parents=True, exist_ok=True)

    cfg = artifact_checks.PipelineConfig(
        repo=VOL_DIR,
        charpost_dir=layout['charpost_dir'],
        folds=FOLDS,
        tile=4,
        folds_json=layout['folds_json'],
        line_all_train=layout['line_dir'] / 'all_train',
        line_oof_fold={fold: layout['line_dir'] / f'oof_fold{fold}' for fold in FOLDS},
        charpost_all_train_tag='all_train',
        charpost_oof_prefix='oof',
        charpost_orders=ORDERS,
        image_cache_dir=layout['image_cache_dir'],
        ranker_dir=layout['ranker_dir'],
        candidates_json=layout['ranker_dir'] / 'candidates.json',
        ranker_json=layout['ranker_dir'] / 'ranker_scores.json',
        ranker_methods='ridge',
        charpost_ranker_features='visual_sum,mean_char_logprob,length,length_delta,lm6_sum,lm10_sum,lm12_sum',
        rank_fold_seed=LF.RANK_FOLD_SEED,
    )

    artifacts = artifact_checks.ArtifactChecks(cfg)
    pipeline = charpost_pipeline.CharpostPipeline(
        cfg=cfg,
        artifacts=artifacts,
        rerank=R,
        line_folds=LF,
        char_lattice=CL,
        charpost_ranker=CR,
        run_training_if_missing=True,
    )
    return pipeline


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
    cpu=4,
    retries=JOB_RETRIES,
)
def prepare_inputs(bucket_uri: str = BUCKET_URI, require_line_models: bool = True) -> str:
    def run() -> str:
        _set_env()
        vol.reload()
        _sync_from_bucket(bucket_uri)
        _prepare_raw_data()
        pipeline = _build_pipeline()
        if require_line_models:
            pipeline.ensure_line_models()
        else:
            pipeline.create_oof_folds_if_missing()
            print('line checkpoint verification skipped; full pipeline will train or reuse them', flush=True)
        _commit_volume('prepare inputs')
        return 'inputs ready'

    return _run_locked('prepare-inputs', run)


def _atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _replace_dir_with_backup(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup = dst.with_name(f'.{dst.name}.backup')
    if backup.exists():
        shutil.rmtree(backup)
    if dst.exists():
        dst.rename(backup)
    try:
        src.rename(dst)
    except Exception:
        if backup.exists() and not dst.exists():
            backup.rename(dst)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _line_job_lock_name(kind: str, *, fold: int = -1) -> str:
    if kind == 'all-train':
        return 'line:all-train'
    if kind == 'oof-fold':
        return f'line:oof-fold:{fold}'
    return f'line:{kind}:{fold}'


def _line_job_id(kind: str, *, fold: int = -1) -> str:
    if kind == 'all-train':
        return 'line_all_train'
    if kind == 'oof-fold' and fold in FOLDS:
        return f'rank_SqueezeTrOCR_f{fold}'
    raise ValueError(f'unknown line job: kind={kind!r} fold={fold}')


def _line_target(pipeline, kind: str, *, fold: int = -1) -> Path:
    if kind == 'all-train':
        return pipeline.cfg.line_all_train
    if kind == 'oof-fold' and fold in FOLDS:
        return pipeline.cfg.line_oof_fold[fold]
    raise ValueError(f'unknown line target: kind={kind!r} fold={fold}')


def _line_run_config(kind: str, *, fold: int = -1) -> dict[str, object]:
    run_id = _line_job_id(kind, fold=fold)
    cfg: dict[str, object] = {
        'id': run_id,
        'base': LINE_BASE_MODEL,
        'lmc': str(LINE_TILE),
        'use_synth': False,
        'synth_n': 0,
        'synth_dir': None,
        'aug': 1,
        'processor_base': LINE_PROCESSOR_BASE,
        'max_steps': LINE_MAX_STEPS,
        'batch': LINE_BATCH,
        'eval_batch': LINE_EVAL_BATCH,
        'grad_accum': LINE_GRAD_ACCUM,
        'grad_checkpoint': True,
        'lr': LINE_LR,
        'eval_steps': LINE_EVAL_STEPS,
        'eval_cap': LINE_EVAL_CAP,
        'save_total_limit': LINE_SAVE_TOTAL_LIMIT,
        'eval_gate_cer': 0.2,
        'eval_gate_evals': 0,
        'dual': True,
        'dual_mode': 'min',
        'dual_divergent': 'both',
        'dual_register': True,
        'dual_register_method': 'phase',
        'dual_register_max_shift': 0.15,
        'dual_channel_swap': True,
        'dual_mono_dropout': 0.15,
    }
    if kind == 'oof-fold':
        cfg.update({
            'rank_fold_id': fold,
            'rank_folds': len(FOLDS),
            'rank_seed': 42,
        })
    return cfg


def _line_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = []
    for child in path.glob('checkpoint-*'):
        if not child.is_dir():
            continue
        try:
            step = int(child.name.rsplit('-', 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, child))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def _line_pair_cache_path(job_id: str, split: str) -> Path:
    return DATA_DIR / 'line_pair_cache' / f'{job_id}__{split}.pkl'


def _load_or_build_line_pairs(job_id: str, split: str, df, extract_rows) -> list:
    import pickle
    import duallight as DL

    cache = _line_pair_cache_path(job_id, split)
    if cache.exists():
        print('line pair cache ready ->', cache, flush=True)
        with cache.open('rb') as fh:
            return pickle.load(fh)
    print(f'building line pair cache job={job_id} split={split} squeezes={len(df)}', flush=True)
    pairs = DL.build_pairs_dual(df, extract_rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f'.{cache.name}.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as fh:
            pickle.dump(pairs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f'wrote line pair cache -> {cache} pairs={len(pairs)}', flush=True)
    return pairs


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    timeout=2 * 60 * 60,
    cpu=4,
    retries=JOB_RETRIES,
)
def line_prewarm_job() -> str:
    return _run_locked('line:prewarm', _line_prewarm_unlocked)


def _line_prewarm_unlocked() -> str:
    _set_env()
    vol.reload()
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    for repo_id in (LINE_BASE_MODEL, LINE_PROCESSOR_BASE):
        print('line prewarm HF model ->', repo_id, flush=True)
        snapshot_download(repo_id=repo_id, cache_dir=str(HF_CACHE))
    _commit_volume('line prewarm')
    return 'line prewarm done'


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    timeout=12 * 60 * 60,
    cpu=8,
    retries=JOB_RETRIES,
)
def line_job(kind: str, *, fold: int = -1) -> str:
    return _run_locked(
        _line_job_lock_name(kind, fold=fold),
        lambda: _line_job_unlocked(kind, fold=fold),
    )


def _line_job_unlocked(kind: str, *, fold: int = -1) -> str:
    _set_env()
    os.environ.update({
        'TROCR_BASE': LINE_PROCESSOR_BASE,
        'USE_SYNTH': '0',
        'AUG_LEVEL': '1',
        'PREP_CACHE': '1',
        'PREP_CACHE_DIR': str(DATA_DIR / 'prep_cache'),
        'DUAL_LIGHT': '1',
        'DUAL_MODE': 'min',
        'DUAL_DIVERGENT': 'both',
        'DUAL_REGISTER': '1',
        'DUAL_REGISTER_METHOD': 'phase',
        'DUAL_REGISTER_MAX_SHIFT': '0.15',
        'DUAL_CHANNEL_SWAP': '1',
        'DUAL_MONO_DROPOUT': '0.15',
        'TROCR_DATALOADER_WORKERS': str(LINE_DATALOADER_WORKERS),
        'TROCR_DATALOADER_PREFETCH': str(LINE_DATALOADER_PREFETCH),
    })
    vol.reload()
    pipeline = _build_pipeline()
    pipeline.create_oof_folds_if_missing()
    target = _line_target(pipeline, kind, fold=fold)
    run_config = _line_run_config(kind, fold=fold)
    job_id = str(run_config['id'])
    if pipeline.artifacts.checkpoint_ready(target):
        print('line checkpoint ready ->', target, flush=True)
        return f'{job_id} ready'

    import numpy as np
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
        default_data_collator,
    )
    from transformers.utils import logging as hf_logging
    from trocr_model import fix_trocr_meta

    hf_logging.set_verbosity_error()
    if os.environ.get('TORCH_SHARE_STRATEGY'):
        try:
            torch.multiprocessing.set_sharing_strategy(os.environ['TORCH_SHARE_STRATEGY'])
        except Exception as exc:
            print(f'warning: torch sharing strategy not set: {exc!r}', flush=True)

    # Line-recognizer context: dataset index, extractors, base processor,
    # prep_image, and training helpers -- no model weights loaded.
    g = pipeline.R.load(tile=LINE_TILE, load_model=False, verbose=True, train_context=True)

    index_df = g['index_df']
    train_df = index_df[index_df.split == 'train']
    if kind == 'oof-fold':
        folds = pipeline.LF.rank_folds_from_index(index_df, n_folds=len(FOLDS), seed=pipeline.cfg.rank_fold_seed)
        holdout = set(folds[fold])
        train_df = train_df[~train_df['base'].isin(holdout)]
        print(f'{job_id}: excluding rank fold {fold} train squeezes={len(holdout)}', flush=True)
    val_df = index_df[index_df.split == 'val']
    train_pairs = _load_or_build_line_pairs(job_id, 'train', train_df, g['extract_rows'])
    val_pairs = _load_or_build_line_pairs(job_id, 'val', val_df, g['extract_rows'])
    if not train_pairs or not val_pairs:
        raise RuntimeError(f'{job_id}: empty line training pairs train={len(train_pairs)} val={len(val_pairs)}')

    device = g['device']
    processor = g['processor']
    model = VisionEncoderDecoderModel.from_pretrained(LINE_BASE_MODEL, low_cpu_mem_usage=False).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = None
    model.config.num_beams = None
    model.config.use_cache = False
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = g['MAX_LEN']
    model.generation_config.num_beams = 1
    model.generation_config.max_new_tokens = None
    model = fix_trocr_meta(model, device)

    train_dir = DATA_DIR / 'line_training' / job_id
    train_dir.mkdir(parents=True, exist_ok=True)
    run_meta_path = train_dir / 'run_config.json'
    old_run_config = _read_json(run_meta_path)
    if old_run_config is not None and old_run_config != run_config:
        raise RuntimeError(f'{job_id}: existing training dir has different run_config; remove {train_dir} to retrain')
    _atomic_write_json(run_meta_path, run_config)

    latest = _line_latest_checkpoint(train_dir)
    resume = str(latest) if latest else None
    use_fp16 = bool(g.get('USE_FP16')) and torch.cuda.is_available()
    using_cuda = str(device).startswith('cuda')
    if using_cuda:
        torch.cuda.reset_peak_memory_stats()
    training_kwargs = dict(
        output_dir=str(train_dir),
        predict_with_generate=True,
        generation_max_length=g['MAX_LEN'],
        generation_num_beams=1,
        per_device_train_batch_size=LINE_BATCH,
        per_device_eval_batch_size=LINE_EVAL_BATCH,
        gradient_accumulation_steps=LINE_GRAD_ACCUM,
        gradient_checkpointing=True,
        num_train_epochs=100,
        max_steps=LINE_MAX_STEPS,
        learning_rate=LINE_LR,
        weight_decay=LINE_WEIGHT_DECAY,
        fp16=use_fp16,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        dataloader_pin_memory=using_cuda,
        dataloader_num_workers=LINE_DATALOADER_WORKERS,
        dataloader_prefetch_factor=LINE_DATALOADER_PREFETCH if LINE_DATALOADER_WORKERS else None,
        dataloader_persistent_workers=LINE_DATALOADER_WORKERS > 0,
        eval_steps=LINE_EVAL_STEPS,
        save_steps=LINE_EVAL_STEPS,
        save_total_limit=LINE_SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model='cer',
        greater_is_better=False,
        logging_steps=50,
        report_to='none',
    )
    try:
        args = Seq2SeqTrainingArguments(eval_strategy='steps', save_strategy='steps', **training_kwargs)
    except TypeError:
        args = Seq2SeqTrainingArguments(evaluation_strategy='steps', save_strategy='steps', **training_kwargs)

    import duallight as DL

    dataset_cls = DL.make_dual_dataset({
        'processor': processor,
        'prep_image': g['prep_image'],
        'augment': g['_augment'],
        'MAX_LEN': g['MAX_LEN'],
        'rng': g['_AUG_RNG'],
        'np': np,
        'mode': 'min',
        'channel_swap': True,
        'mono_dropout': 0.15,
    })
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset_cls(train_pairs, training=True),
        eval_dataset=dataset_cls(val_pairs[:LINE_EVAL_CAP], training=False),
        data_collator=default_data_collator,
        compute_metrics=g['compute_metrics'],
    )
    print(
        f'{job_id}: training start resume={resume or False} train_pairs={len(train_pairs)} '
        f'val_pairs={len(val_pairs)} batch={LINE_BATCH} eval_batch={LINE_EVAL_BATCH} fp16={use_fp16}',
        flush=True,
    )
    t0 = time.time()
    train_output = trainer.train(resume_from_checkpoint=resume)
    if using_cuda:
        torch.cuda.synchronize()
    train_profile: dict[str, object] = {
        'train_wall_seconds': time.time() - t0,
        'train_metrics': getattr(train_output, 'metrics', {}) or {},
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
    }
    if using_cuda:
        props = torch.cuda.get_device_properties(0)
        train_profile.update({
            'cuda_device_name': props.name,
            'cuda_total_memory_gb': props.total_memory / (1024 ** 3),
            'cuda_peak_allocated_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
            'cuda_peak_reserved_gb': torch.cuda.max_memory_reserved() / (1024 ** 3),
        })

    publish_tmp = target.with_name(f'.{target.name}.tmp-{os.getpid()}')
    if publish_tmp.exists():
        shutil.rmtree(publish_tmp)
    publish_tmp.mkdir(parents=True)
    trainer.model.save_pretrained(publish_tmp)
    processor.save_pretrained(publish_tmp)
    _atomic_write_json(publish_tmp / 'run_config.json', run_config)
    _atomic_write_json(publish_tmp / 'training_summary.json', train_profile)
    _replace_dir_with_backup(publish_tmp, target)
    if not pipeline.artifacts.checkpoint_ready(target):
        raise RuntimeError(f'{job_id}: published checkpoint failed validation: {target}')
    shutil.rmtree(train_dir, ignore_errors=True)
    _commit_volume(f'line job {job_id}')
    return f'{job_id} trained -> {target}'


def _line_job_ready(pipeline, kind: str, *, fold: int = -1) -> bool:
    return pipeline.artifacts.checkpoint_ready(_line_target(pipeline, kind, fold=fold))


@app.function(image=image, volumes={str(VOL_DIR): vol}, timeout=30 * 60, cpu=2)
def verify_line_models_job() -> dict:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    pipeline.ensure_line_models()
    return pipeline.artifacts.status().get('line_checkpoints', {})


def _cpu_job_lock_name(kind: str, *, fold: int, order: int, split: str, shards: int, shard: int) -> str:
    if kind in {'image-cache', 'merge-candidates', 'fit-ranker'}:
        return f'cpu:{kind}'
    if kind in {'oof-lms', 'oof-fold'}:
        return f'cpu:{kind}:fold={fold}'
    if kind == 'final-lm':
        return f'cpu:{kind}:order={order}'
    if kind in {'decode', 'decode-existing'}:
        return f'cpu:{kind}:split={split}'
    if kind in {'candidate-shard', 'decode-existing-shard'}:
        return f'cpu:{kind}:split={split}:shard={shard}-of-{shards}'
    if kind == 'merge-decodes':
        return f'cpu:{kind}:split={split}'
    return f'cpu:{kind}:fold={fold}:order={order}:split={split}:shard={shard}-of-{shards}'


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    timeout=12 * 60 * 60,
    cpu=8,
    retries=JOB_RETRIES,
)
def cpu_job(kind: str, *, fold: int = -1, order: int = -1, split: str = '', shards: int = 1, shard: int = -1, workers: int = 1) -> str:
    return _run_locked(
        _cpu_job_lock_name(kind, fold=fold, order=order, split=split, shards=shards, shard=shard),
        lambda: _cpu_job_unlocked(kind, fold=fold, order=order, split=split, shards=shards, shard=shard, workers=workers),
    )


def _cpu_job_unlocked(kind: str, *, fold: int = -1, order: int = -1, split: str = '', shards: int = 1, shard: int = -1, workers: int = 1) -> str:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    if kind == 'image-cache':
        pipeline.ensure_image_cache(workers=workers)
    elif kind == 'oof-lms':
        pipeline.ensure_oof_lms(fold)
    elif kind == 'final-lm':
        pipeline.ensure_final_lm(order)
    elif kind == 'candidate-shard':
        pipeline.ensure_candidate_shard(shard, shards)
    elif kind == 'merge-candidates':
        pipeline.merge_candidate_shards(shards)
    elif kind == 'fit-ranker':
        pipeline.ensure_ranker(candidate_shards=shards)
    elif kind == 'decode':
        pipeline.ensure_decode(split, ensure_inputs=False, device='cpu')
    elif kind == 'decode-existing':
        if split not in {'val', 'test'}:
            raise ValueError(f'decode-existing split must be val or test; got {split!r}')
        out = pipeline.cfg.charpost_dir / f'{pipeline.cfg.charpost_all_train_tag}__{split}_decode.json'
        args = argparse.Namespace(
            tag=pipeline.cfg.charpost_all_train_tag,
            split=split,
            ranker=pipeline.cfg.ranker_json,
            orders=pipeline.order_arg,
            lm_template=str(pipeline.cfg.charpost_dir / 'train_lm_order{order}_train' / 'lm.json'),
            proposal_weights='0,0.25,0.5,0.75,1.0',
            beam=256,
            char_topk=8,
            device='cpu',
            row_limit=0,
            no_score=False,
            keep_candidates=False,
            keep_features=False,
            out=out,
            row_shard_count=1,
            row_shard_id=0,
        )
        pipeline.CR.command_decode(args)
    elif kind == 'decode-existing-shard':
        if split not in {'val', 'test'}:
            raise ValueError(f'decode-existing-shard split must be val or test; got {split!r}')
        if not 0 <= shard < shards:
            raise ValueError(f'decode-existing-shard requires shard in [0, {shards}); got {shard}')
        out = pipeline.cfg.charpost_dir / f'{pipeline.cfg.charpost_all_train_tag}__{split}_decode_shard{shard:02d}-of-{shards:02d}.json'
        if pipeline.artifacts.decode_shard_ready(out, split, shard, shards):
            print('decode shard ready ->', out)
        else:
            args = argparse.Namespace(
                tag=pipeline.cfg.charpost_all_train_tag,
                split=split,
                ranker=pipeline.cfg.ranker_json,
                orders=pipeline.order_arg,
                lm_template=str(pipeline.cfg.charpost_dir / 'train_lm_order{order}_train' / 'lm.json'),
                proposal_weights='0,0.25,0.5,0.75,1.0',
                beam=256,
                char_topk=8,
                device='cpu',
                row_limit=0,
                no_score=False,
                keep_candidates=False,
                keep_features=False,
                out=out,
                row_shard_count=shards,
                row_shard_id=shard,
            )
            pipeline.CR.command_decode(args)
    elif kind == 'merge-decodes':
        if split not in {'val', 'test'}:
            raise ValueError(f'merge-decodes split must be val or test; got {split!r}')
        out = pipeline.cfg.charpost_dir / f'{pipeline.cfg.charpost_all_train_tag}__{split}_decode.json'
        args = argparse.Namespace(
            inputs=str(pipeline.cfg.charpost_dir / f'{pipeline.cfg.charpost_all_train_tag}__{split}_decode_shard*-of-{shards:02d}.json'),
            expect_shards=shards,
            out=out,
        )
        pipeline.CR.command_merge_decodes(args)
    else:
        raise ValueError(f'unknown CPU job: {kind}')
    _commit_volume(f'cpu job {kind}')
    return f'{kind} done'


def _gpu_job_lock_name(kind: str, *, fold: int, split: str) -> str:
    if kind == 'oof-fold':
        return f'gpu:{kind}:fold={fold}'
    if kind == 'final-logits':
        return f'gpu:{kind}:split={split}'
    return f'gpu:{kind}'


def _gpu_job_ready(pipeline, kind: str, *, fold: int = -1, split: str = '') -> bool:
    if kind == 'oof-fold':
        tag = pipeline.charpost_fold_tag(fold)
        model_dir = pipeline.cfg.charpost_dir / tag
        logits = pipeline.CL.logits_path(tag, f'fold{fold}')
        return (
            pipeline.artifacts.charpost_classifier_ready(
                model_dir,
                tag=tag,
                init_ckpt=pipeline.cfg.line_oof_fold[fold],
                train_exclude_fold_id=fold,
                val_fold_id=fold,
            )
            and pipeline.artifacts.charpost_logits_ready(
                logits,
                tag=tag,
                split=f'fold{fold}',
                model_dir=model_dir,
                include_fold_id=fold,
            )
        )
    if kind == 'final':
        model_dir = pipeline.cfg.charpost_dir / pipeline.cfg.charpost_all_train_tag
        return (
            pipeline.artifacts.charpost_classifier_ready(
                model_dir,
                tag=pipeline.cfg.charpost_all_train_tag,
                init_ckpt=pipeline.cfg.line_all_train,
            )
            and all(pipeline.artifacts.final_charpost_logits_ready(final_split, model_dir) for final_split in ('val', 'test'))
        )
    if kind == 'final-logits':
        model_dir = pipeline.cfg.charpost_dir / pipeline.cfg.charpost_all_train_tag
        return pipeline.artifacts.final_charpost_logits_ready(split, model_dir)
    return False


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    timeout=10 * 60 * 60,
    cpu=8,
    retries=JOB_RETRIES,
)
def gpu_job(kind: str, *, fold: int = -1, split: str = '') -> str:
    return _run_locked(
        _gpu_job_lock_name(kind, fold=fold, split=split),
        lambda: _gpu_job_unlocked(kind, fold=fold, split=split),
    )


def _gpu_job_unlocked(kind: str, *, fold: int = -1, split: str = '') -> str:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    if kind == 'oof-fold':
        pipeline.ensure_oof_classifier(fold)
        pipeline.ensure_oof_logits(fold)
    elif kind == 'final':
        pipeline.ensure_final_classifier()
        for final_split in ('val', 'test'):
            pipeline.ensure_final_logits(final_split)
    elif kind == 'final-logits':
        pipeline.ensure_final_logits(split)
    else:
        raise ValueError(f'unknown GPU job: {kind}')
    _commit_volume(f'gpu job {kind}')
    return f'{kind} done'


@app.function(image=image, volumes={str(VOL_DIR): vol}, timeout=90 * 60, cpu=4)
def cpu_smoke_job(kind: str, *, fold: int = -1, order: int = -1, split: str = '', shards: int = 1, shard: int = -1) -> str:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    def smoke_image_cache() -> None:
        out_dir = SMOKE_DIR / 'image_cache'
        _clear_smoke_path(out_dir)
        pipeline.run_char_lattice([
            'cache-images',
            '--init-ckpt', str(pipeline.cfg.line_all_train),
            '--input-mode', 'dual_pack',
            '--dual-mode', 'min',
            '--splits', 'train,val,test',
            '--image-cache-dir', str(out_dir),
            '--limit', '8',
            '--workers', '1',
            '--log-every', '8',
        ])
        manifest = out_dir / 'dual_pack__min' / 'manifest.json'
        if not manifest.exists():
            raise RuntimeError(f'image-cache smoke did not write manifest: {manifest}')

    def smoke_oof_lm(smoke_fold: int) -> None:
        if smoke_fold not in FOLDS:
            raise ValueError(f'oof-lms smoke requires fold in {FOLDS}; got {smoke_fold}')
        out_dir = SMOKE_DIR / 'oof_ranker' / f'train_lm_order{SMOKE_ORDER}_exclude_f{smoke_fold}'
        _clear_smoke_path(out_dir)
        pipeline.run_char_lattice([
            'train-lm',
            '--order', str(SMOKE_ORDER),
            '--splits', 'train',
            '--folds', str(pipeline.cfg.folds_json),
            '--exclude-fold-id', str(smoke_fold),
            '--out-dir', str(out_dir),
        ])
        if not (out_dir / 'lm.json').exists():
            raise RuntimeError(f'OOF LM smoke did not write {out_dir / "lm.json"}')

    def smoke_final_lm(smoke_order: int) -> None:
        out_dir = SMOKE_DIR / f'final_lm_order{smoke_order}_train'
        _clear_smoke_path(out_dir)
        pipeline.run_char_lattice([
            'train-lm',
            '--order', str(smoke_order),
            '--splits', 'train',
            '--out-dir', str(out_dir),
        ])
        if not (out_dir / 'lm.json').exists():
            raise RuntimeError(f'final LM smoke did not write {out_dir / "lm.json"}')

    def smoke_candidate_shard(smoke_shard: int, smoke_shards: int) -> None:
        if smoke_shards < 1:
            raise ValueError('candidate-shard smoke requires shards >= 1')
        if not 0 <= smoke_shard < smoke_shards:
            raise ValueError(f'candidate-shard smoke requires shard in [0, {smoke_shards}); got {smoke_shard}')
        out = SMOKE_DIR / 'oof_ranker' / f'candidates_shard{smoke_shard:02d}-of-{smoke_shards:02d}.json'
        _clear_smoke_path(out)
        pipeline.run_charpost_ranker([
            'build-candidates',
            '--tag-prefix', 'smoke_oof',
            '--folds-ids', ','.join(str(fold_id) for fold_id in SMOKE_FOLDS),
            '--orders', str(SMOKE_ORDER),
            '--fold-lm-template', _smoke_oof_lm_template(),
            '--proposal-weights', '0,0.5',
            '--beam', '32',
            '--char-topk', '4',
            '--row-shard-count', str(smoke_shards),
            '--row-shard-id', str(smoke_shard),
            '--row-limit', str(SMOKE_ROWS),
            '--out', str(out),
        ])
        if not out.exists():
            raise RuntimeError(f'candidate-shard smoke did not write {out}')

    def smoke_merge_candidates(smoke_shards: int) -> None:
        inputs = [SMOKE_DIR / 'oof_ranker' / f'candidates_shard{sid:02d}-of-{smoke_shards:02d}.json' for sid in range(smoke_shards)]
        missing = [str(path) for path in inputs if not path.exists()]
        if missing:
            raise FileNotFoundError('candidate smoke shards missing before merge: ' + ', '.join(missing))
        out = SMOKE_DIR / 'oof_ranker' / 'candidates.json'
        _clear_smoke_path(out)
        pipeline.run_charpost_ranker([
            'merge-candidates',
            '--inputs', ','.join(str(path) for path in inputs),
            '--expect-shards', str(smoke_shards),
            '--out', str(out),
        ])
        if not out.exists():
            raise RuntimeError(f'merge-candidates smoke did not write {out}')

    def smoke_fit_ranker() -> None:
        candidates = SMOKE_DIR / 'oof_ranker' / 'candidates.json'
        if not candidates.exists():
            raise FileNotFoundError(f'candidate smoke merge missing before fit: {candidates}')
        out = SMOKE_DIR / 'oof_ranker' / 'ranker_scores.json'
        _clear_smoke_path(out)
        pipeline.run_charpost_ranker([
            'fit-rankers',
            '--candidates', str(candidates),
            '--out', str(out),
            '--features', pipeline.cfg.charpost_ranker_features,
            '--methods', pipeline.cfg.ranker_methods,
        ])
        if not out.exists():
            raise RuntimeError(f'fit-ranker smoke did not write {out}')

    def smoke_decode(smoke_split: str) -> None:
        if smoke_split not in {'val', 'test'}:
            raise ValueError(f'decode smoke split must be val or test; got {smoke_split!r}')
        ranker = SMOKE_DIR / 'oof_ranker' / 'ranker_scores.json'
        if not ranker.exists():
            raise FileNotFoundError(f'smoke ranker missing before decode: {ranker}')
        out = SMOKE_DIR / 'oof_ranker' / f'smoke_all_train__{smoke_split}_decode.json'
        _clear_smoke_path(out)
        pipeline.run_charpost_ranker([
            'decode',
            '--tag', 'smoke_all_train',
            '--split', smoke_split,
            '--ranker', str(ranker),
            '--orders', str(SMOKE_ORDER),
            '--lm-template', _smoke_final_lm_template(),
            '--proposal-weights', '0,0.5',
            '--beam', '32',
            '--char-topk', '4',
            '--device', 'cpu',
            '--row-limit', str(SMOKE_ROWS),
            '--out', str(out),
        ])
        if not out.exists():
            raise RuntimeError(f'decode smoke did not write {out}')
    
    smoke_shards = min(max(int(shards), 1), 2)
    if kind == 'image-cache':
        smoke_image_cache()
    elif kind == 'lms':
        for smoke_fold in SMOKE_FOLDS:
            smoke_oof_lm(smoke_fold)
        smoke_final_lm(SMOKE_ORDER)
    elif kind == 'oof-lms':
        smoke_oof_lm(fold)
    elif kind == 'final-lm':
        smoke_final_lm(order if order > 0 else SMOKE_ORDER)
    elif kind == 'candidates':
        ranker_dir = SMOKE_DIR / 'oof_ranker'
        _clear_smoke_path(ranker_dir / 'candidates.json')
        _clear_smoke_path(ranker_dir / 'ranker_scores.json')
        for smoke_shard in range(smoke_shards):
            smoke_candidate_shard(smoke_shard, smoke_shards)
    elif kind == 'candidate-shard':
        smoke_candidate_shard(shard, max(int(shards), 1))
    elif kind == 'ranker':
        smoke_merge_candidates(smoke_shards)
        smoke_fit_ranker()
    elif kind == 'merge-candidates':
        smoke_merge_candidates(max(int(shards), 1))
    elif kind == 'fit-ranker':
        smoke_fit_ranker()
    elif kind == 'decodes':
        for smoke_split in ('val', 'test'):
            smoke_decode(smoke_split)
    elif kind == 'decode':
        smoke_decode(split)
    else:
        raise ValueError(f'unknown CPU smoke job: {kind}')
    _commit_volume(f'cpu smoke job {kind}')
    return f'{kind} smoke done'


@app.function(image=image, volumes={str(VOL_DIR): vol}, timeout=90 * 60, cpu=8)
def gpu_smoke_job(kind: str, *, fold: int = -1) -> str:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    def smoke_oof_fold(smoke_fold: int) -> None:
        if smoke_fold not in FOLDS:
            raise ValueError(f'oof-fold smoke requires fold in {FOLDS}; got {smoke_fold}')
        tag = f'smoke_oof_f{smoke_fold}'
        model_dir = SMOKE_DIR / 'models' / tag
        image_cache = SMOKE_DIR / 'gpu_image_cache' / tag
        _clear_smoke_path(model_dir)
        _clear_smoke_path(image_cache)
        logits_path = pipeline.CL.logits_path(tag, f'fold{smoke_fold}')
        if logits_path.exists():
            logits_path.unlink()
        pipeline.run_char_lattice([
            'train-classifier',
            '--tag', tag,
            '--init-ckpt', str(pipeline.cfg.line_oof_fold[smoke_fold]),
            '--out-dir', str(model_dir),
            '--input-mode', 'dual_pack',
            '--dual-mode', 'min',
            '--batch', '8',
            '--eval-batch', '8',
            '--grad-accum', '1',
            '--max-steps', '1',
            '--eval-steps', '0',
            '--log-steps', '1',
            '--lr', '1e-5',
            '--weight-decay', '1e-4',
            '--class-weight', 'sqrt',
            '--workers', '0',
            '--train-limit', '16',
            '--val-limit', '8',
            '--image-cache-dir', str(image_cache),
            '--no-save-during-training',
            '--no-resume',
            '--folds', str(pipeline.cfg.folds_json),
            '--train-exclude-fold-id', str(smoke_fold),
            '--val-fold-id', str(smoke_fold),
        ])
        pipeline.run_char_lattice([
            'cache-logits',
            '--tag', tag,
            '--model-dir', str(model_dir),
            '--splits', 'train',
            '--output-split', f'fold{smoke_fold}',
            '--include-fold-id', str(smoke_fold),
            '--folds', str(pipeline.cfg.folds_json),
            '--batch', '8',
            '--limit', '8',
            '--tta', 'shift5',
            '--tta-pixels', '4',
            '--tta-scale', '0.04',
            '--image-cache-dir', str(image_cache),
        ])
        if not logits_path.exists():
            raise RuntimeError(f'OOF logits smoke did not write {logits_path}')

    def smoke_final_classifier() -> None:
        tag = 'smoke_all_train'
        model_dir = SMOKE_DIR / 'models' / tag
        image_cache = SMOKE_DIR / 'gpu_image_cache' / tag
        _clear_smoke_path(model_dir)
        _clear_smoke_path(image_cache)
        for final_split in ('val', 'test'):
            logits_path = pipeline.CL.logits_path(tag, final_split)
            if logits_path.exists():
                logits_path.unlink()
        pipeline.run_char_lattice([
            'train-classifier',
            '--tag', tag,
            '--init-ckpt', str(pipeline.cfg.line_all_train),
            '--out-dir', str(model_dir),
            '--input-mode', 'dual_pack',
            '--dual-mode', 'min',
            '--batch', '8',
            '--eval-batch', '8',
            '--grad-accum', '1',
            '--max-steps', '1',
            '--eval-steps', '0',
            '--log-steps', '1',
            '--lr', '1e-5',
            '--weight-decay', '1e-4',
            '--class-weight', 'sqrt',
            '--workers', '0',
            '--train-limit', '16',
            '--val-limit', '8',
            '--image-cache-dir', str(image_cache),
            '--no-save-during-training',
            '--no-resume',
        ])
        for final_split in ('val', 'test'):
            pipeline.run_char_lattice([
                'cache-logits',
                '--tag', tag,
                '--model-dir', str(model_dir),
                '--splits', final_split,
                '--batch', '8',
                '--limit', '8',
                '--tta', 'shift5',
                '--tta-pixels', '4',
                '--tta-scale', '0.04',
                '--image-cache-dir', str(image_cache),
            ])
            logits_path = pipeline.CL.logits_path(tag, final_split)
            if not logits_path.exists():
                raise RuntimeError(f'final logits smoke did not write {logits_path}')

    if kind == 'classifiers':
        for smoke_fold in SMOKE_FOLDS:
            smoke_oof_fold(smoke_fold)
        smoke_final_classifier()
    elif kind == 'oof-fold':
        smoke_oof_fold(fold)
    elif kind == 'final':
        smoke_final_classifier()
    else:
        raise ValueError(f'unknown GPU smoke job: {kind}')
    _commit_volume(f'gpu smoke job {kind}')
    return f'{kind} smoke done'


@app.function(image=image, volumes={str(VOL_DIR): vol}, timeout=30 * 60, cpu=2)
def status_job() -> dict:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    return pipeline.artifacts.status()


@app.function(image=image, volumes={str(VOL_DIR): vol}, timeout=30 * 60, cpu=2)
def summary_job() -> dict:
    _set_env()
    vol.reload()
    pipeline = _build_pipeline()
    ranker_path = pipeline.cfg.ranker_json
    ranker = _read_json(ranker_path)
    out: dict[str, object] = {
        'ranker_path': str(ranker_path),
        'ranker_exists': ranker_path.exists(),
        'oof_best': None,
        'oof_folds': [],
        'final_fit': None,
        'decodes': {},
    }
    if isinstance(ranker, dict):
        best = ranker.get('best') or {}
        final_fit = ranker.get('final_fit') or {}
        out['oof_best'] = {
            'method': best.get('method'),
            'cer': best.get('cer'),
            'errors': best.get('errors'),
            'chars': best.get('chars'),
            'rows': best.get('rows'),
            'oracle_cer': best.get('oracle_cer'),
        }
        out['oof_folds'] = (best.get('folds') or []) if isinstance(best, dict) else []
        out['final_fit'] = {
            'method': final_fit.get('method'),
            'features': final_fit.get('features') or final_fit.get('cols'),
            'failed': final_fit.get('failed'),
        }
    for split in ('val', 'test'):
        path = pipeline.cfg.charpost_dir / f'{pipeline.cfg.charpost_all_train_tag}__{split}_decode.json'
        data = _read_json(path)
        rec: dict[str, object] = {'path': str(path), 'exists': path.exists(), 'scored': None}
        if isinstance(data, dict):
            rec.update({
                'source': data.get('source'),
                'ranker_method': data.get('ranker_method'),
                'features': data.get('features'),
                'orders': data.get('orders'),
                'proposal_weights': data.get('proposal_weights'),
                'beam': data.get('beam'),
                'char_topk': data.get('char_topk'),
                'scored': data.get('scored'),
            })
        out['decodes'][split] = rec
    return out


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    secrets=[hf_secret],
    timeout=12 * 60 * 60,
    cpu=8,
    retries=JOB_RETRIES,
)
def prod_eval(
    source: str = PROD_DEFAULT_SOURCE,
    *,
    source_split: str = 'test',
    name: str = '',
    revision: str = '',
    bucket_uri: str = BUCKET_URI,
    image_workers: int = 8,
    decode_shards: int = 1,
    force: bool = False,
) -> dict:
    lock = f'prod:{_prod_slug(name or source)}:{source_split}:{revision}'
    return _run_locked(
        lock,
        lambda: _prod_eval_unlocked(
            source=source,
            source_split=source_split,
            name=name,
            revision=revision,
            bucket_uri=bucket_uri,
            image_workers=image_workers,
            decode_shards=decode_shards,
            force=force,
        ),
    )


def _prod_eval_unlocked(
    *,
    source: str,
    source_split: str,
    name: str,
    revision: str,
    bucket_uri: str,
    image_workers: int,
    decode_shards: int,
    force: bool,
) -> dict:
    if decode_shards < 1:
        raise ValueError('decode_shards must be >= 1')
    _set_env()
    try:
        vol.reload()
    except Exception as exc:
        print(f'warning: prod vol.reload skipped: {exc}', flush=True)
    if bucket_uri:
        _sync_from_bucket(bucket_uri)
    pipeline = _build_pipeline()
    _require_prod_frozen_artifacts(pipeline)

    manifest = _stage_prod_dataset(
        source=source,
        source_split=source_split,
        name=name,
        revision=revision,
        force=force,
    )
    slug = str(manifest['slug'])
    tag = f'prod_{slug}'
    split = PROD_OUTPUT_SPLIT
    input_root = Path(str(manifest['input_root']))
    model_dir = pipeline.cfg.charpost_dir / pipeline.cfg.charpost_all_train_tag
    image_cache_dir = pipeline.cfg.charpost_dir / 'prod_image_cache' / slug
    logits_path = pipeline.cfg.charpost_dir / f'{tag}__{split}_logits.npz'
    decode_path = pipeline.cfg.charpost_dir / f'{tag}__{split}_decode.json'
    out_dir = PROD_OUTPUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    rebuild_outputs = force or bool(manifest.get('_rebuilt'))
    if rebuild_outputs and image_cache_dir.exists():
        shutil.rmtree(image_cache_dir)

    with _prod_input_context(pipeline, input_root):
        cache_args = [
            'cache-images',
            '--init-ckpt', str(pipeline.cfg.line_all_train),
            '--input-mode', 'dual_pack',
            '--dual-mode', 'min',
            '--splits', 'test',
            '--image-cache-dir', str(image_cache_dir),
            '--workers', str(max(1, int(image_workers))),
        ]
        if rebuild_outputs:
            cache_args.append('--force')
        pipeline.run_char_lattice(cache_args)

        if rebuild_outputs:
            logits_path.unlink(missing_ok=True)
        if _prod_logits_ready(logits_path):
            print('prod logits ready ->', logits_path, flush=True)
        else:
            pipeline.run_char_lattice([
                'cache-logits',
                '--tag', tag,
                '--model-dir', str(model_dir),
                '--splits', 'test',
                '--output-split', split,
                '--batch', '64',
                '--device', 'auto',
                '--tta', 'shift5',
                '--tta-pixels', '4',
                '--tta-scale', '0.04',
                '--image-cache-dir', str(image_cache_dir),
                '--require-image-cache',
            ])
            if not _prod_logits_ready(logits_path):
                raise RuntimeError(f'prod logits failed validation: {logits_path}')

    def decode_args(out: Path, *, shard_id: int, shard_count: int) -> argparse.Namespace:
        return argparse.Namespace(
            tag=tag,
            split=split,
            ranker=pipeline.cfg.ranker_json,
            orders=','.join(str(order) for order in pipeline.cfg.charpost_orders),
            lm_template=str(pipeline.cfg.charpost_dir / 'train_lm_order{order}_train' / 'lm.json'),
        proposal_weights='0,0.25,0.5,0.75,1.0',
        beam=256,
        char_topk=8,
        device='cpu',
        row_limit=0,
            row_shard_count=shard_count,
            row_shard_id=shard_id,
            no_score=False,
            keep_candidates=False,
            keep_features=False,
            out=out,
        )

    if rebuild_outputs:
        decode_path.unlink(missing_ok=True)
        for path in pipeline.cfg.charpost_dir.glob(f'{tag}__{split}_decode_shard*-of-*.json'):
            path.unlink(missing_ok=True)
    if decode_shards == 1:
        if _prod_decode_matches(decode_path, tag=tag, split=split, shard=0, shards=1):
            print('prod decode ready ->', decode_path, flush=True)
        else:
            pipeline.CR.command_decode(decode_args(decode_path, shard_id=0, shard_count=1))
    else:
        for sid in range(decode_shards):
            shard_path = pipeline.cfg.charpost_dir / f'{tag}__{split}_decode_shard{sid:02d}-of-{decode_shards:02d}.json'
            if _prod_decode_matches(shard_path, tag=tag, split=split, shard=sid, shards=decode_shards):
                print('prod decode shard ready ->', shard_path, flush=True)
                continue
            pipeline.CR.command_decode(decode_args(shard_path, shard_id=sid, shard_count=decode_shards))
        pipeline.CR.command_merge_decodes(argparse.Namespace(
            inputs=str(pipeline.cfg.charpost_dir / f'{tag}__{split}_decode_shard*-of-{decode_shards:02d}.json'),
            expect_shards=decode_shards,
            out=decode_path,
        ))
    if not _prod_decode_matches(decode_path, tag=tag, split=split):
        raise RuntimeError(f'prod decode failed validation: {decode_path}')
    transcripts_dir = _write_prod_transcripts(decode_path, out_dir)
    summary = _prod_summary(
        source=source,
        manifest=manifest,
        tag=tag,
        split=split,
        logits=logits_path,
        decode=decode_path,
        transcripts_dir=transcripts_dir,
    )
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    _commit_volume(f'prod eval {slug}')
    return summary


@app.function(image=image, volumes={str(VOL_DIR): vol}, timeout=10 * 60, cpu=1)
def clear_locks_job() -> dict:
    _set_env()
    vol.reload()
    removed: list[str] = []
    if LOCK_DIR.exists():
        for path in sorted(LOCK_DIR.iterdir()):
            removed.append(path.name)
        shutil.rmtree(LOCK_DIR)
    _commit_volume('clear locks')
    return {'removed': removed}


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
    cpu=16,
    retries=JOB_RETRIES,
)
def upload_artifacts(bucket_uri: str = BUCKET_URI) -> str:
    return _run_locked(
        'upload',
        lambda: _upload_artifacts_unlocked(bucket_uri),
    )


def _upload_artifacts_unlocked(bucket_uri: str = BUCKET_URI) -> str:
    _set_env()
    vol.reload()
    _cleanup_smoke_artifacts()
    for name in ('line_recognizer', 'splits', 'results', 'character_posterior'):
        src = DATA_DIR / name
        if not src.exists():
            print('sync skipped missing ->', src, flush=True)
            continue
        target_uri = bucket_uri.rstrip('/') + f'/data/{name}'
        print('hf sync upload ->', src, '=>', target_uri, flush=True)
        _run(['hf', 'sync', str(src), target_uri])
    _commit_volume('upload full artifacts')
    return f'uploaded artifacts to {bucket_uri}'


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    secrets=[hf_secret],
    timeout=60 * 60,
    cpu=4,
    retries=JOB_RETRIES,
)
def upload_prod_result(
    slug: str = 'trogs26-test',
    *,
    bucket_uri: str = BUCKET_URI,
    include_logits: bool = True,
) -> str:
    return _run_locked(
        f'upload-prod:{_prod_slug(slug)}',
        lambda: _upload_prod_result_unlocked(slug, bucket_uri=bucket_uri, include_logits=include_logits),
    )


def _upload_prod_result_unlocked(slug: str, *, bucket_uri: str = BUCKET_URI, include_logits: bool = True) -> str:
    _set_env()
    try:
        vol.reload()
    except Exception as exc:
        print(f'warning: upload-prod vol.reload skipped: {exc}', flush=True)
    slug = _prod_slug(slug)
    stage = _stage_prod_result_for_upload(slug, include_logits=include_logits)
    target_uri = bucket_uri.rstrip('/') + f'/data/prod_runs/{slug}'
    print('hf sync upload ->', target_uri, flush=True)
    _run(['hf', 'sync', str(stage), target_uri])
    _commit_volume(f'upload prod result {slug}')
    return f'uploaded prod result {slug} to {target_uri}'


def _wait(label: str, calls: list[object]) -> list[str]:
    results = []
    for idx, call in enumerate(calls, start=1):
        result = call.get()
        print(f'{label} {idx}/{len(calls)}: {result}', flush=True)
        results.append(result)
    return results


def _dispatch(label: str, calls: list[object], *, bg: bool) -> list[str]:
    if not calls:
        return []
    if bg:
        for idx, call in enumerate(calls, start=1):
            print(f'{label} {idx}/{len(calls)} spawned: {call.object_id} -- safe to disconnect; watch the dashboard', flush=True)
        return []
    return _wait(label, calls)


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _run_smoke_sequence(gpu: str, candidate_shards: int) -> None:
    print('smoke: image cache', flush=True)
    print(cpu_smoke_job.remote('image-cache'))

    print('smoke: language models', flush=True)
    print(cpu_smoke_job.remote('lms'))

    print('smoke: classifiers and logits', flush=True)
    gpu_fn = gpu_smoke_job.with_options(gpu=gpu)
    print(gpu_fn.remote('classifiers'))

    print('smoke: candidates', flush=True)
    print(cpu_smoke_job.remote('candidates', shards=candidate_shards))

    print('smoke: ranker', flush=True)
    print(cpu_smoke_job.remote('ranker', shards=candidate_shards))

    print('smoke: decodes', flush=True)
    print(cpu_smoke_job.remote('decodes'))


def _parse_single_gpu_job(value: str) -> tuple[str, dict[str, int | str]]:
    name = value.strip().lower().replace('_', '-')
    if not name:
        raise SystemExit('single_gpu_job must be non-empty')
    if name in {'final', 'all-train', 'alltrain'}:
        return 'final', {}
    if name.startswith('final-logits-'):
        split = name.removeprefix('final-logits-')
        if split not in {'val', 'test'}:
            raise SystemExit('single_gpu_job final-logits-* split must be val or test')
        return 'final-logits', {'split': split}
    for prefix in ('oof-fold', 'oof', 'fold', 'f'):
        if name.startswith(prefix):
            raw = name.removeprefix(prefix).lstrip('-')
            if raw.isdigit() and int(raw) in FOLDS:
                return 'oof-fold', {'fold': int(raw)}
    raise SystemExit(
        'single_gpu_job must be one of: oof0, oof1, oof2, oof3, oof4, final, final-logits-val, final-logits-test'
    )


@app.function(
    image=image,
    volumes={str(VOL_DIR): vol},
    timeout=24 * 60 * 60,
    cpu=2,
    retries=ORCHESTRATOR_RETRIES,
)
def run_full_pipeline(
    bucket_uri: str = BUCKET_URI,
    gpu: str = GPU_DEFAULT,
    candidate_shards: int = 8,
    decode_shards: int = 8,
    image_workers: int = 8,
    train_line_models: bool = True,
    upload_full: bool = False,
) -> dict:
    return _run_locked(
        'full-pipeline',
        lambda: _run_full_pipeline_unlocked(
            bucket_uri=bucket_uri,
            gpu=gpu,
            candidate_shards=candidate_shards,
            decode_shards=decode_shards,
            image_workers=image_workers,
            train_line_models=train_line_models,
            upload_full=upload_full,
        ),
    )


def _run_full_pipeline_unlocked(
    bucket_uri: str = BUCKET_URI,
    gpu: str = GPU_DEFAULT,
    candidate_shards: int = 8,
    decode_shards: int = 8,
    image_workers: int = 8,
    train_line_models: bool = True,
    upload_full: bool = False,
) -> dict:
    """Run the entire charpost pipeline end-to-end, server-side.

    Phase 0: prepare inputs and train/reuse the SqueezeTrOCR line recognizer checkpoints.
    Phase 1 (head): image cache, OOF/final LMs, OOF/final GPU classifiers+logits.
    Phase 2 (tail): candidate shards, merge, fit ranker, sharded decode, merge-decodes.
    Then optional full artifact upload, and a summary.

    Designed to be launched with .spawn() from a detached local entrypoint: this
    function lives on Modal and orchestrates the sub-functions via .remote()/.spawn(),
    so it keeps running after the local caller disconnects.
    """
    _set_env()

    print('=== full pipeline: prepare ===', flush=True)
    print(prepare_inputs.remote(bucket_uri, require_line_models=not train_line_models), flush=True)

    if train_line_models:
        print('=== full pipeline: line recognizers (SqueezeTrOCR all-train + clean OOF folds) ===', flush=True)
        try:
            vol.reload()
        except Exception as exc:
            print(f'warning: full pipeline vol.reload before line stage skipped: {exc}', flush=True)
        pipeline = _build_pipeline()
        line_specs = [('all-train', {})] + [('oof-fold', {'fold': fold}) for fold in FOLDS]
        needed = [
            (kind, kwargs)
            for kind, kwargs in line_specs
            if not _line_job_ready(pipeline, kind, **kwargs)
        ]
        if needed:
            print(line_prewarm_job.remote(), flush=True)
            _wait_for_locks_to_clear(
                [
                    _line_job_lock_name(kind, fold=int(kwargs.get('fold', -1)))
                    for kind, kwargs in needed
                ],
                label='line',
            )
            line_fn = line_job.with_options(gpu=gpu)
            line_calls = [line_fn.spawn(kind, **kwargs) for kind, kwargs in needed]
            _wait('line job', line_calls)
        else:
            print('line recognizers already ready -> skip training', flush=True)
        print(json.dumps(verify_line_models_job.remote(), indent=2, sort_keys=True), flush=True)

    print('=== full pipeline: head (image-cache + LMs, then GPU) ===', flush=True)
    image_call = cpu_job.spawn('image-cache', workers=image_workers)
    lm_calls = [
        cpu_job.spawn('oof-lms', fold=fold)
        for fold in FOLDS
    ] + [
        cpu_job.spawn('final-lm', order=order)
        for order in ORDERS
    ]
    _wait('image-cache', [image_call])

    print('=== full pipeline: GPU classifiers/logits ===', flush=True)
    gpu_specs = [
        ('oof-fold', {'fold': fold})
        for fold in FOLDS
    ] + [
        ('final', {}),
    ]
    _wait_for_locks_to_clear(
        [
            _gpu_job_lock_name(kind, fold=int(kwargs.get('fold', -1)), split=str(kwargs.get('split', '')))
            for kind, kwargs in gpu_specs
        ],
        label='GPU',
    )
    pipeline = _build_pipeline()
    gpu_fn = gpu_job.with_options(gpu=gpu)
    gpu_calls = []
    for kind, kwargs in gpu_specs:
        if _gpu_job_ready(pipeline, kind, **kwargs):
            print(f'gpu job ready -> {kind} {kwargs}', flush=True)
            continue
        gpu_calls.append(gpu_fn.spawn(kind, **kwargs))
    _wait('language-model job', lm_calls)
    _wait('gpu job', gpu_calls)

    print('=== full pipeline: score tail (candidates + ranker + decode) ===', flush=True)
    shard_calls = [
        cpu_job.spawn('candidate-shard', shard=sid, shards=candidate_shards)
        for sid in range(candidate_shards)
    ]
    _wait('candidate shard', shard_calls)
    print(cpu_job.remote('merge-candidates', shards=candidate_shards), flush=True)
    print(cpu_job.remote('fit-ranker', shards=candidate_shards), flush=True)

    decode_calls = [
        cpu_job.spawn('decode-existing-shard', split=split, shard=sid, shards=decode_shards)
        for split in ('val', 'test')
        for sid in range(decode_shards)
    ]
    _wait('decode shard', decode_calls)
    for split in ('val', 'test'):
        print(cpu_job.remote('merge-decodes', split=split, shards=decode_shards), flush=True)

    print('=== full pipeline: summary ===', flush=True)
    result = summary_job.remote()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if upload_full:
        print('=== full pipeline: upload-full ===', flush=True)
        print(upload_artifacts.remote(bucket_uri), flush=True)
    return result


@app.local_entrypoint()
def main(
    run: bool = False,
    full: bool = False,
    line_only: bool = False,
    prod: bool = False,
    smoke: bool = False,
    skip_smoke: bool = False,
    prepare: bool = False,
    status: bool = False,
    summary: bool = False,
    clear_locks: bool = False,
    score_only: bool = False,
    upload: bool = False,
    upload_prod_result_only: bool = False,
    skip_line_training: bool = False,
    bucket_uri: str = BUCKET_URI,
    gpu: str = GPU_DEFAULT,
    prod_source: str = PROD_DEFAULT_SOURCE,
    prod_source_split: str = 'test',
    prod_name: str = '',
    prod_revision: str = '',
    prod_force: bool = False,
    prod_decode_shards: int = 1,
    prod_include_logits: bool = True,
    candidate_shards: int = 8,
    decode_shards: int = 8,
    image_workers: int = 8,
    single_gpu_job: str = '',
    bg: bool = False,
) -> None:
    if candidate_shards < 1:
        raise SystemExit('candidate_shards must be >= 1')
    if decode_shards < 1:
        raise SystemExit('decode_shards must be >= 1')
    if prod_decode_shards < 1:
        raise SystemExit('prod_decode_shards must be >= 1')
    if summary:
        print(json.dumps(summary_job.remote(), indent=2, sort_keys=True))
        return
    if status:
        print(json.dumps(status_job.remote(), indent=2, sort_keys=True))
        return
    if clear_locks:
        print(json.dumps(clear_locks_job.remote(), indent=2, sort_keys=True))
        return
    if prepare:
        print(prepare_inputs.remote(bucket_uri))
        return
    if line_only:
        print(prepare_inputs.remote(bucket_uri, require_line_models=False))
        print(line_prewarm_job.remote())
        line_fn = line_job.with_options(gpu=gpu)
        specs = [('all-train', {})] + [('oof-fold', {'fold': fold}) for fold in FOLDS]
        calls = [line_fn.spawn(kind, **kwargs) for kind, kwargs in specs]
        _dispatch('line job', calls, bg=bg)
        if not bg:
            print(json.dumps(verify_line_models_job.remote(), indent=2, sort_keys=True))
        return
    if upload_prod_result_only:
        slug = prod_name or _prod_slug(_prod_repo_id(prod_source) + f'_{prod_source_split}')
        print(upload_prod_result.remote(slug, bucket_uri=bucket_uri, include_logits=prod_include_logits))
        return
    if prod:
        prod_fn = prod_eval.with_options(gpu=gpu)
        kwargs = dict(
            source=prod_source,
            source_split=prod_source_split,
            name=prod_name,
            revision=prod_revision,
            bucket_uri=bucket_uri,
            image_workers=image_workers,
            decode_shards=prod_decode_shards,
            force=prod_force,
        )
        if bg:
            call = prod_fn.spawn(**kwargs)
            print(f'prod eval spawned: {call.object_id} -- safe to disconnect; watch the dashboard', flush=True)
        else:
            print(json.dumps(prod_fn.remote(**kwargs), indent=2, sort_keys=True))
        return
    if full:
        full_kwargs = dict(
            bucket_uri=bucket_uri,
            gpu=gpu,
            candidate_shards=candidate_shards,
            decode_shards=decode_shards,
            image_workers=image_workers,
            train_line_models=not skip_line_training,
            upload_full=upload,
        )
        if bg:
            call = run_full_pipeline.spawn(**full_kwargs)
            print(f'full pipeline spawned: {call.object_id} -- safe to disconnect; watch the dashboard', flush=True)
        else:
            result = run_full_pipeline.remote(**full_kwargs)
            print(json.dumps(result, indent=2, sort_keys=True))
        return
    if upload and not run and not score_only:
        if bg:
            call = upload_artifacts.spawn(bucket_uri)
            print(f'upload spawned: {call.object_id} -- safe to disconnect; watch the dashboard', flush=True)
        else:
            print(upload_artifacts.remote(bucket_uri))
        return
    if smoke:
        if bg:
            raise SystemExit('--bg is not supported with --smoke; smoke is an ordered multi-step sequence. Use --run --bg instead.')
        _run_smoke_sequence(gpu, candidate_shards)
        return
    if score_only:
        lm_calls = [
            cpu_job.spawn('oof-lms', fold=fold)
            for fold in FOLDS
        ] + [
            cpu_job.spawn('final-lm', order=order)
            for order in ORDERS
        ]
        _dispatch('language-model job', lm_calls, bg=bg)
        if bg:
            print('score-only spawned in the background; merge/fit/decode tail must run later with --score-only (fg) or check the dashboard.', flush=True)
            return

        shard_calls = [
            cpu_job.spawn('candidate-shard', shard=sid, shards=candidate_shards)
            for sid in range(candidate_shards)
        ]
        _wait('candidate shard', shard_calls)

        print(cpu_job.remote('merge-candidates', shards=candidate_shards))
        print(cpu_job.remote('fit-ranker', shards=candidate_shards))

        decode_calls = [
            cpu_job.spawn('decode-existing-shard', split=split, shard=sid, shards=decode_shards)
            for split in ('val', 'test')
            for sid in range(decode_shards)
        ]
        _wait('decode shard', decode_calls)
        for split in ('val', 'test'):
            print(cpu_job.remote('merge-decodes', split=split, shards=decode_shards))
        print(json.dumps(summary_job.remote(), indent=2, sort_keys=True))
        if upload:
            print(upload_artifacts.remote(bucket_uri))
        return
    if not run:
        print(
            'Use --prepare to download inputs, --smoke to run quick probes, '
            '--prod to apply the frozen pipeline to an external HF/local imagefolder dataset, '
            '--line-only to train/reuse just the SqueezeTrOCR line recognizers, '
            '--upload-prod-result-only to upload a completed prod run, '
            '--run to build charpost artifacts after line checkpoints exist, --full to run the entire pipeline '
            'end-to-end (line recognizers + charpost head + score tail; pair with --bg to survive disconnect, '
            '--upload to also upload the full artifact tree including charpost classifiers/logits/ranker/decodes at the end; '
            '--skip-line-training restores the old full behavior), '
            '--single-gpu-job oof0 to trial one GPU model, '
            '--score-only to run the CPU ridge/decode tail, --summary to print CERs, '
            '--upload to sync all artifacts, '
            'or --status to inspect caches. Add --bg with --run/--score-only/--full to fire-and-forget '
            '(needs `modal run --detach`).'
        )
        return

    if bg:
        _dispatch('prepare', [prepare_inputs.spawn(bucket_uri)], bg=True)
    else:
        print(prepare_inputs.remote(bucket_uri))
    if not skip_smoke and not bg:
        _run_smoke_sequence(gpu, candidate_shards)

    if single_gpu_job:
        _dispatch('image-cache', [cpu_job.spawn('image-cache', workers=image_workers)], bg=bg)
        kind, kwargs = _parse_single_gpu_job(single_gpu_job)
        gpu_fn = gpu_job.with_options(gpu=gpu)
        print(f'single GPU trial: {single_gpu_job} -> {kind} {kwargs}', flush=True)
        _dispatch('single-gpu', [gpu_fn.spawn(kind, **kwargs)], bg=bg)
        if bg:
            print('single-GPU trial spawned in the background; run --status / --summary later.', flush=True)
            return
        print(json.dumps(status_job.remote(), indent=2, sort_keys=True))
        if upload:
            print(upload_artifacts.remote(bucket_uri))
        return

    lm_calls = [
        cpu_job.spawn('oof-lms', fold=fold)
        for fold in FOLDS
    ] + [
        cpu_job.spawn('final-lm', order=order)
        for order in ORDERS
    ]

    _dispatch('image-cache', [cpu_job.spawn('image-cache', workers=image_workers)], bg=bg)

    gpu_fn = gpu_job.with_options(gpu=gpu)
    gpu_calls = [
        gpu_fn.spawn('oof-fold', fold=fold)
        for fold in FOLDS
    ] + [
        gpu_fn.spawn('final'),
    ]

    _dispatch('language-model job', lm_calls, bg=bg)
    _dispatch('gpu job', gpu_calls, bg=bg)
    if bg:
        print('run spawned in the background; candidate/merge/fit/decode tail must run later with --score-only (fg) or check the dashboard.', flush=True)
        return

    shard_calls = [
        cpu_job.spawn('candidate-shard', shard=sid, shards=candidate_shards)
        for sid in range(candidate_shards)
    ]
    _wait('candidate shard', shard_calls)

    print(cpu_job.remote('merge-candidates', shards=candidate_shards))
    print(cpu_job.remote('fit-ranker', shards=candidate_shards))

    decode_calls = [
        cpu_job.spawn('decode', split=split)
        for split in ('val', 'test')
    ]
    _wait('decode job', decode_calls)

    print(json.dumps(status_job.remote(), indent=2, sort_keys=True))
    if upload:
        print(upload_artifacts.remote(bucket_uri))
