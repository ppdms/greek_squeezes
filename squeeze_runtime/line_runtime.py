#!/usr/bin/env python3
"""Line-recognizer runtime context, as real functions.

This module replaces the old exec-by-substring loader that pulled code cells
out of ``analysis_cells.py`` at runtime. Each section below is a faithful
extraction of one archived research-notebook cell; behavior-affecting details
(env knobs, RNG seeds, defaults, math) are unchanged so the frozen artifacts
remain reproducible. ``rerank.load()`` composes these into the ``g`` context
dict that char_lattice / charpost_pipeline / report_figs / modal_app consume.

Tiling note (pre-existing behavior, preserved deliberately): the default
``max_chars`` of ``extract_rows`` comes from the ``LINE_MAX_CHARS`` env var
(default 8), NOT from the ``tile`` argument of ``rerank.load()``. Callers that
want a specific tiling (charpost tile1 crops, report tile4 figures) pass
``max_chars=...`` explicitly. Modal line training goes through
``duallight.build_pairs_dual``, which does not pass ``max_chars`` and therefore
trains on the env default unless ``LINE_MAX_CHARS`` is set.
"""
from __future__ import annotations

import glob
import math
import os
import random
import re
import warnings
from typing import Any

import contest_evaluation as CE


# ---------------------------------------------------------------------------
# Environment: warnings, device pick. (from the archived env-config cell)
# ---------------------------------------------------------------------------

def setup_environment() -> dict[str, Any]:
    """Quiet third-party logging and pick the compute device.

    Returns DEVICE/DTYPE/USE_FP16/empty_cache, the device context the rest of
    the loader builds on.
    """
    import torch
    from PIL import Image
    from transformers.utils import logging as hf_logging
    from huggingface_hub.utils import logging as hub_logging

    warnings.filterwarnings('ignore', category=UserWarning, module='umap')
    warnings.filterwarnings('ignore', message='.*IProgress not found.*')
    warnings.filterwarnings('ignore', message='.*DecompressionBombWarning.*')
    warnings.filterwarnings('ignore', message='.*Both `max_new_tokens`.*')
    Image.MAX_IMAGE_PIXELS = None  # dataset images are trusted, very large squeeze scans
    hf_logging.set_verbosity_error()
    hub_logging.set_verbosity_error()

    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # CPU fallback for unsupported MPS ops
    if torch.cuda.is_available():
        device, dtype = 'cuda', torch.float16
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device, dtype = 'mps', torch.bfloat16
    else:
        device, dtype = 'cpu', torch.float32
    use_fp16 = (device == 'cuda')  # fp16 mixed-precision training only on CUDA; MPS trains fp32

    def empty_cache():
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()

    return {'DEVICE': device, 'DTYPE': dtype, 'USE_FP16': use_fp16, 'empty_cache': empty_cache}


# ---------------------------------------------------------------------------
# Dataset index + official split. (from the archived shared-setup cell)
# ---------------------------------------------------------------------------

def base_of(name: str) -> str:
    """Physical-squeeze id: strips the _RotationN_300dpi(_letters) suffix."""
    name = os.path.basename(name).replace('.png', '').replace('.txt', '')
    return re.sub(r'_Rotation\d+_300dpi(_letters)?$', '', name)


def build_index(work_dir: str) -> dict[str, Any]:
    """Index every squeeze (both rotation images + annotation) and attach the
    official train/val/test split. Fails loudly on missing split files or
    unassigned squeezes.
    """
    import numpy as np
    import pandas as pd

    # Global seeds, kept for parity with the archived cell (training-order
    # reproducibility of anything downstream that draws from these RNGs).
    random.seed(42)
    np.random.seed(42)

    ann_dir = os.path.join(work_dir, 'Annotations', 'Annotations') + os.sep
    img_dir = os.path.join(work_dir, 'Images') + os.sep

    rows = []
    for ann in sorted(glob.glob(os.path.join(ann_dir, '*_Rotation1_300dpi_letters.txt'))):
        base = base_of(ann)
        r1 = os.path.join(img_dir, f'{base}_Rotation1_300dpi.png')
        r2 = os.path.join(img_dir, f'{base}_Rotation2_300dpi.png')
        if os.path.exists(r1) or os.path.exists(r2):
            rows.append({'base': base, 'ann_path': ann,
                         'rot1_img': r1 if os.path.exists(r1) else None,
                         'rot2_img': r2 if os.path.exists(r2) else None})
    index_df = pd.DataFrame(rows)
    assert len(index_df) > 0, "No squeezes indexed — check ANN_DIR / IMG_DIR paths."

    img_records = []
    for _, r in index_df.iterrows():
        for rot, col in [('rot1', 'rot1_img'), ('rot2', 'rot2_img')]:
            if r[col]:
                img_records.append({'base': r['base'], 'rot': rot, 'img_path': r[col],
                                    'ann_path': r['ann_path']})
    images_df = pd.DataFrame(img_records)

    # The official split is required for reproducibility and comparability.
    split_dir = os.path.dirname(ann_dir.rstrip(os.sep))  # parent of inner Annotations/
    official = {n: os.path.join(split_dir, f'{n}_set.txt')
                for n in ('training', 'validation', 'test')}
    missing = [p for p in official.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Official split files are required for reproducible results. Missing: '
            + ', '.join(missing)
        )
    split: dict[str, str] = {}
    for split_name, fname in (('train', 'training'), ('val', 'validation'), ('test', 'test')):
        with open(official[fname]) as f:
            for line in f:
                b = line.strip()
                if b:
                    split[b] = split_name

    index_df['split'] = index_df['base'].map(split)
    images_df['split'] = images_df['base'].map(split)
    unassigned = index_df[index_df['split'].isna()]['base'].tolist()
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} indexed squeezes are not in the official split; "
            f"first few: {unassigned[:5]}"
        )
    return {
        'index_df': index_df.reset_index(drop=True),
        'images_df': images_df.reset_index(drop=True),
        'ANN_DIR': ann_dir,
        'IMG_DIR': img_dir,
        'SPLIT_SOURCE': 'official (Annotations/{training,validation,test}_set.txt)',
    }


# ---------------------------------------------------------------------------
# Image preprocessing. (from the archived preprocessing cell)
# ---------------------------------------------------------------------------

def to_gray(path: str):
    import cv2
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def preprocess_steps(gray) -> dict[str, Any]:
    """Ordered dict of intermediate images, one per cleaning step."""
    import cv2
    # 1. Illumination flattening: divide out a heavily-blurred background estimate to
    #    cancel the uneven raking-light gradient typical of squeezes.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=41)
    flat = cv2.divide(gray, bg, scale=255).astype('uint8')
    # 2. Edge-preserving denoise to suppress paper-grain speckle.
    den = cv2.fastNlMeansDenoising(flat, None, h=10,
                                   templateWindowSize=7, searchWindowSize=21)
    # 3. Local contrast enhancement (CLAHE) so shallow impressions become legible.
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    return {'1. grayscale (original)': gray, '2. illumination-flattened': flat,
            '3. denoised': den, '4. CLAHE (→ OCR)': enh}


def preprocess(gray):
    """Grayscale cleaning used pipeline-wide (returns the CLAHE output).

    prep_cache.wrap_preprocess memoizes this to disk when PREP_CACHE=1; keep
    prep_cache.CACHE_VERSION in sync if this math ever changes.
    """
    return preprocess_steps(gray)['4. CLAHE (→ OCR)']


def center_crop(img, s: int = 500):
    h, w = img.shape[:2]
    y, x = max(0, (h - s) // 2), max(0, (w - s) // 2)
    return img[y:y + s, x:x + s]


# ---------------------------------------------------------------------------
# Line / sub-line crop extraction. (from the archived extraction cell)
# ---------------------------------------------------------------------------

def _row_bbox(row_boxes, W, H, pad=8):
    xs = [c for b in row_boxes for c in (b.xnw, b.xne, b.xse, b.xsw)]
    ys = [c for b in row_boxes for c in (b.ynw, b.yne, b.yse, b.ysw)]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(W, int(max(xs)) + pad), min(H, int(max(ys)) + pad))


def _norm_angle(a: float) -> float:
    """Normalize a radian angle to (-pi, pi].

    The annotation files sometimes store gangle modulo 2*pi (e.g. 6.149 rad,
    which is really -0.134 rad / -7.7 deg) -- we have to unwrap before using it.
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


def _deskew(gray, boxes, gangle, min_deg=0.3):
    """Rotate `gray` and the corners of every box so the row baseline is horizontal.

    cv2.getRotationMatrix2D uses pixel-space orientation (origin top-left, y axis
    pointing down), so a positive angle visually rotates the image clockwise --
    which is exactly what we need to undo a positive gangle. Empirically: applying
    `+degrees(gangle)` drives the residual top-edge angle of every box to ~0.

    gangle is in radians, as returned by readBoxFile(); we normalize to (-pi, pi]
    first to handle files that store it modulo 2*pi. We use REPLICATE border so the
    deskewed image stays clean around the edges. Box corners are reconstructed with
    the original order (NW, SW, SE, NE) preserved so downstream consumers keep working.
    """
    import cv2
    import numpy as np

    if gangle is None:
        return gray, boxes
    gangle = _norm_angle(gangle)
    if abs(math.degrees(gangle)) < min_deg:
        return gray, boxes
    h, w = gray.shape[:2]
    angle_deg = math.degrees(gangle)  # see docstring re: sign
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    # Expand canvas so corners aren't clipped.
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - w / 2.0
    M[1, 2] += (new_h / 2.0) - h / 2.0
    rotated = cv2.warpAffine(gray, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    box_cls = type(boxes[0])
    out_boxes = []
    for b in boxes:
        pts = np.array([[b.xnw, b.ynw], [b.xsw, b.ysw],
                        [b.xse, b.yse], [b.xne, b.yne]])
        pts_h = np.hstack([pts, np.ones((4, 1))])
        rot = pts_h @ M.T
        out_boxes.append(box_cls(rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1],
                                 rot[2, 0], rot[2, 1], rot[3, 0], rot[3, 1]))
    return rotated, out_boxes


def line_max_chars_from_env() -> int | None:
    """Default letter-box tiling: env LINE_MAX_CHARS, '8' if unset, None to disable."""
    lmc = os.environ.get('LINE_MAX_CHARS', '8')
    return int(lmc) if lmc not in ('', '0') else None


def _balanced_parts(ids, max_chars):
    """Split `ids` into nearly-equal contiguous parts of <= max_chars each, avoiding a
    tiny trailing chunk that could fall below the min-crop-size guard."""
    if not max_chars or max_chars <= 0 or len(ids) <= max_chars:
        return [ids]
    nck = math.ceil(len(ids) / max_chars)
    base, rem, s, parts = len(ids) // nck, len(ids) % nck, 0, []
    for c in range(nck):
        sz = base + (1 if c < rem else 0)
        parts.append(ids[s:s + sz]); s += sz
    return parts


def make_extractors(ann_dir: str, ctx: dict[str, Any]):
    """Build extract_rows/extract_lines bound to one annotation dir.

    `ctx` is the loader's context dict; the cleaning step resolves
    ``ctx['preprocess']`` at call time so prep_cache.maybe_wrap can install its
    on-disk memoization by replacing that single dict entry.

    Returns (extract_rows, extract_lines, default_max_chars).
    """
    from PIL import Image

    default_max_chars = line_max_chars_from_env()

    def _ann_for_image(img_path):
        return os.path.join(ann_dir,
                            os.path.basename(img_path).replace('.png', '_letters.txt'))

    def extract_rows(img_path, clean=True, deskew=True, max_chars='default'):
        """Return [[(PIL 'L' crop, text), ...chunks...], ...rows...] for one squeeze image.

        Uses each image's OWN rotation annotation, so coordinates always match the
        pixels. Reading order matches the official getRowTranscript(); deskew is
        applied afterwards to both the image and the ordered boxes. With max_chars
        set, each row is tiled into <= max_chars-letter chunks.
        """
        if max_chars == 'default':
            max_chars = default_max_chars
        ann = _ann_for_image(img_path)
        if not os.path.exists(ann):
            return []
        gangle, boxes, transcript, lines = CE.readBoxFile(ann)
        if not boxes or not transcript:
            return []
        boxes2 = CE.orderBoxes(boxes, gangle, lines)               # transcript reading order
        rowlist, idlist = CE.getRowBoxes(boxes2, gangle=gangle)    # ignore multicolumn (as official)
        lintran = ''.join(transcript)
        gray = to_gray(img_path)
        if gray is None:
            return []
        if clean:
            gray = ctx['preprocess'](gray)
        if deskew:
            gray, boxes2 = _deskew(gray, boxes2, gangle)
        H, W = gray.shape
        rows = []
        for rowid in idlist:
            ids = [i for i in rowid if i < len(lintran)]
            if not ids:
                continue
            chunks = []
            for sub in _balanced_parts(ids, max_chars):
                text = ''.join(lintran[i] for i in sub).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = _row_bbox([boxes2[i] for i in sub], W, H)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                chunks.append((Image.fromarray(gray[y0:y1, x0:x1]).convert('L'), text))
            if chunks:
                rows.append(chunks)
        return rows

    def extract_lines(img_path, clean=True, deskew=True, max_chars='default'):
        """Flat [(PIL 'L' crop, text), ...] across all rows/chunks."""
        return [c for row in extract_rows(img_path, clean=clean, deskew=deskew,
                                          max_chars=max_chars) for c in row]

    return extract_rows, extract_lines, default_max_chars


# ---------------------------------------------------------------------------
# TrOCR processor + input prep. (from the archived baseline-model cell, minus
# the baseline model itself: no active consumer ever used it, and the one
# caller that exec'd it deleted it immediately to free GPU memory.)
# ---------------------------------------------------------------------------

PROC_SIZE = 384


def build_trocr_context(device: str) -> dict[str, Any]:
    """Load the base TrOCR processor (env TROCR_BASE) and build prep_image.

    prep_image letterboxes when KEEP_ASPECT=1 (opt-in; default is TrOCR's plain
    stretch). Both knobs are read from the environment here, at load time, as
    the archived cell did.
    """
    import numpy as np
    from PIL import Image
    from transformers import TrOCRProcessor

    base_ckpt = os.environ.get('TROCR_BASE', 'microsoft/trocr-base-printed')
    processor = TrOCRProcessor.from_pretrained(base_ckpt)
    keep_aspect = os.environ.get('KEEP_ASPECT', '0') == '1'

    def prep_image(pil, size=PROC_SIZE):
        """RGB size x size image. Letterbox when KEEP_ASPECT, else plain RGB (the
        processor then stretches it, i.e. the original behaviour)."""
        if not keep_aspect:
            return pil.convert('RGB')
        im = pil.convert('L')
        w, h = im.size
        s = size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        im = im.resize((nw, nh), Image.BILINEAR)
        pad = int(np.median(np.asarray(im)))
        canvas = Image.new('L', (size, size), color=pad)
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        return canvas.convert('RGB')

    return {
        'device': device,
        'processor': processor,
        'prep_image': prep_image,
        'BASE_CKPT': base_ckpt,
        'KEEP_ASPECT': keep_aspect,
        'PROC_SIZE': PROC_SIZE,
    }


# ---------------------------------------------------------------------------
# Line-training helpers. (from the archived fine-tune cell; only the pieces
# the Modal line-training job consumes — the dataset class itself comes from
# duallight.make_dual_dataset.)
# ---------------------------------------------------------------------------

MAX_LEN = 64


def make_train_helpers(processor) -> dict[str, Any]:
    """Augmentation + metrics for line training.

    Returns MAX_LEN, a fresh seeded _AUG_RNG, the _augment closure over it, and
    compute_metrics bound to `processor` — the exact surface
    duallight.make_dual_dataset and the Seq2SeqTrainer need.
    """
    import random as _rand

    import cv2
    import numpy as np
    from PIL import Image
    from textdistance import levenshtein

    aug_rng = _rand.Random(42)

    def _augment(im):
        """Light, label-preserving augmentation for ONE PIL 'L' (grayscale) line image.

        Each transform fires independently with low-ish probability; effects compose.
        Designed to mimic the residual noise/skew we actually see on squeezes:
          brightness +/- 10%, contrast +/- 10%, +/- 1.5 deg residual rotation,
          single-pixel erosion or dilation, light gaussian speckle (sigma=2.5).
        """
        arr = np.array(im, dtype=np.uint8)
        # brightness
        if aug_rng.random() < 0.5:
            arr = np.clip(arr.astype(np.int16) +
                          aug_rng.randint(-25, 25), 0, 255).astype(np.uint8)
        # contrast
        if aug_rng.random() < 0.5:
            f = 1.0 + aug_rng.uniform(-0.10, 0.10)
            arr = np.clip((arr.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)
        # tiny residual rotation (deskew was already applied upstream)
        if aug_rng.random() < 0.4:
            h, w = arr.shape
            ang = aug_rng.uniform(-1.5, 1.5)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            arr = cv2.warpAffine(arr, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        # small horizontal shear (heavier geometric augmentation)
        if aug_rng.random() < 0.25:
            h, w = arr.shape
            M = np.float32([[1, aug_rng.uniform(-0.12, 0.12), 0], [0, 1, 0]])
            arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        # morphology -- simulates over- vs under-exposed squeezes
        if aug_rng.random() < 0.3:
            k = np.ones((2, 2), np.uint8)
            arr = (cv2.erode if aug_rng.random() < 0.5 else cv2.dilate)(arr, k, iterations=1)
        # very mild speckle so the model doesn't memorize background grain
        if aug_rng.random() < 0.3:
            noise = np.random.normal(0, 2.5, arr.shape).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # --- Degradation-aware augmentation, opt-in via AUG_LEVEL (0=off) ------
        # Targets the failure physics seen in the error gallery (worn/merged
        # strokes behind Lambda/Alpha crossbars & Theta bars, raking-light
        # shading, chips, resolution loss). TRAIN-only.
        lvl = int(os.environ.get('AUG_LEVEL', '0'))
        if lvl > 0:
            p = 0.30 if lvl == 1 else 0.45
            h, w = arr.shape
            # (1) stroke-width jitter: stronger/variable erode|dilate (worn vs over-pressed strokes)
            if aug_rng.random() < p:
                ks = aug_rng.choice([2, 3] if lvl == 1 else [2, 3, 3])
                k2 = np.ones((ks, ks), np.uint8)
                arr = (cv2.erode if aug_rng.random() < 0.5 else cv2.dilate)(
                    arr, k2, iterations=aug_rng.randint(1, 2))
            # (2) elastic / local warp: non-rigid paper-cast stretch (classic HTR aug)
            if aug_rng.random() < p:
                alpha = 4.0 if lvl == 1 else 7.0
                dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), 6.0)
                dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), 6.0)
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                arr = cv2.remap(arr, (xx + alpha * dx).astype(np.float32),
                                (yy + alpha * dy).astype(np.float32),
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # (3) raking-light shading: random directional brightness gradient
            if aug_rng.random() < p:
                amp = 0.25 if lvl == 1 else 0.40
                ang = aug_rng.uniform(0, 6.2831853)
                gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
                grad = np.cos(ang) * gx + np.sin(ang) * gy
                grad = grad / (np.abs(grad).max() + 1e-6)
                arr = np.clip(arr.astype(np.float32) * (1.0 + amp * grad), 0, 255).astype(np.uint8)
            # (4) cutout: chips / missing relief (fill with median tone)
            if aug_rng.random() < p:
                med = int(np.median(arr))
                for _ in range(aug_rng.randint(1, 2 if lvl == 1 else 3)):
                    cw = aug_rng.randint(max(2, w // 12), max(3, w // 6))
                    ch = aug_rng.randint(max(2, h // 6), max(3, h // 3))
                    x0 = aug_rng.randint(0, max(1, w - cw)); y0 = aug_rng.randint(0, max(1, h - ch))
                    arr[y0:y0 + ch, x0:x0 + cw] = med
            # (5) resolution jitter: focus/scan blur via down- then up-sample
            if aug_rng.random() < p:
                f = aug_rng.uniform(0.5 if lvl == 2 else 0.65, 0.9)
                small = cv2.resize(arr, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
                arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(arr, mode='L')

    def compute_metrics(pred):
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        preds = np.asarray(preds)
        if preds.ndim != 2:
            raise ValueError(f'Expected generated token ids, got predictions with shape {preds.shape}')
        if np.any((preds < 0) | (preds >= len(processor.tokenizer))):
            raise ValueError('Generated predictions contain token ids outside the tokenizer vocabulary')
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(preds.astype(np.int64), skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        n_err = sum(levenshtein(p.upper(), l.upper()) for p, l in zip(pred_str, label_str))
        n_chr = sum(len(l) for l in label_str) or 1
        return {'cer': n_err / n_chr}

    return {
        'MAX_LEN': MAX_LEN,
        '_AUG_RNG': aug_rng,
        '_augment': _augment,
        'compute_metrics': compute_metrics,
    }
