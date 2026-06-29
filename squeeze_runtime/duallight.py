#!/usr/bin/env python3
"""Dual-light rot1/rot2 input packing for TrOCR.

Aligned crops are packed as RGB: rot1 in R, rot2 in G, and a configured combiner in B.
If a second crop is unavailable or structurally mismatched, rot1 is duplicated across all
channels so the output remains a normal three-channel TrOCR input.
"""
import os

import numpy as np
from PIL import Image, ImageFilter

# Third-channel combiners for RGB packing. The shipped configuration uses `min`.
VALID_MODES = (
    'mean', 'min', 'max', 'diff', 'diffnorm', 'highpass', 'cavity', 'grad', 'rot1',
    'robustmin', 'cavity2',
)
DEFAULT_MODE = 'mean'

# Training fallback for structurally divergent rows. `both` keeps both rotations as
# separate mono examples; `rot1` keeps only the first rotation.
VALID_DIVERGENT = ('both', 'rot1')
DEFAULT_DIVERGENT = 'both'
VALID_REGISTER_METHODS = ('none', 'phase', 'phase_guarded')
DEFAULT_REGISTER_METHOD = 'phase'


def enabled():
    """True when dual-lighting input is switched on (DUAL_LIGHT=1)."""
    return os.environ.get('DUAL_LIGHT', '0') == '1'


def mode(default=DEFAULT_MODE):
    """The configured B-channel combiner (DUAL_MODE), validated to VALID_MODES."""
    m = os.environ.get('DUAL_MODE', default).strip().lower()
    return m if m in VALID_MODES else default


def divergent_mode(default=DEFAULT_DIVERGENT):
    """How training handles structurally divergent chunks."""
    m = os.environ.get('DUAL_DIVERGENT', default).strip().lower()
    return m if m in VALID_DIVERGENT else default


def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ('', '0', 'false', 'no', 'off')


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def register_enabled(default: bool = False) -> bool:
    """Whether to translate-align crop2 onto crop1 before packing."""
    return _flag('DUAL_REGISTER', default)


def register_method(default: str = DEFAULT_REGISTER_METHOD) -> str:
    """The configured pixel-registration method."""
    m = os.environ.get('DUAL_REGISTER_METHOD', default).strip().lower()
    return m if m in VALID_REGISTER_METHODS else default


def register_max_shift(default: float = 0.15) -> float:
    """Max allowed phase-correlation translation as a fraction of the smaller crop side."""
    return max(0.0, _float_env('DUAL_REGISTER_MAX_SHIFT', default))


def channel_swap_enabled(default: bool = False) -> bool:
    """Whether training randomly swaps the two lighting channels."""
    return _flag('DUAL_CHANNEL_SWAP', default)


def mono_dropout(default: float = 0.0) -> float:
    """Probability that training feeds only one lighting for an otherwise aligned pair."""
    return min(1.0, max(0.0, _float_env('DUAL_MONO_DROPOUT', default)))


# ----------------------------------------------------------------- packing
def _percentile_stretch(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Robustly stretch a grayscale uint8 array to 0..255; fail closed to the input."""
    x = a.astype(np.float32)
    p0, p1 = np.percentile(x, [lo, hi])
    if not np.isfinite([p0, p1]).all() or p1 <= p0 + 1e-6:
        return np.clip(x, 0, 255).astype(np.uint8)
    return np.clip((x - p0) * (255.0 / (p1 - p0)), 0, 255).astype(np.uint8)


def _evidence_to_dark(evidence: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Map non-negative evidence to OCR-style dark strokes (black=high evidence)."""
    hi = float(np.percentile(evidence, percentile))
    if not np.isfinite(hi) or hi <= 1e-6:
        return np.full(evidence.shape, 255, dtype=np.uint8)
    scaled = np.clip(evidence / hi, 0.0, 1.0)
    return np.clip(255.0 * (1.0 - scaled), 0, 255).astype(np.uint8)


def _morph_close(a: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological close with OpenCV when present, PIL fallback otherwise."""
    k = min(21, max(5, int(kernel_size) | 1))
    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.uint8)
    except Exception:
        # PIL MaxFilter then MinFilter is a rectangular close; good enough as a dependency-free fallback.
        return np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k)))


def _combine(a1, a2, m):
    """B-channel from two uint8 arrays of identical shape -> uint8 of that shape."""
    if m == 'rot1':
        return a1
    x1 = a1.astype(np.float32)
    x2 = a2.astype(np.float32)
    base = np.minimum(x1, x2)
    if m == 'min':
        out = base
    elif m == 'robustmin':
        out = np.minimum(_percentile_stretch(a1), _percentile_stretch(a2)).astype(np.float32)
    elif m == 'cavity2':
        # Improved valley detector: robust dark-stroke union first, then black-hat morphology.
        # The output keeps the OCR convention: darker pixels indicate stronger groove evidence.
        robust = np.minimum(_percentile_stretch(a1), _percentile_stretch(a2))
        k = max(5, int(round(min(robust.shape) / 14)) | 1)
        closed = _morph_close(robust, k).astype(np.float32)
        out = _evidence_to_dark(np.maximum(closed - robust.astype(np.float32), 0.0)).astype(np.float32)
    elif m == 'max':
        out = np.maximum(x1, x2)
    elif m == 'diff':
        # Centered relief cue lifted to mid-gray so it survives the processor's
        # symmetric (mean=0.5) normalization instead of clipping to black.
        out = 128.0 + 0.5 * (x1 - x2)
    elif m == 'diffnorm':
        d = x1 - x2
        scale = np.percentile(np.abs(d), 95) or 1.0
        out = 128.0 + 64.0 * np.clip(d / scale, -2.0, 2.0)
    elif m in ('highpass', 'cavity', 'grad'):
        try:
            import cv2
            if m == 'highpass':
                blur = cv2.GaussianBlur(base, (0, 0), sigmaX=5.0, sigmaY=5.0)
                out = 128.0 + 2.0 * (base - blur)
            elif m == 'cavity':
                k = max(3, int(round(min(base.shape) / 18)) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                closed = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.float32)
                out = 255.0 - np.clip(3.0 * (closed - base), 0.0, 255.0)
            else:
                gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx * gx + gy * gy)
                scale = np.percentile(mag, 98) or 1.0
                out = 255.0 * np.clip(mag / scale, 0.0, 1.0)
        except Exception:
            out = base
    else:                                   # 'mean' (default / fallback)
        out = 0.5 * (x1 + x2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _phase_register(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Translate a2 onto a1 with phase correlation; fail closed to the original a2."""
    h, w = a1.shape
    if min(h, w) < 8:
        return a2
    try:
        import cv2
        f1 = a1.astype(np.float32)
        f2 = a2.astype(np.float32)
        s1 = float(f1.std())
        s2 = float(f2.std())
        if s1 < 1e-3 or s2 < 1e-3:
            return a2
        f1 = (f1 - float(f1.mean())) / (s1 + 1e-6)
        f2 = (f2 - float(f2.mean())) / (s2 + 1e-6)
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(f1, f2, win)
        if not np.isfinite([dx, dy, response]).all() or response <= 0:
            return a2
        cap = max(2.0, min(h, w) * register_max_shift())
        if abs(dx) > cap or abs(dy) > cap:
            return a2
        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        return cv2.warpAffine(a2, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return a2


def _registration_distance(a1: np.ndarray, a2: np.ndarray) -> float:
    """Robust distance for deciding whether registration helped.

    Illumination differs between the two lightings, so this is only a guardrail against
    obviously-bad translations rather than a perfect alignment score.
    """
    r1 = _percentile_stretch(a1).astype(np.float32)
    r2 = _percentile_stretch(a2).astype(np.float32)
    diff = np.abs(r1 - r2)
    return float(np.percentile(diff, 75))


def _phase_register_guarded(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Phase-register a2 onto a1 only when a robust similarity score improves."""
    before = _registration_distance(a1, a2)
    candidate = _phase_register(a1, a2)
    after = _registration_distance(a1, candidate)
    min_gain = max(0.0, _float_env('DUAL_REGISTER_GUARD_MIN_GAIN', 0.02))
    # Accept only meaningful improvements; otherwise preserve the resized crop.
    if np.isfinite([before, after]).all() and after <= before * (1.0 - min_gain):
        return candidate
    return a2


def _register(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    if not register_enabled() or register_method() == 'none':
        return a2
    if register_method() == 'phase_guarded':
        return _phase_register_guarded(a1, a2)
    return _phase_register(a1, a2)


def pack(crop1, crop2, m=None):
    """Pack two PIL crops covering the same letters into one RGB image.

    R=first lighting, G=second lighting, B=combiner. crop2 is resized to crop1's size first;
    when DUAL_REGISTER=1 it is then translated onto crop1. If crop2 is None, crop1 is
    duplicated across all channels. The processor handles model-specific resizing."""
    m = m or mode()
    a1 = np.asarray(crop1.convert('L'), dtype=np.uint8)
    if crop2 is None:
        return Image.fromarray(np.stack([a1, a1, a1], axis=-1))   # (H,W,3) uint8 -> RGB (inferred)
    if crop2.size != crop1.size:
        crop2 = crop2.resize(crop1.size, Image.BILINEAR)
    a2 = np.asarray(crop2.convert('L'), dtype=np.uint8)
    a2 = _register(a1, a2)
    b = _combine(a1, a2, m)
    return Image.fromarray(np.stack([a1, a2, b], axis=-1))


# ------------------------------------------------------- row/chunk pairing
def _rows_text_aligned(row1, row2):
    """True when two rows share chunk count and per-chunk text."""
    return (len(row1) == len(row2)
            and all(t1 == t2 for (_, t1), (_, t2) in zip(row1, row2)))


def _new_pairing_stats(rows1, rows2):
    return {
        'rows_total': len(rows1),
        'rows_dual': 0,
        'rows_fallback': 0,
        'chunks_total': sum(len(row) for row in rows1),
        'chunks_dual': 0,
        'chunks_fallback': 0,
        'row_count_match': len(rows1) == len(rows2),
    }


def _finish_pairing_stats(stats):
    stats['row_dual_rate'] = stats['rows_dual'] / max(stats['rows_total'], 1)
    stats['chunk_dual_rate'] = stats['chunks_dual'] / max(stats['chunks_total'], 1)
    return stats


def _sum_pairing_stats(items):
    out = {
        'squeezes': len(items),
        'row_count_mismatch_squeezes': 0,
        'rows_total': 0,
        'rows_dual': 0,
        'rows_fallback': 0,
        'chunks_total': 0,
        'chunks_dual': 0,
        'chunks_fallback': 0,
    }
    for st in items:
        out['row_count_mismatch_squeezes'] += int(not st.get('row_count_match', False))
        for k in ('rows_total', 'rows_dual', 'rows_fallback',
                  'chunks_total', 'chunks_dual', 'chunks_fallback'):
            out[k] += int(st.get(k, 0))
    out['row_dual_rate'] = out['rows_dual'] / max(out['rows_total'], 1)
    out['chunk_dual_rate'] = out['chunks_dual'] / max(out['chunks_total'], 1)
    return out


def pair_rows(rows1, rows2, m=None, return_stats=False):
    """Align two rotations' extract_rows() output and pack each chunk.

    The returned nested row structure matches extract_rows(). Pairing is positional and
    text-checked; mismatched rows fall back to rot1 duplicated across channels."""
    m = m or mode()
    stats = _new_pairing_stats(rows1, rows2)
    if not rows1:
        return ([], _finish_pairing_stats(stats)) if return_stats else []
    if len(rows1) != len(rows2):                 # whole-squeeze mismatch -> mono rot1
        stats['rows_fallback'] = stats['rows_total']
        stats['chunks_fallback'] = stats['chunks_total']
        out = [[(pack(im, None, m), txt) for im, txt in row] for row in rows1]
        return (out, _finish_pairing_stats(stats)) if return_stats else out
    out = []
    for row1, row2 in zip(rows1, rows2):
        if _rows_text_aligned(row1, row2):
            stats['rows_dual'] += 1
            stats['chunks_dual'] += len(row1)
            out.append([(pack(c1, c2, m), t1)
                        for (c1, t1), (c2, _) in zip(row1, row2)])
        else:                                    # per-row fallback to mono rot1
            stats['rows_fallback'] += 1
            stats['chunks_fallback'] += len(row1)
            out.append([(pack(c1, None, m), t1) for c1, t1 in row1])
    return (out, _finish_pairing_stats(stats)) if return_stats else out

def extract_rows_dual(extract_rows, rot1_img, rot2_img, max_chars='default',
                      clean=True, deskew=True, m=None, return_stats=False):
    """Dual-light analogue of extract_rows() for one squeeze.

    `extract_rows` is injected by the notebook runtime. If rot2_img is falsy, rows fall
    back to rot1 duplicated across channels."""
    rows1 = extract_rows(rot1_img, clean=clean, deskew=deskew, max_chars=max_chars)
    if not rot2_img:
        rows2 = []
        out = [[(pack(im, None, m), txt) for im, txt in row] for row in rows1]
        stats = _new_pairing_stats(rows1, rows2)
        stats['rows_fallback'] = stats['rows_total']
        stats['chunks_fallback'] = stats['chunks_total']
        return (out, _finish_pairing_stats(stats)) if return_stats else out
    rows2 = extract_rows(rot2_img, clean=clean, deskew=deskew, max_chars=max_chars)
    return pair_rows(rows1, rows2, m=m, return_stats=return_stats)

def summarize_pairing_stats(stats_items):
    """Aggregate per-squeeze stats from extract_rows_dual(..., return_stats=True)."""
    return _sum_pairing_stats(stats_items)


# ------------------------------------------------------- training pairs
def _mono(row):
    """Single-lighting examples for one row."""
    return [((im, None), txt) for im, txt in row]


def _pairs_for_squeeze(rows1, rows2, divergent=None):
    """Flat [((crop1, crop2_or_None), text), ...] for one squeeze.

    The dataset packs examples after train-time augmentation. Aligned chunks stay paired;
    divergent chunks become single-lighting examples according to DUAL_DIVERGENT."""
    divergent = divergent or divergent_mode()
    use_rot2 = divergent == 'both'
    out = []
    if len(rows1) != len(rows2):                 # whole-squeeze mismatch -> separate monos
        for row in rows1:
            out += _mono(row)
        if use_rot2:
            for row in rows2:
                out += _mono(row)
        return out
    for row1, row2 in zip(rows1, rows2):
        if _rows_text_aligned(row1, row2):
            out.extend(((c1, c2), t1) for (c1, t1), (c2, _) in zip(row1, row2))
        else:                                    # per-row divergence -> separate monos
            out += _mono(row1)
            if use_rot2:
                out += _mono(row2)
    return out


def build_pairs_dual(df_split, extract_rows, m=None):
    """Dual-light analogue of build_pairs() for training.

    Returns flat [((crop1, crop2_or_None), text), ...] pairs across a split.
    Structurally divergent chunks use the DUAL_DIVERGENT fallback policy."""
    del m                                        # combiner is applied at pack time
    pairs = []
    for _, r in df_split.iterrows():
        if not r['rot1_img']:
            continue
        rows1 = extract_rows(r['rot1_img'], clean=True)
        rows2 = extract_rows(r['rot2_img'], clean=True) if r['rot2_img'] else []
        pairs.extend(_pairs_for_squeeze(rows1, rows2))
    return pairs


# --------------------------------------------------- augmentation (locked geometry)
def augment_pair(crop1, crop2, augment, rng, np_mod):
    """Apply identical random augmentation draws to both crops."""
    if crop2 is None:
        return augment(crop1), None
    py_state = rng.getstate()
    np_state = np_mod.random.get_state()
    a1 = augment(crop1)
    rng.setstate(py_state)
    np_mod.random.set_state(np_state)
    a2 = augment(crop2)
    return a1, a2


def training_pair_augments(crop1, crop2, rng, mono_p=0.0, channel_swap=False):
    """Apply train-only channel-role regularizers after geometric augmentation.

    Mono-dropout duplicates one lighting into RGB; channel swap exchanges R/G."""
    if crop2 is None:
        return crop1, None
    if mono_p > 0 and rng.random() < mono_p:
        return (crop2 if rng.random() < 0.5 else crop1), None
    if channel_swap and rng.random() < 0.5:
        return crop2, crop1
    return crop1, crop2


# --------------------------------------------------- training Dataset (lazy torch)
def make_dual_dataset(deps):
    """Build a torch Dataset class for dual-light training pairs."""
    import torch
    from torch.utils.data import Dataset

    processor = deps['processor']
    prep_image = deps['prep_image']
    augment = deps['augment']
    max_len = deps['MAX_LEN']
    rng = deps['rng']
    np_mod = deps['np']
    combiner = deps.get('mode') or mode()
    do_channel_swap = bool(deps.get('channel_swap', channel_swap_enabled()))
    mono_p = float(deps.get('mono_dropout', mono_dropout()))
    pad_id = processor.tokenizer.pad_token_id

    class DualLineDataset(Dataset):
        def __init__(self, pairs, training=False):
            self.pairs = [(im, t) for im, t in pairs if 0 < len(t) <= max_len]
            self.training = training
            self._worker_seeded = False

        def __len__(self):
            return len(self.pairs)

        def _seed_worker_once(self):
            """Keep multi-worker loaders deterministic and worker-distinct."""
            if self._worker_seeded:
                return
            info = torch.utils.data.get_worker_info()
            if info is not None:
                rng.seed(info.seed)
                np_mod.random.seed(info.seed % (2 ** 32))
            self._worker_seeded = True

        def __getitem__(self, i):
            (crop1, crop2), text = self.pairs[i]
            if self.training:
                self._seed_worker_once()
                crop1, crop2 = augment_pair(crop1, crop2, augment, rng, np_mod)
                crop1, crop2 = training_pair_augments(crop1, crop2, rng, mono_p, do_channel_swap)
            rgb = pack(crop1, crop2, combiner)
            pix = processor(images=prep_image(rgb), return_tensors='pt').pixel_values[0]
            ids = processor.tokenizer(text, padding='max_length', truncation=True,
                                      max_length=max_len).input_ids
            ids = [t if t != pad_id else -100 for t in ids]
            return {'pixel_values': pix, 'labels': torch.tensor(ids)}

    return DualLineDataset
