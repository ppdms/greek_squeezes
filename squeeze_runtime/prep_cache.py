#!/usr/bin/env python3
"""On-disk memoization for the CPU-heavy image preprocessing — opt-in via PREP_CACHE=1.

The bottleneck before TrOCR training/decoding is the notebook's preprocess(): cv2
illumination-flatten + fastNlMeansDenoising (NLM, the slow part) + CLAHE, run on every full
squeeze image. It is deterministic in the input pixels, so we memoize its output to disk.

Two properties make this safe AND useful:
  * BIT-IDENTICAL: a cache hit returns the exact array cv2 produced (PNG is lossless for
    uint8 grayscale), so the model's input distribution and comparability with the existing
    mono baseline + rerank caches are unchanged. Bump CACHE_VERSION if preprocess() ever
    changes, which invalidates stale entries.
  * Removes only REPEATED work: across build_pairs + the rerank cache step + reruns + the
    mono/dual A/B (which preprocess the same images). The first touch of an image still pays
    the NLM once.

This is the right lever instead of a GPU port: the pip opencv wheels have no CUDA
(cv2.cuda unavailable) and there is no faithful GPU NLM, so a GPU reimplementation would
change the pixels and break the A/B. Keyed by content hash so it is robust to path changes.
"""
import hashlib
import os

import cv2
import numpy as np

CACHE_VERSION = 'v1'        # bump if preprocess() math changes -> old entries ignored


def _key(gray, version):
    h = hashlib.blake2b(digest_size=20)
    h.update(np.ascontiguousarray(gray).tobytes())
    h.update(repr(gray.shape).encode())
    h.update(repr(gray.dtype).encode())
    h.update(version.encode())
    return h.hexdigest()


def _load(stem):
    """Load a cached array (PNG for uint8 grayscale, else .npy). None if absent/corrupt."""
    png = stem + '.png'
    if os.path.exists(png):
        img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    npy = stem + '.npy'
    if os.path.exists(npy):
        try:
            return np.load(npy)
        except Exception:
            return None
    return None


def _save(stem, out):
    """Atomically persist. Lossless PNG for uint8 2D (compact, ~3-5x smaller); else .npy."""
    if isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.ndim == 2:
        tmp = f'{stem}.{os.getpid()}.tmp.png'
        if cv2.imwrite(tmp, out):
            os.replace(tmp, stem + '.png')
            return
        if os.path.exists(tmp):
            os.remove(tmp)
    tmp = f'{stem}.{os.getpid()}.tmp.npy'
    np.save(tmp, out)
    os.replace(tmp, stem + '.npy')


def wrap_preprocess(preprocess_fn, cache_dir, version=CACHE_VERSION):
    """Return a drop-in replacement for preprocess(gray) that memoizes to cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)

    def cached(gray):
        stem = os.path.join(cache_dir, _key(gray, version))
        hit = _load(stem)
        if hit is not None:
            return hit
        out = preprocess_fn(gray)
        try:
            _save(stem, out)
        except Exception:
            pass                       # caching is best-effort; never break the pipeline
        return out

    cached.__wrapped__ = preprocess_fn
    return cached


def maybe_wrap(g, default_dir):
    """If PREP_CACHE=1, wrap g['preprocess'] in place. Returns the cache dir, or None."""
    if os.environ.get('PREP_CACHE', '0') != '1' or 'preprocess' not in g:
        return None
    if getattr(g.get('preprocess'), '__wrapped__', None) is not None:
        return None                    # already wrapped
    cache_dir = os.environ.get('PREP_CACHE_DIR') or default_dir
    g['preprocess'] = wrap_preprocess(g['preprocess'], cache_dir)
    return cache_dir
