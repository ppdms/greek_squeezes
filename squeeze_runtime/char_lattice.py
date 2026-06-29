#!/usr/bin/env python3
"""Character-posterior lattice scaffolding for tile1 OCR experiments.

This script implements the character-posterior path:

* inspect the tile1 dual-light character dataset that a classifier would train on;
* train fold-aware train-only character n-gram LMs over official-train rows;
* run a toy lattice self-check for the visual-posterior + LM beam decoder.

Future subcommands should add the visual encoder classifier and posterior caching under
``data/character_posterior/<tag>__{split}.pkl``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank as R  # noqa: E402

WORK = Path(os.environ.get("SQUEEZE_WORK_DIR", "data"))
CHARPOST_DIR = WORK / "character_posterior"
DEFAULT_FOLDS = WORK / "splits" / "oof_folds.json"
# The notebook's in-domain CharNGramLM reports vocab=27 on this dataset: the 24 proxy letters
# plus three non-letter symbols that appear in the row transcripts.
DEFAULT_ALPHABET = "ABCDEFGHIKLMNOPQRSTUWXYZ&:)"


def logits_path(tag: str, split: str) -> Path:
    return CHARPOST_DIR / f"{tag}__{split}_logits.npz"


def log(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: Any, **json_kwargs: Any) -> None:
    atomic_write_text(path, json.dumps(obj, **json_kwargs), encoding="utf-8")


def atomic_np_savez_compressed(path: Path, *args: Any, **kwargs: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, *args, **kwargs)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_torch_save(obj: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_dir_backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.backup")


def recover_atomic_dir(path: Path) -> None:
    backup = atomic_dir_backup_path(path)
    if path.exists():
        if backup.exists():
            shutil.rmtree(backup)
        return
    if backup.exists():
        backup.rename(path)


def replace_dir_with_backup(src: Path, dst: Path) -> None:
    backup = atomic_dir_backup_path(dst)
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


def load_helpers(tile: int = 1) -> dict[str, Any]:
    return R.load(tile=tile, load_model=False, verbose=True)


def gt_text(g: dict[str, Any], ann_path: str, cache: dict[str, str]) -> str:
    if ann_path not in cache:
        cache[ann_path] = g["getRowTranscript"](ann_path).upper()
    return cache[ann_path]


def split_rows(g: dict[str, Any], splits: Iterable[str], exclude_bases: set[str] | None = None) -> list[str]:
    exclude = exclude_bases or set()
    wanted = set(splits)
    rows: list[str] = []
    gt_cache: dict[str, str] = {}
    for _, row in g["index_df"][g["index_df"]["split"].isin(wanted)].iterrows():
        if row["base"] in exclude:
            continue
        text = gt_text(g, row["ann_path"], gt_cache)
        rows.extend(line for line in text.split("\n") if line)
    return rows


def load_fold_bases(path: Path, fold_id: int) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    folds = data.get("folds", [])
    if isinstance(folds, dict):
        return set(folds[str(fold_id)])
    for fold in folds:
        if int(fold.get("id")) == int(fold_id):
            return set(fold.get("bases", []))
    raise ValueError(f"fold {fold_id} not found in {path}")


def keep_chars(text: str, alphabet: str) -> str:
    allowed = set(alphabet)
    return "".join(ch for ch in text.upper() if ch in allowed)


def safe_name(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "none"


def iter_char_examples(
    g: dict[str, Any],
    split: str,
    tile: int,
    alphabet: str,
    include_bases: set[str] | None = None,
    exclude_bases: set[str] | None = None,
) -> Iterable[dict[str, Any]]:
    extract_rows = g["extract_rows"]
    allowed = set(alphabet)
    for _, row in g["index_df"][g["index_df"]["split"] == split].iterrows():
        base = str(row["base"])
        if include_bases is not None and base not in include_bases:
            continue
        if exclude_bases is not None and base in exclude_bases:
            continue
        rows1 = extract_rows(row["rot1_img"], clean=True, max_chars=tile) if row["rot1_img"] else []
        rows2 = extract_rows(row["rot2_img"], clean=True, max_chars=tile) if row["rot2_img"] else []
        aligned_rows = len(rows1) == len(rows2)
        for row_idx, row1 in enumerate(rows1):
            row_text = "".join(text for _, text in row1)
            row2 = rows2[row_idx] if aligned_rows and row_idx < len(rows2) else []
            aligned_chunks = aligned_rows and len(row1) == len(row2)
            for chunk_idx, (crop1, text) in enumerate(row1):
                label = keep_chars(text, alphabet)
                if len(label) != 1 or label not in allowed:
                    continue
                crop2 = None
                chunk_aligned = False
                if aligned_chunks and chunk_idx < len(row2) and row2[chunk_idx][1] == text:
                    crop2 = row2[chunk_idx][0]
                    chunk_aligned = True
                yield {
                    "crop1": crop1,
                    "crop2": crop2,
                    "label": label,
                    "split": split,
                    "base": base,
                    "rot1_img": row["rot1_img"],
                    "rot2_img": row.get("rot2_img"),
                    "row_idx": int(row_idx),
                    "chunk_idx": int(chunk_idx),
                    "row_text": row_text,
                    "chunk_aligned": chunk_aligned,
                }


class CharPostNGramLM:
    """Small interpolated char n-gram LM over an explicit charpost alphabet."""

    def __init__(self, order: int = 10, lam: float = 0.4, alphabet: str = DEFAULT_ALPHABET) -> None:
        self.order = int(order)
        self.lam = float(lam)
        self.alphabet = alphabet
        self.vocab = set(alphabet)
        self.counts: list[dict[str, Counter[str]]] = [dict() for _ in range(self.order + 1)]
        self.totals: list[dict[str, int]] = [dict() for _ in range(self.order + 1)]

    def train(self, lines: Iterable[str]) -> "CharPostNGramLM":
        from collections import defaultdict

        counts: list[defaultdict[str, Counter[str]]] = [defaultdict(Counter) for _ in range(self.order + 1)]
        for raw in lines:
            line = keep_chars(raw, self.alphabet)
            if not line:
                continue
            for i, ch in enumerate(line):
                max_order = min(self.order, i + 1)
                for m in range(1, max_order + 1):
                    ctx = line[i - (m - 1):i] if m > 1 else ""
                    counts[m][ctx][ch] += 1
        self.counts = [dict(c) for c in counts]
        self.totals = [{ctx: sum(nexts.values()) for ctx, nexts in self.counts[m].items()}
                       for m in range(self.order + 1)]
        return self

    def cond_logprob(self, ch: str, context: str) -> float:
        if ch not in self.vocab:
            return math.log(1e-12)
        p = 1.0 / max(len(self.vocab), 1)
        for m in range(1, self.order + 1):
            ctx = keep_chars(context, self.alphabet)[-(m - 1):] if m > 1 else ""
            cnt = self.counts[m].get(ctx)
            total = self.totals[m].get(ctx, 0)
            ph = (cnt.get(ch, 0) / total) if cnt and total else 0.0
            p = self.lam * ph + (1.0 - self.lam) * p
        return math.log(max(p, 1e-12))

    def mean_logprob(self, text: str) -> float:
        text = keep_chars(text, self.alphabet)
        if not text:
            return 0.0
        return sum(self.cond_logprob(text[i], text[:i]) for i in range(len(text))) / len(text)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "CharPostNGramLM":
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                # Older artifacts were written by running char_lattice.py as __main__,
                # so pickle recorded __main__.CharPostNGramLM. Remap that class when loading
                # from imported modules.
                if module == "__main__" and name == "CharPostNGramLM":
                    return CharPostNGramLM
                return super().find_class(module, name)

        with open(path, "rb") as f:
            obj = _CompatUnpickler(f).load()
        if not hasattr(obj, "mean_logprob"):
            raise TypeError(f"{path} is not a CharPostNGramLM artifact")
        return obj


def train_char_lm(lines: Iterable[str], order: int, smooth_lambda: float,
                  alphabet: str = DEFAULT_ALPHABET) -> CharPostNGramLM:
    return CharPostNGramLM(order=order, lam=smooth_lambda, alphabet=alphabet).train(lines)


def write_lm_artifact(lm: CharPostNGramLM, out_dir: Path, manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"charpost_train_order{lm.order}.pkl"
    model_path = out_dir / model_name
    lm.save(str(model_path))
    atomic_write_json(out_dir / "lm.json", {**manifest, "backend": "charpost_python", "model": model_name}, indent=2, sort_keys=True)


def load_charpost_lm(path: str) -> Any:
    path_obj = Path(path)
    if path_obj.is_dir():
        path_obj = path_obj / "lm.json"
    if path_obj.suffix == ".json":
        manifest = json.loads(path_obj.read_text(encoding="utf-8"))
        if manifest.get("backend") != "charpost_python":
            raise ValueError(f"unsupported charpost LM backend in {path_obj}: {manifest.get('backend')!r}")
        model_path = Path(str(manifest["model"]))
        if not model_path.is_absolute():
            model_path = path_obj.parent / model_path
        return CharPostNGramLM.load(str(model_path))
    return CharPostNGramLM.load(str(path_obj))


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def command_train_lm(args: argparse.Namespace) -> None:
    g = load_helpers(tile=args.tile)
    exclude: set[str] = set()
    if args.exclude_fold_id is not None:
        exclude.update(load_fold_bases(Path(args.folds), args.exclude_fold_id))
    rows = split_rows(g, args.splits.split(","), exclude)
    lm = train_char_lm(rows, args.order, args.smooth_lambda, args.alphabet)
    manifest = {
        "created_by": "squeeze_runtime/char_lattice.py train-lm",
        "order": args.order,
        "smooth_lambda": args.smooth_lambda,
        "splits": args.splits,
        "excluded_bases": sorted(exclude),
        "train_rows": len(rows),
        "train_chars": sum(len(keep_chars(row, args.alphabet)) for row in rows),
        "alphabet": args.alphabet,
        "note": "Train-only char LM for tile1 character-posterior lattice decoding.",
    }
    write_lm_artifact(lm, Path(args.out_dir), manifest)
    print(
        f"train-lm wrote {Path(args.out_dir) / 'lm.json'} "
        f"order={args.order} rows={manifest['train_rows']} chars={manifest['train_chars']} "
        f"excluded_bases={len(exclude)}",
        flush=True,
    )


INPUT_MODES = {"dual_pack"}
POOLING_MODES = ("cls", "mean", "cls_mean_max", "spatial_pyramid", "attn")
TTA_MODES = ("none", "shift5", "scale3", "shift5_scale3")
MULTIVIEW_FUSE_MODES = ("concat", "mean")


def pooling_dim(hidden_size: int, pooling: str) -> int:
    if pooling == "cls_mean_max":
        return hidden_size * 3
    if pooling == "spatial_pyramid":
        # CLS + global patch mean + 2x2 patch-region means.
        return hidden_size * 6
    return hidden_size


def pool_encoder_output(last_hidden_state: Any, pooling: str) -> Any:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    patches = last_hidden_state[:, 1:, :]
    if pooling == "mean":
        return patches.mean(dim=1)
    if pooling == "cls_mean_max":
        return torch_cat_pool(last_hidden_state)
    if pooling == "spatial_pyramid":
        return torch_spatial_pyramid_pool(last_hidden_state)
    if pooling == "attn":
        raise ValueError("attn pooling is handled by AttentionPoolHead")
    raise ValueError(f"unknown pooling={pooling}")


def torch_cat_pool(last_hidden_state: Any) -> Any:
    import torch

    patches = last_hidden_state[:, 1:, :]
    return torch.cat([last_hidden_state[:, 0, :], patches.mean(dim=1), patches.max(dim=1).values], dim=1)


def torch_spatial_pyramid_pool(last_hidden_state: Any) -> Any:
    import math as _math
    import torch

    patches = last_hidden_state[:, 1:, :]
    bsz, patch_count, hidden = patches.shape
    side = int(round(_math.sqrt(int(patch_count))))
    if side * side != int(patch_count):
        return torch.cat([last_hidden_state[:, 0, :], patches.mean(dim=1).repeat(1, 5)], dim=1)
    grid = patches.reshape(bsz, side, side, hidden)
    global_mean = patches.mean(dim=1)
    cells: list[Any] = []
    for y0, y1 in ((0, side // 2), (side // 2, side)):
        for x0, x1 in ((0, side // 2), (side // 2, side)):
            cell = grid[:, y0:y1, x0:x1, :].reshape(bsz, -1, hidden)
            cells.append(cell.mean(dim=1))
    return torch.cat([last_hidden_state[:, 0, :], global_mean, *cells], dim=1)


class AttentionPoolHead:
    """Factory for a trainable attention-pooling classifier over ViT patch tokens."""

    def __new__(cls, hidden_size: int, n_classes: int, use_mlp: bool = False,
                hidden: int = 0, dropout: float = 0.1) -> Any:
        import math as _math
        import torch

        class _AttentionPoolHead(torch.nn.Module):
            takes_hidden_state = True

            def __init__(self) -> None:
                super().__init__()
                self.query = torch.nn.Parameter(torch.zeros(hidden_size))
                self.norm = torch.nn.LayerNorm(hidden_size)
                torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
                self.classifier = build_classifier_head(
                    hidden_size,
                    n_classes,
                    use_mlp=use_mlp,
                    hidden=hidden,
                    dropout=dropout,
                    pooling="cls",
                )

            def forward(self, last_hidden_state: Any) -> Any:
                patches = self.norm(last_hidden_state[:, 1:, :])
                scores = (patches * self.query).sum(dim=-1) / _math.sqrt(float(hidden_size))
                weights = torch.softmax(scores, dim=-1)
                pooled = (patches * weights.unsqueeze(-1)).sum(dim=1)
                return self.classifier(pooled)

        return _AttentionPoolHead()


def build_classifier_head(in_dim: int, n_classes: int, use_mlp: bool = False,
                          hidden: int = 0, dropout: float = 0.1,
                          pooling: str = "cls") -> Any:
    import torch

    if pooling == "attn":
        return AttentionPoolHead(in_dim, n_classes, use_mlp=use_mlp, hidden=hidden, dropout=dropout)
    if not use_mlp:
        return torch.nn.Linear(in_dim, n_classes)
    hidden_dim = int(hidden) if hidden and hidden > 0 else max(256, min(2048, in_dim))
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(float(dropout)),
        torch.nn.Linear(hidden_dim, n_classes),
    )


def image_for_char_example(
    item: dict[str, Any],
    helpers: dict[str, Any] | None,
    tile: int,
    input_mode: str,
    dual_mode: str,
    canonical_cache: dict[str, tuple[Any, list[Any], list[list[int]], str]] | None = None,
    image_policy: dict[str, Any] | None = None,
) -> Any:
    import duallight as DL

    del helpers, tile, canonical_cache, image_policy
    if input_mode != "dual_pack":
        raise ValueError(f"unknown input_mode={input_mode}")
    return DL.pack(item["crop1"], item["crop2"], dual_mode)


def image_policy_for_args(args: Any, input_mode: str | None = None) -> dict[str, Any]:
    del args, input_mode
    return {}


def image_policy_cache_key(input_mode: str, dual_mode: str, image_policy: dict[str, Any] | None = None) -> str:
    del image_policy
    if input_mode != "dual_pack":
        raise ValueError(f"unknown input_mode={input_mode}")
    base = f"{input_mode}__{dual_mode}"
    return base


def char_image_cache_path(
    root: str | Path,
    item: dict[str, Any],
    input_mode: str,
    dual_mode: str,
    image_policy: dict[str, Any] | None = None,
    view_idx: int | None = None,
) -> Path:
    base = safe_name(str(item["base"]))
    split = safe_name(str(item.get("split", "split")))
    suffix = "" if view_idx is None else f"__v{int(view_idx)}"
    name = f"{split}__{base}__r{int(item['row_idx']):03d}__c{int(item['chunk_idx']):03d}{suffix}.png"
    return Path(root) / image_policy_cache_key(input_mode, dual_mode, image_policy) / name


def cached_example_index_path(
    root: str | Path,
    input_mode: str,
    dual_mode: str,
    image_policy: dict[str, Any] | None = None,
) -> Path:
    return Path(root) / image_policy_cache_key(input_mode, dual_mode, image_policy) / "examples.jsonl"


def cached_example_record(item: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "split": str(item.get("split", "")),
        "base": str(item.get("base", "")),
        "rot1_img": str(item.get("rot1_img") or ""),
        "rot2_img": str(item.get("rot2_img") or ""),
        "row_idx": int(item["row_idx"]),
        "chunk_idx": int(item["chunk_idx"]),
        "row_text": str(item.get("row_text", "")),
        "label": str(item.get("label", "")),
        "chunk_aligned": bool(item.get("chunk_aligned", False)),
    }
    if "base_pos" in item:
        rec["base_pos"] = int(item["base_pos"])
    return rec


def cached_example_sort_key(split_order: dict[str, int], rec: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        split_order.get(str(rec.get("split", "")), 10_000),
        int(rec.get("base_pos", 0)),
        int(rec.get("row_idx", 0)),
        int(rec.get("chunk_idx", 0)),
        str(rec.get("base", "")),
    )


def write_cached_example_index(args: argparse.Namespace, examples: list[dict[str, Any]]) -> Path:
    image_policy = image_policy_for_args(args)
    path = cached_example_index_path(args.image_cache_dir, args.input_mode, args.dual_mode, image_policy)
    split_order = {split: idx for idx, split in enumerate(part.strip() for part in args.splits.split(",") if part.strip())}
    records = sorted((cached_example_record(item) for item in examples), key=lambda rec: cached_example_sort_key(split_order, rec))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "".join(json.dumps(rec, sort_keys=True) + "\n" for rec in records))
    return path


def load_cached_example_index(
    image_cache_dir: str | Path,
    input_mode: str,
    dual_mode: str,
    image_policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    path = cached_example_index_path(image_cache_dir, input_mode, dual_mode, image_policy)
    if not path.exists():
        raise FileNotFoundError(
            f"missing cached example index: {path}\n"
            "Rebuild the image cache on CPU so GPU training/logit jobs do not re-extract crops."
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                rec["row_idx"] = int(rec["row_idx"])
                rec["chunk_idx"] = int(rec["chunk_idx"])
                if "base_pos" in rec:
                    rec["base_pos"] = int(rec["base_pos"])
            except Exception as exc:
                raise ValueError(f"bad cached example index row {line_no} in {path}: {exc}") from exc
            records.append(rec)
    if not records:
        raise ValueError(f"cached example index is empty: {path}")
    return records, path


def filter_cached_examples(
    records: list[dict[str, Any]],
    split: str,
    alphabet: str,
    limit: int = 0,
    include_bases: set[str] | None = None,
    exclude_bases: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = set(alphabet)
    out: list[dict[str, Any]] = []
    for rec in records:
        if str(rec.get("split")) != split:
            continue
        base = str(rec.get("base", ""))
        if include_bases is not None and base not in include_bases:
            continue
        if exclude_bases is not None and base in exclude_bases:
            continue
        label = str(rec.get("label", ""))
        if len(label) != 1 or label not in allowed:
            continue
        out.append(dict(rec))
        if limit > 0 and len(out) >= limit:
            break
    return out


def cached_image_for_char_example(
    item: dict[str, Any],
    helpers: dict[str, Any] | None,
    tile: int,
    input_mode: str,
    dual_mode: str,
    canonical_cache: dict[str, tuple[Any, list[Any], list[list[int]], str]] | None,
    image_cache_dir: str | Path,
    require_cached: bool = False,
    image_policy: dict[str, Any] | None = None,
) -> Any:
    from PIL import Image

    if not image_cache_dir:
        if require_cached:
            raise FileNotFoundError("image cache required but --image-cache-dir is empty")
        return image_for_char_example(item, helpers, tile, input_mode, dual_mode, canonical_cache, image_policy)
    path = char_image_cache_path(image_cache_dir, item, input_mode, dual_mode, image_policy)
    if path.exists():
        return Image.open(path).convert("RGB")
    if require_cached:
        raise FileNotFoundError(f"missing cached image: {path}")
    img = image_for_char_example(item, helpers, tile, input_mode, dual_mode, canonical_cache, image_policy).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img


def cached_visual_inputs_for_char_example(
    item: dict[str, Any],
    helpers: dict[str, Any] | None,
    tile: int,
    input_mode: str,
    dual_mode: str,
    canonical_cache: dict[str, tuple[Any, list[Any], list[list[int]], str]] | None,
    image_cache_dir: str | Path,
    require_cached: bool = False,
    image_policy: dict[str, Any] | None = None,
) -> Any:
    return cached_image_for_char_example(
        item, helpers, tile, input_mode, dual_mode, canonical_cache,
        image_cache_dir, require_cached, image_policy,
    )


def median_fill_rgb(img: Any) -> tuple[int, int, int]:
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    med = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(x) for x in med)


def affine_image(img: Any, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> Any:
    from PIL import Image

    img = img.convert("RGB")
    inv = 1.0 / max(float(scale), 1e-6)
    # PIL affine parameters map output -> input coordinates.
    matrix = (inv, 0.0, -float(dx) * inv, 0.0, inv, -float(dy) * inv)
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BILINEAR,
        fillcolor=median_fill_rgb(img),
    )


def augment_image(img: Any, args: Any) -> Any:
    import random
    import numpy as np
    from PIL import ImageEnhance, ImageFilter, Image

    img = img.convert("RGB")
    jitter_px = int(getattr(args, "aug_jitter_px", 0) or 0)
    scale_jitter = float(getattr(args, "aug_scale", 0.0) or 0.0)
    if jitter_px > 0 or scale_jitter > 0:
        dx = random.uniform(-jitter_px, jitter_px) if jitter_px > 0 else 0.0
        dy = random.uniform(-jitter_px, jitter_px) if jitter_px > 0 else 0.0
        scale = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter) if scale_jitter > 0 else 1.0
        img = affine_image(img, dx=dx, dy=dy, scale=scale)

    brightness = float(getattr(args, "aug_brightness", 0.0) or 0.0)
    if brightness > 0:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - brightness, 1.0 + brightness))
    contrast = float(getattr(args, "aug_contrast", 0.0) or 0.0)
    if contrast > 0:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - contrast, 1.0 + contrast))

    if random.random() < float(getattr(args, "aug_blur_prob", 0.0) or 0.0):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    if random.random() < float(getattr(args, "aug_sharpen_prob", 0.0) or 0.0):
        img = img.filter(ImageFilter.SHARPEN)

    arr = np.asarray(img, dtype=np.float32)
    noise = float(getattr(args, "aug_noise_std", 0.0) or 0.0)
    if noise > 0:
        arr = arr + np.random.normal(0.0, noise, size=arr.shape).astype(np.float32)

    channel_dropout = float(getattr(args, "aug_channel_dropout", 0.0) or 0.0)
    if channel_dropout > 0 and random.random() < channel_dropout:
        channel = random.randrange(3)
        arr[:, :, channel] = np.median(arr[:, :, channel])

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def tta_specs(mode: str, pixels: int = 4, scale: float = 0.04) -> list[tuple[float, float, float]]:
    if mode == "none":
        return [(0.0, 0.0, 1.0)]
    specs: list[tuple[float, float, float]] = []
    if mode in ("shift5", "shift5_scale3"):
        p = float(pixels)
        specs.extend([(0.0, 0.0, 1.0), (-p, 0.0, 1.0), (p, 0.0, 1.0), (0.0, -p, 1.0), (0.0, p, 1.0)])
    if mode in ("scale3", "shift5_scale3"):
        s = float(scale)
        specs.extend([(0.0, 0.0, 1.0 - s), (0.0, 0.0, 1.0 + s)])
    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(specs))


def apply_tta_image(img: Any, spec: tuple[float, float, float]) -> Any:
    dx, dy, scale = spec
    return affine_image(img, dx=dx, dy=dy, scale=scale)


def processor_kwargs_for_size(input_size: int) -> dict[str, Any]:
    if not input_size or input_size <= 0:
        return {}
    return {"size": {"height": int(input_size), "width": int(input_size)}}


def process_visual_inputs(processor: Any, prep_image: Any, visual: Any, input_size: int = 0) -> Any:
    import torch
    from PIL import Image

    kwargs = processor_kwargs_for_size(input_size)
    def call_processor(images: Any) -> Any:
        try:
            return processor(images=images, return_tensors="pt", **kwargs).pixel_values
        except TypeError:
            if not kwargs:
                raise
            if isinstance(images, list):
                resized = [img.resize((int(input_size), int(input_size)), Image.Resampling.BILINEAR) for img in images]
            else:
                resized = images.resize((int(input_size), int(input_size)), Image.Resampling.BILINEAR)
            return processor(images=resized, return_tensors="pt").pixel_values

    if isinstance(visual, list):
        prepared = [prep_image(img) for img in visual]
        return call_processor(prepared)
    return call_processor(prep_image(visual))[0]


def make_default_prep_image(input_size: int = 0) -> Any:
    """Local equivalent of the notebook prep_image used by charpost classifier cache-logits.

    Classifier checkpoints save their processor, so cache-logits should not need to load the
    full TrOCR baseline path just to recover a one-line image prep function.
    """
    def prep_image(pil: Any, size: int = input_size or 384) -> Any:
        import numpy as np
        from PIL import Image

        if os.environ.get("KEEP_ASPECT", "0") != "1":
            return pil.convert("RGB")
        im = pil.convert("L")
        w, h = im.size
        scale = size / max(w, h, 1)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        im = im.resize((nw, nh), Image.BILINEAR)
        pad = int(np.median(np.asarray(im)))
        canvas = Image.new("L", (size, size), color=pad)
        canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
        return canvas.convert("RGB")

    return prep_image


class CharCropDataset:
    """Lazy tile1 crop dataset; imports torch only when instantiated."""

    def __init__(self, examples: list[dict[str, Any]], processor: Any, prep_image: Any,
                 label_to_id: dict[str, int], dual_mode: str = "min", input_mode: str = "dual_pack",
                 helpers: dict[str, Any] | None = None, tile: int = 1,
                 augment: bool = False, augment_args: Any | None = None,
                 image_cache_dir: str | Path = "", require_image_cache: bool = False,
                 image_policy: dict[str, Any] | None = None,
                 input_size: int = 0,
                 teacher_logits: Any | None = None) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self) -> None:
                self.examples = examples
                self.canonical_cache: dict[str, tuple[Any, list[Any], list[list[int]], str]] = {}

            def __len__(self) -> int:
                return len(self.examples)

            def __getitem__(self, index: int) -> dict[str, Any]:
                item = self.examples[index]
                visual = cached_visual_inputs_for_char_example(
                    item,
                    helpers,
                    tile,
                    input_mode,
                    dual_mode,
                    self.canonical_cache,
                    image_cache_dir,
                    require_image_cache,
                    image_policy,
                )
                if augment and augment_args is not None:
                    if isinstance(visual, list):
                        visual = [augment_image(img, augment_args) for img in visual]
                    else:
                        visual = augment_image(visual, augment_args)
                pixel_values = process_visual_inputs(processor, prep_image, visual, input_size)
                label = torch.tensor(label_to_id[item["label"]], dtype=torch.long)
                out = {"pixel_values": pixel_values, "labels": label}
                if teacher_logits is not None:
                    out["teacher_logits"] = torch.tensor(teacher_logits[index], dtype=torch.float32)
                return out

        self.dataset = _Dataset()


def classifier_examples(
    g: dict[str, Any],
    split: str,
    tile: int,
    alphabet: str,
    limit: int = 0,
    include_bases: set[str] | None = None,
    exclude_bases: set[str] | None = None,
) -> list[dict[str, Any]]:
    examples = []
    for item in iter_char_examples(g, split, tile, alphabet, include_bases, exclude_bases):
        examples.append(item)
        if limit > 0 and len(examples) >= limit:
            break
    return examples


def fold_bases_for_args(args: argparse.Namespace, attr: str) -> set[str] | None:
    fold_id = getattr(args, attr, None)
    if fold_id is None:
        return None
    return load_fold_bases(Path(getattr(args, "folds", DEFAULT_FOLDS)), int(fold_id))


_CACHE_IMAGE_WORKER: dict[str, Any] = {}


def _configure_cache_image_worker_loader(config: dict[str, Any]) -> None:
    global WORK, CHARPOST_DIR, DEFAULT_FOLDS

    work = Path(config.get("work") or WORK)
    WORK = work
    CHARPOST_DIR = Path(config.get("charpost_dir") or CHARPOST_DIR)
    DEFAULT_FOLDS = Path(config.get("default_folds") or DEFAULT_FOLDS)
    R.WORK = str(work.resolve())


def _cache_images_worker_init(config: dict[str, Any]) -> None:
    # Each worker owns its helper state and per-image canonical cache. The helper
    # object contains notebook-defined functions and is intentionally not pickled
    # from the parent process.
    _configure_cache_image_worker_loader(config)
    g = R.load(ckpt=config["init_ckpt"], tile=int(config["tile"]), load_model=False, verbose=False)
    _CACHE_IMAGE_WORKER.clear()
    _CACHE_IMAGE_WORKER.update({"config": config, "helpers": g, "canonical_cache": {}})


def _cache_images_for_base(task: dict[str, Any]) -> dict[str, Any]:
    cfg = _CACHE_IMAGE_WORKER["config"]
    g = _CACHE_IMAGE_WORKER["helpers"]
    canonical_cache = _CACHE_IMAGE_WORKER["canonical_cache"]
    extract_rows = g["extract_rows"]
    alphabet = str(cfg["alphabet"])
    allowed = set(alphabet)
    tile = int(cfg["tile"])
    input_mode = str(cfg["input_mode"])
    dual_mode = str(cfg["dual_mode"])
    image_cache_dir = str(cfg["image_cache_dir"])
    image_policy = dict(cfg.get("image_policy") or {})
    force = bool(cfg["force"])

    rows1 = extract_rows(task["rot1_img"], clean=True, max_chars=tile) if task.get("rot1_img") else []
    rows2 = extract_rows(task["rot2_img"], clean=True, max_chars=tile) if task.get("rot2_img") else []
    aligned_rows = len(rows1) == len(rows2)
    total = 0
    created = 0
    examples: list[dict[str, Any]] = []
    for row_idx, row1 in enumerate(rows1):
        row2 = rows2[row_idx] if aligned_rows and row_idx < len(rows2) else []
        aligned_chunks = aligned_rows and len(row1) == len(row2)
        row_text = "".join(text for _crop, text in row1)
        for chunk_idx, (crop1, text) in enumerate(row1):
            label = keep_chars(text, alphabet)
            if len(label) != 1 or label not in allowed:
                continue
            crop2 = None
            chunk_aligned = False
            if aligned_chunks and chunk_idx < len(row2) and row2[chunk_idx][1] == text:
                crop2 = row2[chunk_idx][0]
                chunk_aligned = True
            item = {
                "crop1": crop1,
                "crop2": crop2,
                "label": label,
                "split": task["split"],
                "base": task["base"],
                "rot1_img": task["rot1_img"],
                "rot2_img": task.get("rot2_img"),
                "row_idx": int(row_idx),
                "chunk_idx": int(chunk_idx),
                "row_text": row_text,
                "chunk_aligned": chunk_aligned,
                "base_pos": int(task.get("base_pos", 0)),
            }
            examples.append(cached_example_record(item))
            path = char_image_cache_path(image_cache_dir, item, input_mode, dual_mode, image_policy)
            total += 1
            if path.exists() and not force:
                continue
            img = image_for_char_example(item, g, tile, input_mode, dual_mode, canonical_cache, image_policy).convert("RGB")
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
            created += 1
    return {"split": task["split"], "base": task["base"], "total": total, "created": created, "examples": examples}


def cache_images_parallel(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
    import multiprocessing as mp

    config = {
        "init_ckpt": args.init_ckpt,
        "tile": args.tile,
        "alphabet": args.alphabet,
        "input_mode": args.input_mode,
        "dual_mode": args.dual_mode,
        "image_policy": image_policy_for_args(args),
        "image_cache_dir": args.image_cache_dir,
        "work": str(WORK),
        "charpost_dir": str(CHARPOST_DIR),
        "default_folds": str(DEFAULT_FOLDS),
        "force": args.force,
    }
    total = 0
    created = 0
    examples: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    with mp.get_context("spawn").Pool(processes=workers, initializer=_cache_images_worker_init, initargs=(config,)) as pool:
        for idx, rec in enumerate(pool.imap_unordered(_cache_images_for_base, tasks), 1):
            total += int(rec["total"])
            created += int(rec["created"])
            examples.extend(rec.get("examples") or [])
            if idx == len(tasks) or idx % max(args.log_every_bases, 1) == 0:
                print(f"  bases: {idx}/{len(tasks)} total={total} created={created}", flush=True)
    return total, created, examples


def write_image_cache_manifest(args: argparse.Namespace, total: int, created: int, examples: list[dict[str, Any]]) -> None:
    import time

    policy = image_policy_for_args(args)
    cache_key = image_policy_cache_key(args.input_mode, args.dual_mode, policy)
    out_dir = Path(args.image_cache_dir) / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)
    example_index = write_cached_example_index(args, examples)
    manifest = {
        "kind": "charpost_image_cache",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "init_ckpt": args.init_ckpt,
        "tile": args.tile,
        "splits": args.splits,
        "alphabet": args.alphabet,
        "input_mode": args.input_mode,
        "dual_mode": args.dual_mode,
        "image_policy": policy,
        "cache_key": cache_key,
        "total_files": total,
        "created_files": created,
        "examples": len(examples),
        "example_index": example_index.name,
        "workers": args.workers,
    }
    atomic_write_json(out_dir / "manifest.json", manifest, indent=2, sort_keys=True)
    default_cache_dir = (CHARPOST_DIR / "image_cache").resolve()
    if Path(args.image_cache_dir).resolve() == default_cache_dir:
        matrix_dir = CHARPOST_DIR / "cache_matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(matrix_dir / f"{cache_key}.json", manifest, indent=2, sort_keys=True)


def command_cache_images(args: argparse.Namespace) -> None:
    g = R.load(ckpt=args.init_ckpt, tile=args.tile, load_model=False, verbose=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    idf = g["index_df"]
    tasks = [
        {
            "split": str(row["split"]),
            "base": str(row["base"]),
            "rot1_img": row["rot1_img"],
            "rot2_img": row.get("rot2_img"),
            "base_pos": int(base_pos),
        }
        for base_pos, (_, row) in enumerate(idf[idf["split"].isin(set(splits))].iterrows())
    ]
    base_pos_by_key = {(str(task["split"]), str(task["base"])): int(task["base_pos"]) for task in tasks}
    if args.workers > 1 and args.limit == 0:
        print(f"cache-images parallel workers={args.workers} bases={len(tasks)} input_mode={args.input_mode}", flush=True)
        total, created, examples = cache_images_parallel(args, tasks)
        write_image_cache_manifest(args, total, created, examples)
        print(f"cache-images done total={total} created={created} examples={len(examples)} dir={args.image_cache_dir}")
        return

    cache: dict[str, tuple[Any, list[Any], list[list[int]], str]] = {}
    image_policy = image_policy_for_args(args)
    total = 0
    created = 0
    examples_index: list[dict[str, Any]] = []
    for split in splits:
        examples = classifier_examples(g, split, args.tile, args.alphabet, args.limit)
        print(f"{split}: cache-images examples={len(examples)} input_mode={args.input_mode}", flush=True)
        for idx, item in enumerate(examples, 1):
            item = dict(item)
            item["base_pos"] = base_pos_by_key.get((str(item.get("split")), str(item.get("base"))), 0)
            examples_index.append(cached_example_record(item))
            path = char_image_cache_path(args.image_cache_dir, item, args.input_mode, args.dual_mode, image_policy)
            total += 1
            if path.exists() and not args.force:
                pass
            else:
                img = image_for_char_example(item, g, args.tile, args.input_mode, args.dual_mode, cache, image_policy).convert("RGB")
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)
                created += 1
            if idx == len(examples) or idx % max(args.log_every, 1) == 0:
                print(f"  {split}: {idx}/{len(examples)} created={created}", flush=True)
    write_image_cache_manifest(args, total, created, examples_index)
    print(f"cache-images done total={total} created={created} examples={len(examples_index)} dir={args.image_cache_dir}")


def classifier_feature_dim(hidden_size: int, pooling: str, input_mode: str, multiview_fuse: str) -> int:
    del input_mode, multiview_fuse
    base = pooling_dim(hidden_size, pooling)
    return base


def encoder_forward(encoder: Any, pixel_values: Any, interpolate_pos_encoding: bool = False) -> Any:
    kwargs = {"pixel_values": pixel_values, "return_dict": True}
    if interpolate_pos_encoding:
        kwargs["interpolate_pos_encoding"] = True
    try:
        return encoder(**kwargs)
    except TypeError:
        kwargs.pop("interpolate_pos_encoding", None)
        if interpolate_pos_encoding:
            raise
        return encoder(**kwargs)


def classifier_logits_from_hidden(head: Any, last_hidden_state: Any, pooling: str) -> Any:
    if getattr(head, "takes_hidden_state", False):
        return head(last_hidden_state)
    return head(pool_encoder_output(last_hidden_state, pooling))


def classifier_forward(
    encoder: Any,
    head: Any,
    pixel_values: Any,
    pooling: str,
    multiview_fuse: str = "concat",
    interpolate_pos_encoding: bool = False,
) -> Any:
    import torch

    if pixel_values.dim() == 5:
        if getattr(head, "takes_hidden_state", False):
            raise ValueError("attention pooling head is not supported for multiview inputs")
        bsz, views, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(bsz * views, channels, height, width)
        out = encoder_forward(encoder, flat, interpolate_pos_encoding=interpolate_pos_encoding)
        feats = pool_encoder_output(out.last_hidden_state, pooling).reshape(bsz, views, -1)
        if multiview_fuse == "mean":
            fused = feats.mean(dim=1)
        elif multiview_fuse == "concat":
            fused = feats.reshape(bsz, views * feats.shape[-1])
        else:
            raise ValueError(f"unknown multiview_fuse={multiview_fuse}")
        return head(fused)
    out = encoder_forward(encoder, pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    return classifier_logits_from_hidden(head, out.last_hidden_state, pooling)


def evaluate_classifier(encoder: Any, head: Any, loader: Any, device: Any,
                        class_weight: Any | None = None, use_amp: bool = True, pooling: str = "cls",
                        multiview_fuse: str = "concat",
                        interpolate_pos_encoding: bool = False) -> dict[str, float]:
    import torch

    encoder.eval()
    head.eval()
    total = 0
    loss_sum = 0.0
    correct1 = 0
    correct3 = 0
    correct5 = 0
    with torch.no_grad():
        for batch in loader:
            pix = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                logits = classifier_forward(
                    encoder,
                    head,
                    pix,
                    pooling,
                    multiview_fuse=multiview_fuse,
                    interpolate_pos_encoding=interpolate_pos_encoding,
                )
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weight)
            total += labels.numel()
            loss_sum += float(loss.detach()) * labels.numel()
            top = logits.detach().topk(k=min(5, logits.shape[-1]), dim=-1).indices
            correct1 += int((top[:, 0] == labels).sum().item())
            correct3 += int((top[:, :min(3, top.shape[1])] == labels[:, None]).any(dim=1).sum().item())
            correct5 += int((top == labels[:, None]).any(dim=1).sum().item())
    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top3": correct3 / max(total, 1),
        "top5": correct5 / max(total, 1),
        "n": total,
    }


def class_weights(labels: Any, n_classes: int, mode: str) -> Any:
    import numpy as np

    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    if mode == "none":
        return np.ones(n_classes, dtype=np.float32)
    safe = np.maximum(counts, 1.0)
    if mode == "balanced":
        weights = labels.size / (n_classes * safe)
    else:
        weights = np.sqrt(labels.size / (n_classes * safe))
    weights *= n_classes / max(float(weights.sum()), 1e-6)
    return weights.astype(np.float32)


def restore_classifier_train_mode(encoder: Any, head: Any) -> None:
    import torch

    encoder.train()
    head.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_augment_profile(args: argparse.Namespace) -> None:
    profiles: dict[str, dict[str, float]] = {
        "light": {
            "aug_brightness": 0.05,
            "aug_contrast": 0.06,
            "aug_noise_std": 1.0,
            "aug_blur_prob": 0.02,
            "aug_sharpen_prob": 0.02,
            "aug_jitter_px": 2,
            "aug_scale": 0.02,
            "aug_channel_dropout": 0.0,
        },
        "realistic": {
            "aug_brightness": 0.08,
            "aug_contrast": 0.10,
            "aug_noise_std": 1.5,
            "aug_blur_prob": 0.03,
            "aug_sharpen_prob": 0.02,
            "aug_jitter_px": 2,
            "aug_scale": 0.025,
            "aug_channel_dropout": 0.03,
        },
        "strong": {
            "aug_brightness": 0.12,
            "aug_contrast": 0.14,
            "aug_noise_std": 2.5,
            "aug_blur_prob": 0.05,
            "aug_sharpen_prob": 0.05,
            "aug_jitter_px": 4,
            "aug_scale": 0.04,
            "aug_channel_dropout": 0.08,
        },
    }
    if args.augment_profile == "manual":
        return
    if args.augment_profile == "none":
        args.augment = False
        return
    for key, value in profiles[args.augment_profile].items():
        setattr(args, key, value)
    args.augment = True


def load_teacher_logits(paths_value: str, examples: list[dict[str, Any]], alphabet: str) -> Any | None:
    if not paths_value:
        return None
    import numpy as np

    expected_bases = [str(item["base"]) for item in examples]
    expected_rows = np.asarray([int(item["row_idx"]) for item in examples], dtype=np.int32)
    expected_chunks = np.asarray([int(item["chunk_idx"]) for item in examples], dtype=np.int32)
    parts: list[Any] = []
    for raw_path in paths_value.split(","):
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        data = np.load(path, allow_pickle=True)
        got_alphabet = "".join(str(x) for x in data["alphabet"].tolist())
        if got_alphabet != alphabet:
            raise SystemExit(f"teacher alphabet mismatch in {path}: {got_alphabet!r} != {alphabet!r}")
        if data["logits"].shape[0] != len(examples):
            raise SystemExit(f"teacher length mismatch in {path}: {data['logits'].shape[0]} != {len(examples)}")
        if [str(x) for x in data["bases"].tolist()] != expected_bases:
            raise SystemExit(f"teacher base order mismatch in {path}")
        if not np.array_equal(data["row_idx"].astype(np.int32), expected_rows):
            raise SystemExit(f"teacher row order mismatch in {path}")
        if not np.array_equal(data["chunk_idx"].astype(np.int32), expected_chunks):
            raise SystemExit(f"teacher chunk order mismatch in {path}")
        parts.append(data["logits"].astype(np.float32))
    if not parts:
        return None
    return np.mean(np.stack(parts, axis=0), axis=0)


def topk_metrics(logits: Any, labels: Any, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    import numpy as np

    if labels.size == 0:
        out = {f"top{k}": 0.0 for k in ks}
        out.update({"n": 0, "errors": 0})
        return out
    order = np.argsort(-logits, axis=1)
    out: dict[str, float] = {}
    for k in ks:
        out[f"top{k}"] = float(np.mean([labels[i] in order[i, :k] for i in range(labels.shape[0])]))
    pred = order[:, 0]
    out["n"] = int(labels.size)
    out["errors"] = int(np.sum(pred != labels))
    return out


def classifier_loss(
    logits: Any,
    labels: Any,
    weights: Any,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    import torch

    ce = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    teacher = batch.get("teacher_logits")
    if teacher is None or float(args.distill_alpha) <= 0:
        return ce
    temp = max(float(args.distill_temperature), 1e-6)
    teacher = teacher.to(logits.device)
    soft = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits / temp, dim=-1),
        torch.nn.functional.softmax(teacher / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
    alpha = float(args.distill_alpha)
    return (1.0 - alpha) * ce + alpha * soft


def _cpu_tree(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(val) for val in value)
    return value


def load_torch_file(path: Path, map_location: str | Any = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_classifier_train_state(
    out_dir: Path,
    opt: Any,
    scaler: Any,
    args: argparse.Namespace,
    *,
    global_step: int,
    epoch: int,
    batch_idx: int,
    best: dict[str, float],
    history: list[dict[str, Any]],
) -> None:
    if not args.save_train_state:
        return
    import random
    import numpy as np
    import torch

    state = {
        "kind": "charpost_classifier_train_state",
        "version": 1,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "max_steps": int(args.max_steps),
        "epochs": int(args.epochs),
        "train_complete": bool(args.max_steps and int(global_step) >= int(args.max_steps)),
        "batch": int(args.batch),
        "grad_accum": int(args.grad_accum),
        "optimizer": _cpu_tree(opt.state_dict()),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best": best,
        "history_len": len(history),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    final = out_dir / "train_state.pt"
    atomic_torch_save(state, final)
    print(f"saved train state -> {final} step={global_step}", flush=True)


def restore_classifier_train_state(
    out_dir: Path,
    opt: Any,
    scaler: Any,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    path = out_dir / "train_state.pt"
    if not (args.resume and path.exists()):
        return 0, 1, 0
    import random
    import numpy as np
    import torch

    state = load_torch_file(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"bad train state: {path}")
    saved_batch = int(state.get("batch", args.batch))
    saved_accum = int(state.get("grad_accum", args.grad_accum))
    if saved_batch != int(args.batch) or saved_accum != int(args.grad_accum):
        raise RuntimeError(
            f"resume train_state batch/accum mismatch: "
            f"{saved_batch}/{saved_accum} != {args.batch}/{args.grad_accum}"
        )
    opt.load_state_dict(state["optimizer"])
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    rng = state.get("rng") or {}
    try:
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as exc:
        print(f"warning: failed to restore RNG state from {path}: {exc}", flush=True)
    global_step = int(state.get("global_step", 0))
    epoch = max(1, int(state.get("epoch", 1)))
    batch_idx = max(0, int(state.get("batch_idx", 0)))
    print(f"restored train state <- {path} step={global_step} epoch={epoch} batch_idx={batch_idx}", flush=True)
    return global_step, epoch, batch_idx


def classifier_resume_state_issue(out_dir: Path) -> str:
    state_path = out_dir / "train_state.pt"
    head_path = out_dir / "head.pt"
    meta_path = out_dir / "metadata.json"
    if not state_path.exists():
        return "missing train_state.pt"
    if not head_path.exists():
        return "missing head.pt"
    if not meta_path.exists():
        return "missing metadata.json"
    try:
        state = load_torch_file(state_path, map_location="cpu")
        head = load_torch_file(head_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"unreadable resume files: {exc}"
    if not isinstance(state, dict) or state.get("kind") != "charpost_classifier_train_state":
        return "bad train_state.pt kind"
    try:
        state_step = int(state.get("global_step", -1))
        head_step = int(head.get("global_step", -1))
        meta_step = int(meta.get("global_step", -1))
    except Exception as exc:
        return f"bad resume step metadata: {exc}"
    if state_step < 0:
        return "missing train_state global_step"
    if head_step != state_step or meta_step != state_step:
        return f"step mismatch state/head/meta={state_step}/{head_step}/{meta_step}"
    return ""


def command_train_classifier(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    apply_augment_profile(args)
    label_to_id = {ch: i for i, ch in enumerate(args.alphabet)}
    train_exclude = fold_bases_for_args(args, "train_exclude_fold_id")
    val_include = fold_bases_for_args(args, "val_fold_id")
    train_splits = [part.strip() for part in args.train_splits.split(",") if part.strip()]
    if not train_splits:
        raise SystemExit("--train-splits must contain at least one split")
    if train_exclude is not None and train_splits != ["train"]:
        raise SystemExit("--train-exclude-fold-id is only valid with --train-splits train")
    val_split = "train" if val_include is not None else args.val_split
    image_policy = image_policy_for_args(args)
    cached_index: list[dict[str, Any]] | None = None
    cached_index_path: Path | None = None
    if args.require_image_cache:
        cached_index, cached_index_path = load_cached_example_index(
            args.image_cache_dir,
            args.input_mode,
            args.dual_mode,
            image_policy,
        )
        print(f"cached example index loaded: examples={len(cached_index)} path={cached_index_path}", flush=True)

    train_examples: list[dict[str, Any]] = []
    if cached_index is not None:
        for train_split in train_splits:
            train_examples.extend(filter_cached_examples(
                cached_index,
                train_split,
                args.alphabet,
                args.train_limit,
                exclude_bases=train_exclude,
            ))
        val_examples = filter_cached_examples(
            cached_index,
            val_split,
            args.alphabet,
            args.val_limit,
            include_bases=val_include,
        )
        example_source = f"cached-index:{cached_index_path}"
    else:
        g = R.load(ckpt=args.init_ckpt, tile=args.tile, load_model=True, verbose=True)
        for train_split in train_splits:
            train_examples.extend(classifier_examples(
                g,
                train_split,
                args.tile,
                args.alphabet,
                args.train_limit,
                exclude_bases=train_exclude,
            ))
        val_examples = classifier_examples(
            g,
            val_split,
            args.tile,
            args.alphabet,
            args.val_limit,
            include_bases=val_include,
        )
        example_source = "direct-crop-extraction"
    if train_exclude is not None or val_include is not None:
        print(
            f"fold-aware train: train_examples={len(train_examples)} "
            f"train_splits={','.join(train_splits)} val_examples={len(val_examples)} val_split={val_split} "
            f"train_exclude_fold_id={args.train_exclude_fold_id} val_fold_id={args.val_fold_id}",
            flush=True,
        )
    if not train_examples or not val_examples:
        raise SystemExit("train/val examples are required")
    print(
        f"classifier examples ready: source={example_source} train={len(train_examples)} val={len(val_examples)}",
        flush=True,
    )
    if cached_index is not None:
        print("loading line recognizer for classifier after cached example index", flush=True)
        g = R.load(ckpt=args.init_ckpt, tile=args.tile, load_model=True, verbose=True)
    processor = g["processor"]
    prep_image = g["prep_image"]
    ved = g.pop("ft_model")
    encoder = ved.encoder
    # Drop the decoder from memory; this experiment trains only the visual encoder + char head.
    try:
        del ved.decoder
    except Exception:
        pass
    del ved
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_logits = load_teacher_logits(args.teacher_logits, train_examples, args.alphabet)
    dataset_helpers = None if cached_index is not None else g
    train_ds = CharCropDataset(
        train_examples,
        processor,
        prep_image,
        label_to_id,
        args.dual_mode,
        input_mode=args.input_mode,
        helpers=dataset_helpers,
        tile=args.tile,
        augment=args.augment,
        augment_args=args,
        image_cache_dir=args.image_cache_dir,
        require_image_cache=args.require_image_cache,
        image_policy=image_policy,
        input_size=args.input_size,
        teacher_logits=teacher_logits,
    ).dataset
    val_ds = CharCropDataset(
        val_examples,
        processor,
        prep_image,
        label_to_id,
        args.dual_mode,
        input_mode=args.input_mode,
        helpers=dataset_helpers,
        tile=args.tile,
        augment=False,
        augment_args=args,
        image_cache_dir=args.image_cache_dir,
        require_image_cache=args.require_image_cache,
        image_policy=image_policy,
        input_size=args.input_size,
    ).dataset

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False, **loader_kwargs)
    print(
        f"dataloaders ready: workers={args.workers} prefetch_factor={args.prefetch_factor if args.workers > 0 else 0} "
        f"pin_memory={loader_kwargs['pin_memory']}",
        flush=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recover_atomic_dir(out_dir / "encoder")
    recover_atomic_dir(out_dir / "processor")

    resume_has_weights = (out_dir / "encoder").exists() and (out_dir / "head.pt").exists()
    resume_has_state = (out_dir / "train_state.pt").exists()
    resume_state_issue = classifier_resume_state_issue(out_dir) if resume_has_state else "missing train_state.pt"
    if resume_state_issue:
        resume_has_state = False
    if args.resume and resume_has_weights and resume_has_state:
        from transformers import AutoModel

        print(f"resuming classifier from {out_dir}", flush=True)
        encoder = AutoModel.from_pretrained(out_dir / "encoder")
        head_ckpt = load_torch_file(out_dir / "head.pt", map_location="cpu")
        head = build_classifier_head(
            int(head_ckpt["in_dim"]),
            len(head_ckpt["alphabet"]),
            use_mlp=bool(head_ckpt.get("head_mlp", False)),
            hidden=int(head_ckpt.get("head_hidden", 0)),
            dropout=float(head_ckpt.get("head_dropout", 0.1)),
            pooling=str(head_ckpt.get("pooling", "cls")),
        )
        head.load_state_dict(head_ckpt["state_dict"])
        if str(head_ckpt["alphabet"]) != args.alphabet:
            raise RuntimeError(f"resume alphabet mismatch: {head_ckpt['alphabet']} != {args.alphabet}")
        if str(head_ckpt.get("input_mode", "dual_pack")) != args.input_mode:
            raise RuntimeError(f"resume input_mode mismatch: {head_ckpt.get('input_mode', 'dual_pack')} != {args.input_mode}")
        if str(head_ckpt.get("pooling", "cls")) != args.pooling:
            raise RuntimeError(f"resume pooling mismatch: {head_ckpt.get('pooling', 'cls')} != {args.pooling}")
        if bool(head_ckpt.get("head_mlp", False)) != bool(args.head_mlp):
            raise RuntimeError(f"resume head_mlp mismatch: {head_ckpt.get('head_mlp', False)} != {args.head_mlp}")
        if str(head_ckpt.get("multiview_fuse", "concat")) != args.multiview_fuse:
            raise RuntimeError(f"resume multiview_fuse mismatch: {head_ckpt.get('multiview_fuse', 'concat')} != {args.multiview_fuse}")
        if int(head_ckpt.get("input_size", 0)) != int(args.input_size):
            raise RuntimeError(f"resume input_size mismatch: {head_ckpt.get('input_size', 0)} != {args.input_size}")
        meta_path = out_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if str(meta.get("train_splits", "train")) != str(args.train_splits):
            raise RuntimeError(f"resume train_splits mismatch: {meta.get('train_splits', 'train')} != {args.train_splits}")
        if str(meta.get("val_split", "val")) != str(args.val_split):
            raise RuntimeError(f"resume val_split mismatch: {meta.get('val_split', 'val')} != {args.val_split}")
        if meta.get("train_exclude_fold_id") != args.train_exclude_fold_id:
            raise RuntimeError(
                f"resume train_exclude_fold_id mismatch: {meta.get('train_exclude_fold_id')} != {args.train_exclude_fold_id}"
            )
        if meta.get("val_fold_id") != args.val_fold_id:
            raise RuntimeError(f"resume val_fold_id mismatch: {meta.get('val_fold_id')} != {args.val_fold_id}")
        best: dict[str, float] = dict(meta.get("best") or {"top1": -1.0, "loss": float("inf")})
        history: list[dict[str, Any]] = list(meta.get("history") or [])
    else:
        if args.resume and resume_has_weights and not resume_has_state:
            print(
                f"existing classifier weights at {out_dir} have no usable train_state.pt "
                f"({resume_state_issue}); restarting from init checkpoint instead of warm-resuming",
                flush=True,
            )
        hidden = int(getattr(encoder.config, "hidden_size"))
        head_in_dim = classifier_feature_dim(hidden, args.pooling, args.input_mode, args.multiview_fuse)
        head = build_classifier_head(
            head_in_dim,
            len(args.alphabet),
            use_mlp=args.head_mlp,
            hidden=args.head_hidden,
            dropout=args.head_dropout,
            pooling=args.pooling,
        )
        best = {"top1": -1.0, "loss": float("inf")}
        history = []

    encoder.to(device)
    head.to(device)
    if args.grad_checkpoint and hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
    y_train = np.asarray([label_to_id[item["label"]] for item in train_examples], dtype=np.int64)
    weights_np = class_weights(y_train, len(args.alphabet), args.class_weight)
    weights = torch.from_numpy(weights_np).to(device)
    encoder_lr = float(args.encoder_lr) if args.encoder_lr is not None else float(args.lr)
    head_lr = float(args.head_lr) if args.head_lr is not None else float(args.lr)
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": encoder_lr},
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": head_lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    interpolate_pos_encoding = bool(args.interpolate_pos_encoding or args.input_size > 0)

    global_step, start_epoch, resume_batch_idx = restore_classifier_train_state(out_dir, opt, scaler, args)
    stop = bool(args.max_steps and global_step >= args.max_steps)
    printed_first_batch = False
    print(
        f"training loop start: device={device} max_steps={args.max_steps} start_step={global_step} "
        f"batch={args.batch} grad_accum={args.grad_accum}",
        flush=True,
    )
    epoch = start_epoch
    while not stop:
        if not args.max_steps and epoch > args.epochs:
            break
        encoder.train()
        head.train()
        for batch_idx, batch in enumerate(train_loader, 1):
            if epoch == start_epoch and resume_batch_idx and batch_idx <= resume_batch_idx:
                continue
            if not printed_first_batch:
                print("first train batch ready; starting forward/backward", flush=True)
                printed_first_batch = True
            pix = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                logits = classifier_forward(
                    encoder,
                    head,
                    pix,
                    args.pooling,
                    multiview_fuse=args.multiview_fuse,
                    interpolate_pos_encoding=interpolate_pos_encoding,
                )
                loss = classifier_loss(logits, labels, weights, batch, args) / args.grad_accum
            scaler.scale(loss).backward()
            if batch_idx % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if args.log_steps > 0 and global_step % args.log_steps == 0:
                    print(f"step {global_step}: train_loss={float(loss.detach()) * args.grad_accum:.4f}", flush=True)
                step_metrics = None
                if args.state_steps > 0 and global_step % args.state_steps == 0:
                    state_metrics = {"top1": best.get("top1", -1.0), "loss": best.get("loss", float("inf")), "step": float(global_step)}
                    save_classifier_checkpoint(
                        out_dir,
                        encoder,
                        head,
                        processor,
                        args,
                        best,
                        history,
                        checkpoint_metrics=state_metrics,
                    )
                    save_classifier_train_state(
                        out_dir,
                        opt,
                        scaler,
                        args,
                        global_step=global_step,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        best=best,
                        history=history,
                    )
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    step_metrics = evaluate_classifier(
                        encoder,
                        head,
                        val_loader,
                        device,
                        weights,
                        args.fp16,
                        args.pooling,
                        multiview_fuse=args.multiview_fuse,
                        interpolate_pos_encoding=interpolate_pos_encoding,
                    )
                    rec = {"epoch": epoch, "step": global_step, **{f"val_{k}": v for k, v in step_metrics.items()}}
                    history.append(rec)
                    print("eval " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
                    better = step_metrics["top1"] > best["top1"] or (step_metrics["top1"] == best["top1"] and step_metrics["loss"] < best["loss"])
                    if better:
                        best = {"top1": step_metrics["top1"], "loss": step_metrics["loss"], "step": float(global_step)}
                        if args.save_during_training:
                            save_classifier_checkpoint(out_dir, encoder, head, processor, args, best, history,
                                                       checkpoint_metrics=best)
                            save_classifier_train_state(
                                out_dir,
                                opt,
                                scaler,
                                args,
                                global_step=global_step,
                                epoch=epoch,
                                batch_idx=batch_idx,
                                best=best,
                                history=history,
                            )
                if args.checkpoint_steps > 0 and global_step % args.checkpoint_steps == 0:
                    if step_metrics is None:
                        step_metrics = evaluate_classifier(
                            encoder,
                            head,
                            val_loader,
                            device,
                            weights,
                            args.fp16,
                            args.pooling,
                            multiview_fuse=args.multiview_fuse,
                            interpolate_pos_encoding=interpolate_pos_encoding,
                        )
                        rec = {"epoch": epoch, "step": global_step, **{f"val_{k}": v for k, v in step_metrics.items()}}
                        history.append(rec)
                        print("eval " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
                    snapshot_metrics = {"top1": step_metrics["top1"], "loss": step_metrics["loss"], "step": float(global_step)}
                    if (snapshot_metrics["top1"] > best["top1"] or
                            (snapshot_metrics["top1"] == best["top1"] and snapshot_metrics["loss"] < best["loss"])):
                        best = snapshot_metrics
                    save_classifier_checkpoint(
                        out_dir / f"checkpoint-{global_step}",
                        encoder,
                        head,
                        processor,
                        args,
                        best,
                        history,
                        checkpoint_metrics=snapshot_metrics,
                    )
                if step_metrics is not None:
                    restore_classifier_train_mode(encoder, head)
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break
        epoch += 1
    metrics = evaluate_classifier(
        encoder,
        head,
        val_loader,
        device,
        weights,
        args.fp16,
        args.pooling,
        multiview_fuse=args.multiview_fuse,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )
    better = metrics["top1"] > best["top1"] or (metrics["top1"] == best["top1"] and metrics["loss"] < best["loss"])
    if better:
        best = {"top1": metrics["top1"], "loss": metrics["loss"], "step": float(global_step)}
    final_metrics = {"top1": metrics["top1"], "loss": metrics["loss"], "step": float(global_step)}
    save_classifier_checkpoint(out_dir, encoder, head, processor, args, best, history, checkpoint_metrics=final_metrics)
    save_classifier_train_state(
        out_dir,
        opt,
        scaler,
        args,
        global_step=global_step,
        epoch=epoch,
        batch_idx=0,
        best=best,
        history=history,
    )
    print(f"best val top1={best['top1']:.4f} loss={best['loss']:.4f} step={best.get('step')}")
    print(f"wrote {out_dir}")


def save_classifier_checkpoint(out_dir: Path, encoder: Any, head: Any, processor: Any,
                               args: argparse.Namespace, best: dict[str, float],
                               history: list[dict[str, Any]],
                               checkpoint_metrics: dict[str, float] | None = None) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = int(getattr(encoder.config, "hidden_size"))
    in_dim = classifier_feature_dim(hidden_size, args.pooling, args.input_mode, args.multiview_fuse)
    image_policy = image_policy_for_args(args)
    checkpoint_metrics = checkpoint_metrics or best
    try:
        current_step = int(float(checkpoint_metrics.get("step", best.get("step", 0) or 0)))
    except Exception:
        current_step = 0
    train_complete = bool(args.max_steps and current_step >= int(args.max_steps))
    # Avoid saving GPU tensors directly from the live model. On the Jupyter/PyTorch stack this
    # occasionally tripped a CUDA caching-allocator/NVML assert after save_pretrained returned.
    # Supplying an explicit CPU state_dict keeps the live CUDA module untouched.
    state_dict = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    tmp_encoder = out_dir / f".encoder.tmp.{os.getpid()}"
    if tmp_encoder.exists():
        shutil.rmtree(tmp_encoder)
    try:
        encoder.save_pretrained(tmp_encoder, state_dict=state_dict)
        replace_dir_with_backup(tmp_encoder, out_dir / "encoder")
    finally:
        if tmp_encoder.exists():
            shutil.rmtree(tmp_encoder)
        del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        tmp_processor = out_dir / f".processor.tmp.{os.getpid()}"
        if tmp_processor.exists():
            shutil.rmtree(tmp_processor)
        try:
            processor.save_pretrained(tmp_processor)
            replace_dir_with_backup(tmp_processor, out_dir / "processor")
        finally:
            if tmp_processor.exists():
                shutil.rmtree(tmp_processor)
    except Exception:
        pass
    atomic_torch_save({
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "in_dim": in_dim,
        "hidden_size": hidden_size,
        "alphabet": args.alphabet,
        "init_ckpt": args.init_ckpt,
        "tile": args.tile,
        "train_splits": args.train_splits,
        "val_split": args.val_split,
        "train_exclude_fold_id": args.train_exclude_fold_id,
        "val_fold_id": args.val_fold_id,
        "dual_mode": args.dual_mode,
        "input_mode": args.input_mode,
        "input_size": int(args.input_size),
        "interpolate_pos_encoding": bool(args.interpolate_pos_encoding or args.input_size > 0),
        "pooling": args.pooling,
        "multiview_fuse": args.multiview_fuse,
        "head_mlp": bool(args.head_mlp),
        "head_hidden": int(args.head_hidden),
        "head_dropout": float(args.head_dropout),
        "augment": bool(args.augment),
        "augment_profile": args.augment_profile,
        "image_cache_dir": args.image_cache_dir,
        "image_policy": image_policy,
        "encoder_lr": float(args.encoder_lr) if args.encoder_lr is not None else float(args.lr),
        "head_lr": float(args.head_lr) if args.head_lr is not None else float(args.lr),
        "base_lr": float(args.lr),
        "global_step": current_step,
        "train_complete": train_complete,
        "batch": int(args.batch),
        "eval_batch": int(args.eval_batch),
        "grad_accum": int(args.grad_accum),
        "max_steps": int(args.max_steps),
        "workers": int(args.workers),
        "prefetch_factor": int(args.prefetch_factor),
        "state_steps": int(args.state_steps),
        "save_train_state": bool(args.save_train_state),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "fp16": bool(args.fp16),
        "distill_alpha": float(args.distill_alpha),
        "distill_temperature": float(args.distill_temperature),
        "teacher_logits": args.teacher_logits,
        "augmentation": {
            "brightness": args.aug_brightness,
            "contrast": args.aug_contrast,
            "noise_std": args.aug_noise_std,
            "blur_prob": args.aug_blur_prob,
            "sharpen_prob": args.aug_sharpen_prob,
            "jitter_px": args.aug_jitter_px,
            "scale": args.aug_scale,
            "channel_dropout": args.aug_channel_dropout,
        },
    }, out_dir / "head.pt")
    meta = {
        "tag": args.tag,
        "init_ckpt": args.init_ckpt,
        "tile": args.tile,
        "train_splits": args.train_splits,
        "val_split": args.val_split,
        "train_exclude_fold_id": args.train_exclude_fold_id,
        "val_fold_id": args.val_fold_id,
        "alphabet": args.alphabet,
        "dual_mode": args.dual_mode,
        "input_mode": args.input_mode,
        "input_size": int(args.input_size),
        "interpolate_pos_encoding": bool(args.interpolate_pos_encoding or args.input_size > 0),
        "pooling": args.pooling,
        "multiview_fuse": args.multiview_fuse,
        "head_mlp": bool(args.head_mlp),
        "head_hidden": int(args.head_hidden),
        "head_dropout": float(args.head_dropout),
        "augment": bool(args.augment),
        "augment_profile": args.augment_profile,
        "image_cache_dir": args.image_cache_dir,
        "image_policy": image_policy,
        "encoder_lr": float(args.encoder_lr) if args.encoder_lr is not None else float(args.lr),
        "head_lr": float(args.head_lr) if args.head_lr is not None else float(args.lr),
        "base_lr": float(args.lr),
        "global_step": current_step,
        "train_complete": train_complete,
        "batch": int(args.batch),
        "eval_batch": int(args.eval_batch),
        "grad_accum": int(args.grad_accum),
        "max_steps": int(args.max_steps),
        "workers": int(args.workers),
        "prefetch_factor": int(args.prefetch_factor),
        "state_steps": int(args.state_steps),
        "save_train_state": bool(args.save_train_state),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "fp16": bool(args.fp16),
        "distill_alpha": float(args.distill_alpha),
        "distill_temperature": float(args.distill_temperature),
        "teacher_logits": args.teacher_logits,
        "best": best,
        "checkpoint_metrics": checkpoint_metrics,
        "history": history,
    }
    atomic_write_json(out_dir / "metadata.json", meta, indent=2, sort_keys=True)


def load_classifier_head_file(model_dir: Path, map_location: str | Any = "cpu") -> dict[str, Any]:
    import torch

    try:
        return torch.load(model_dir / "head.pt", map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_dir / "head.pt", map_location=map_location)


def load_classifier_checkpoint(model_dir: Path, init_ckpt: str = "", tile: int = 1, device: str = "auto") -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModel, TrOCRProcessor

    recover_atomic_dir(model_dir / "encoder")
    recover_atomic_dir(model_dir / "processor")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    init = init_ckpt or meta.get("init_ckpt")
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = AutoModel.from_pretrained(model_dir / "encoder").to(dev).eval()
    ckpt = load_classifier_head_file(model_dir, map_location=dev)
    processor_dir = model_dir / "processor"
    if processor_dir.exists():
        processor = TrOCRProcessor.from_pretrained(processor_dir)
    else:
        proc_path = R._processor_path_for_ckpt(str(init))
        if not proc_path:
            raise FileNotFoundError(
                f"missing saved classifier processor at {processor_dir}; "
                f"could not find processor files near init checkpoint {init}"
            )
        processor = TrOCRProcessor.from_pretrained(proc_path)
    prep = {
        "processor": processor,
        "prep_image": make_default_prep_image(int(ckpt.get("input_size", 0))),
        "device": dev,
    }
    head = build_classifier_head(
        int(ckpt["in_dim"]),
        len(ckpt["alphabet"]),
        use_mlp=bool(ckpt.get("head_mlp", False)),
        hidden=int(ckpt.get("head_hidden", 0)),
        dropout=float(ckpt.get("head_dropout", 0.1)),
        pooling=str(ckpt.get("pooling", "cls")),
    ).to(dev)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return encoder, head, ckpt, {**prep, "device": dev}


def command_cache_logits(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    model_dir = Path(args.model_dir)
    cached_index: list[dict[str, Any]] | None = None
    cached_index_path: Path | None = None
    prelim_ckpt: dict[str, Any] | None = None
    if args.require_image_cache:
        prelim_ckpt = load_classifier_head_file(model_dir, map_location="cpu")
        prelim_image_cache_dir = args.image_cache_dir or str(prelim_ckpt.get("image_cache_dir", ""))
        cached_index, cached_index_path = load_cached_example_index(
            prelim_image_cache_dir,
            str(prelim_ckpt.get("input_mode", "dual_pack")),
            str(prelim_ckpt.get("dual_mode", "min")),
            dict(prelim_ckpt.get("image_policy") or {}),
        )
        print(f"cached example index loaded for logits: examples={len(cached_index)} path={cached_index_path}", flush=True)
    encoder, head, ckpt, prep = load_classifier_checkpoint(model_dir, args.init_ckpt, args.tile, args.device)
    processor = prep["processor"]
    prep_image = prep["prep_image"]
    device = prep["device"]
    alphabet = str(ckpt["alphabet"])
    label_to_id = {ch: i for i, ch in enumerate(alphabet)}
    g = None if cached_index is not None else R.load(ckpt=ckpt["init_ckpt"], tile=args.tile, load_model=False, verbose=True)
    input_mode = str(ckpt.get("input_mode", "dual_pack"))
    dual_mode = str(ckpt.get("dual_mode", "min"))
    pooling = str(ckpt.get("pooling", "cls"))
    multiview_fuse = str(ckpt.get("multiview_fuse", "concat"))
    input_size = int(ckpt.get("input_size", 0))
    interpolate_pos_encoding = bool(ckpt.get("interpolate_pos_encoding", input_size > 0))
    image_policy = dict(ckpt.get("image_policy") or {})
    image_cache_dir = args.image_cache_dir or str(ckpt.get("image_cache_dir", ""))
    specs = tta_specs(args.tta, args.tta_pixels, args.tta_scale)
    print(
        f"cache-logits input_mode={input_mode} pooling={pooling} multiview_fuse={multiview_fuse} "
        f"input_size={input_size or 'processor-default'} tta={args.tta} variants={len(specs)} "
        f"image_cache={image_cache_dir}",
        flush=True,
    )
    canonical_cache: dict[str, tuple[Any, list[Any], list[list[int]], str]] = {}

    def apply_tta_visual(visual: Any, spec: tuple[float, float, float]) -> Any:
        if isinstance(visual, list):
            return [apply_tta_image(img, spec) for img in visual]
        return apply_tta_image(visual, spec)

    include_bases = fold_bases_for_args(args, "include_fold_id")
    exclude_bases = fold_bases_for_args(args, "exclude_fold_id")
    split_names = [part.strip() for part in args.splits.split(",") if part.strip()]
    if args.output_split and len(split_names) != 1:
        raise SystemExit("--output-split requires exactly one source split in --splits")
    dataset_helpers = None if cached_index is not None else g
    for split in split_names:
        if cached_index is not None:
            examples = filter_cached_examples(
                cached_index,
                split,
                alphabet,
                args.limit,
                include_bases=include_bases,
                exclude_bases=exclude_bases,
            )
        else:
            examples = classifier_examples(
                g,
                split,
                args.tile,
                alphabet,
                args.limit,
                include_bases=include_bases,
                exclude_bases=exclude_bases,
            )
        out_split = args.output_split or split
        if include_bases is not None or exclude_bases is not None:
            print(
                f"cache-logits slice source_split={split} out_split={out_split} "
                f"examples={len(examples)} include_fold_id={args.include_fold_id} exclude_fold_id={args.exclude_fold_id}",
                flush=True,
            )
        logits_parts: list[np.ndarray] = []
        labels: list[int] = []
        bases: list[str] = []
        row_idx: list[int] = []
        chunk_idx: list[int] = []
        row_texts: list[str] = []
        with torch.no_grad():
            for start in range(0, len(examples), args.batch):
                batch = examples[start:start + args.batch]
                logits_accum = None
                base_visuals = [
                    cached_visual_inputs_for_char_example(
                        item,
                        dataset_helpers,
                        args.tile,
                        input_mode,
                        dual_mode,
                        canonical_cache,
                        image_cache_dir,
                        args.require_image_cache,
                        image_policy,
                    )
                    for item in batch
                ]
                for spec in specs:
                    tensors = [
                        process_visual_inputs(processor, prep_image, apply_tta_visual(visual, spec), input_size)
                        for visual in base_visuals
                    ]
                    pix = torch.stack(tensors, dim=0).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                        logits = classifier_forward(
                            encoder,
                            head,
                            pix,
                            pooling,
                            multiview_fuse=multiview_fuse,
                            interpolate_pos_encoding=interpolate_pos_encoding,
                        )
                    logits_accum = logits.detach().float() if logits_accum is None else logits_accum + logits.detach().float()
                logits_avg = logits_accum / max(len(specs), 1)
                logits_parts.append(logits_avg.cpu().numpy())
                labels.extend(label_to_id[item["label"]] for item in batch)
                bases.extend(str(item["base"]) for item in batch)
                row_idx.extend(int(item["row_idx"]) for item in batch)
                chunk_idx.extend(int(item["chunk_idx"]) for item in batch)
                row_texts.extend(str(item["row_text"]) for item in batch)
                done = min(start + len(batch), len(examples))
                if done == len(examples) or done % max(args.batch * 20, 1) == 0:
                    print(f"  {split}: {done}/{len(examples)}", flush=True)
        arr = np.concatenate(logits_parts, axis=0) if logits_parts else np.zeros((0, len(alphabet)), dtype=np.float32)
        out_path = logits_path(args.tag, out_split)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_np_savez_compressed(
            out_path,
            logits=arr.astype(np.float16),
            labels=np.asarray(labels, dtype=np.int64),
            bases=np.asarray(bases),
            row_idx=np.asarray(row_idx, dtype=np.int32),
            chunk_idx=np.asarray(chunk_idx, dtype=np.int32),
            row_text=np.asarray(row_texts),
            alphabet=np.asarray(list(alphabet)),
        )
        print(f"wrote {out_path} logits={arr.shape}")
        print(f"{split}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in topk_metrics(arr, np.asarray(labels)).items()))


def load_decode_inputs(tag: str, split: str, device: str = "auto") -> dict[str, Any]:
    del device
    import numpy as np
    import torch

    logits_file = logits_path(tag, split)
    if not logits_file.exists():
        raise FileNotFoundError(f"missing cached logits: {logits_file}")
    data = np.load(logits_file, allow_pickle=True)
    logits = data["logits"].astype(np.float32)
    return {
        "logits": logits,
        "log_probs": torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy(),
        "labels": data["labels"].astype(np.int64),
        "bases": data["bases"],
        "row_idx": data["row_idx"],
        "chunk_idx": data["chunk_idx"],
        "row_text": data["row_text"],
        "alphabet": "".join(str(x) for x in data["alphabet"].tolist()),
        "source": str(logits_file),
    }


def lm_delta(lm: Any, prefix: str, ch: str) -> float:
    # CharPostNGramLM exposes exact incremental scores. Recomputing
    # mean_logprob(prefix + ch) for every beam expansion is O(length²) and makes sweeps look hung.
    if hasattr(lm, "cond_logprob"):
        return float(lm.cond_logprob(ch, prefix))
    # Cached wrappers may expose their wrapped LM as .lm; unwrap so cond_logprob paths stay fast.
    if hasattr(lm, "lm"):
        return lm_delta(lm.lm, prefix, ch)
    # Score-level interpolation wrappers can expose in-domain/external LMs; interpolate deltas too.
    if hasattr(lm, "indomain_lm") and hasattr(lm, "external_lm") and hasattr(lm, "external_weight"):
        ew = float(lm.external_weight)
        return ((1.0 - ew) * lm_delta(lm.indomain_lm, prefix, ch) +
                ew * lm_delta(lm.external_lm, prefix, ch))
    # Generic fallback for wrappers without conditional scoring.
    before = lm.mean_logprob(prefix) * max(len(prefix), 1) if prefix else 0.0
    after_text = prefix + ch
    after = lm.mean_logprob(after_text) * len(after_text)
    return after - before


def decode_lattice(
    log_probs: list[dict[str, float]],
    lm: Any | None,
    lm_weight: float,
    beam_size: int,
    char_topk: int = 0,
) -> list[tuple[str, float]]:
    def candidates(pos: dict[str, float]) -> list[tuple[str, float]]:
        items = list(pos.items())
        if char_topk and char_topk > 0 and len(items) > char_topk:
            items.sort(key=lambda item: item[1], reverse=True)
            return items[:char_topk]
        return items

    # Visual-only top-1 factorizes by position; but keep a visual beam when requested so
    # neural LMs can rerank visual candidate rows without using the n-gram as an ensemble.
    if (lm is None or lm_weight == 0.0) and (beam_size <= 1 or char_topk == 1):
        text = ""
        score = 0.0
        for pos in log_probs:
            ch, val = max(pos.items(), key=lambda item: item[1])
            text += ch
            score += float(val)
        return [(text, score)]

    beams: list[tuple[str, float]] = [("", 0.0)]
    for pos in log_probs:
        cand = candidates(pos)
        nxt: list[tuple[str, float]] = []
        for prefix, score in beams:
            for ch, visual in cand:
                bonus = 0.0 if lm is None or lm_weight == 0.0 else lm_weight * lm_delta(lm, prefix, ch)
                nxt.append((prefix + ch, score + float(visual) + bonus))
        nxt.sort(key=lambda item: item[1], reverse=True)
        beams = nxt[:beam_size]
    return beams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    lm = sub.add_parser("train-lm", help="train a train-only/fold-excluded Python char n-gram LM")
    lm.add_argument("--tile", type=int, default=1)
    lm.add_argument("--splits", default="train")
    lm.add_argument("--order", type=int, default=10)
    lm.add_argument("--smooth-lambda", type=float, default=0.4)
    lm.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    lm.add_argument("--folds", default=str(DEFAULT_FOLDS))
    lm.add_argument("--exclude-fold-id", type=int, default=None)
    lm.add_argument("--out-dir", default=str(CHARPOST_DIR / "train_lm_order10"))
    lm.set_defaults(func=command_train_lm)

    cimg = sub.add_parser("cache-images", help="prebuild tile1 RGB input images for classifier training/cache-logits")
    cimg.add_argument("--init-ckpt", default=str(WORK / "SqueezeTrOCR" / "checkpoint-6000"))
    cimg.add_argument("--tile", type=int, default=1)
    cimg.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    cimg.add_argument("--splits", default="train,val,test")
    cimg.add_argument("--input-mode", choices=sorted(INPUT_MODES), default="dual_pack")
    cimg.add_argument("--dual-mode", default="min")
    cimg.add_argument("--image-cache-dir", default=str(CHARPOST_DIR / "image_cache"))
    cimg.add_argument("--limit", type=int, default=0)
    cimg.add_argument("--force", action="store_true")
    cimg.add_argument("--log-every", type=int, default=2000)
    cimg.add_argument("--workers", type=int, default=1)
    cimg.add_argument("--log-every-bases", type=int, default=10)
    cimg.set_defaults(func=command_cache_images)

    trainc = sub.add_parser("train-classifier", help="fine-tune encoder + char classifier head on tile1 crops")
    trainc.add_argument("--tag", default="charpost_t1L_ft")
    trainc.add_argument("--init-ckpt", default=str(WORK / "SqueezeTrOCR" / "checkpoint-6000"))
    trainc.add_argument("--out-dir", default=str(CHARPOST_DIR / "charpost_t1L_ft"))
    trainc.add_argument("--tile", type=int, default=1)
    trainc.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    trainc.add_argument("--dual-mode", default="min")
    trainc.add_argument("--input-mode", choices=sorted(INPUT_MODES), default="dual_pack",
                        help="tile1 visual input representation; canonical modes use annotation-aligned rot1/rot2 crops")
    trainc.add_argument("--pooling", choices=POOLING_MODES, default="cls",
                        help="encoder token pooling for the glyph classifier")
    trainc.add_argument("--multiview-fuse", choices=MULTIVIEW_FUSE_MODES, default="concat",
                        help=argparse.SUPPRESS)
    trainc.add_argument("--head-mlp", action=argparse.BooleanOptionalAction, default=False,
                        help="use a GELU MLP head instead of a linear classifier")
    trainc.add_argument("--head-hidden", type=int, default=0,
                        help="hidden width for --head-mlp; 0 picks a conservative default")
    trainc.add_argument("--head-dropout", type=float, default=0.1)
    trainc.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False,
                        help="enable general train-time visual augmentation")
    trainc.add_argument("--augment-profile", choices=("manual", "none", "light", "realistic", "strong"), default="manual",
                        help="named augmentation recipe; manual keeps the explicit aug-* values")
    trainc.add_argument("--aug-brightness", type=float, default=0.10)
    trainc.add_argument("--aug-contrast", type=float, default=0.10)
    trainc.add_argument("--aug-noise-std", type=float, default=3.0)
    trainc.add_argument("--aug-blur-prob", type=float, default=0.05)
    trainc.add_argument("--aug-sharpen-prob", type=float, default=0.05)
    trainc.add_argument("--aug-jitter-px", type=int, default=4)
    trainc.add_argument("--aug-scale", type=float, default=0.04)
    trainc.add_argument("--aug-channel-dropout", type=float, default=0.10)
    trainc.add_argument("--image-cache-dir", default=str(CHARPOST_DIR / "image_cache"),
                        help="disk cache for constructed tile1 RGB inputs; empty disables")
    trainc.add_argument("--require-image-cache", action="store_true",
                        help="fail instead of constructing missing RGB inputs during training")
    trainc.add_argument("--epochs", type=int, default=3)
    trainc.add_argument("--max-steps", type=int, default=3000)
    trainc.add_argument("--batch", type=int, default=32)
    trainc.add_argument("--eval-batch", type=int, default=64)
    trainc.add_argument("--grad-accum", type=int, default=2)
    trainc.add_argument("--lr", type=float, default=2e-5)
    trainc.add_argument("--encoder-lr", type=float, default=None,
                        help="optional encoder AdamW LR; defaults to --lr")
    trainc.add_argument("--head-lr", type=float, default=None,
                        help="optional classifier-head AdamW LR; defaults to --lr")
    trainc.add_argument("--weight-decay", type=float, default=1e-4)
    trainc.add_argument("--class-weight", choices=("none", "sqrt", "balanced"), default="sqrt")
    trainc.add_argument("--eval-steps", type=int, default=250,
                        help="validation interval in optimizer steps; 0 disables mid-run eval")
    trainc.add_argument("--save-during-training", action=argparse.BooleanOptionalAction, default=False,
                        help="save whenever val improves; default false avoids Jupyter/PyTorch CUDA allocator crashes after mid-run saves")
    trainc.add_argument("--checkpoint-steps", type=int, default=0,
                        help="save numbered checkpoint-<step> snapshots with validation metrics; 0 disables")
    trainc.add_argument("--save-train-state", action=argparse.BooleanOptionalAction, default=True,
                        help="save optimizer/scaler/RNG/global-step state for exact continuation")
    trainc.add_argument("--state-steps", type=int, default=250,
                        help="overwrite the resumable checkpoint/state every N optimizer steps; 0 saves only at the end")
    trainc.add_argument("--log-steps", type=int, default=50)
    trainc.add_argument("--workers", type=int, default=0)
    trainc.add_argument("--prefetch-factor", type=int, default=2)
    trainc.add_argument("--device", default="auto")
    trainc.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    trainc.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    trainc.add_argument("--input-size", type=int, default=0,
                        help="override processor square size; 0 uses checkpoint processor default")
    trainc.add_argument("--interpolate-pos-encoding", action=argparse.BooleanOptionalAction, default=False,
                        help="pass interpolate_pos_encoding=True to ViT; auto-enabled when --input-size is nonzero")
    trainc.add_argument("--teacher-logits", default="",
                        help="comma-separated train-split .npz logits to average as a distillation teacher")
    trainc.add_argument("--distill-alpha", type=float, default=0.0,
                        help="weight on KL distillation loss; 0 disables")
    trainc.add_argument("--distill-temperature", type=float, default=2.0)
    trainc.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="resume from --out-dir if encoder/head artifacts already exist")
    trainc.add_argument("--train-splits", default="train",
                        help="comma-separated labeled splits for classifier fitting; OOF fold exclusion requires train")
    trainc.add_argument("--val-split", default="val",
                        help="held-out split for validation metrics when --val-fold-id is not used")
    trainc.add_argument("--train-limit", type=int, default=0)
    trainc.add_argument("--val-limit", type=int, default=0)
    trainc.add_argument("--folds", default=str(DEFAULT_FOLDS))
    trainc.add_argument("--train-exclude-fold-id", type=int, default=None,
                        help="train on official train split excluding this fold's bases")
    trainc.add_argument("--val-fold-id", type=int, default=None,
                        help="validate on this fold's bases from the official train split")
    trainc.set_defaults(func=command_train_classifier)

    clog = sub.add_parser("cache-logits", help="cache logits from a fine-tuned char classifier")
    clog.add_argument("--tag", default="charpost_t1L_ft")
    clog.add_argument("--model-dir", default=str(CHARPOST_DIR / "charpost_t1L_ft"))
    clog.add_argument("--init-ckpt", default="")
    clog.add_argument("--tile", type=int, default=1)
    clog.add_argument("--splits", default="val,test")
    clog.add_argument("--batch", type=int, default=64)
    clog.add_argument("--device", default="auto")
    clog.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    clog.add_argument("--limit", type=int, default=0)
    clog.add_argument("--tta", choices=TTA_MODES, default="none",
                      help="fixed test-time augmentation over crop translations/scales before averaging logits")
    clog.add_argument("--tta-pixels", type=int, default=4)
    clog.add_argument("--tta-scale", type=float, default=0.04)
    clog.add_argument("--image-cache-dir", default=str(CHARPOST_DIR / "image_cache"),
                      help="disk cache for constructed tile1 RGB inputs; empty disables")
    clog.add_argument("--require-image-cache", action="store_true",
                      help="fail instead of constructing missing RGB inputs during logit caching")
    clog.add_argument("--folds", default=str(DEFAULT_FOLDS))
    clog.add_argument("--include-fold-id", type=int, default=None,
                      help="cache only this fold's bases from the requested source split")
    clog.add_argument("--exclude-fold-id", type=int, default=None,
                      help="cache all except this fold's bases from the requested source split")
    clog.add_argument("--output-split", default="",
                      help="write logits under this split name, e.g. fold0, when slicing a source split")
    clog.set_defaults(func=command_cache_logits)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
