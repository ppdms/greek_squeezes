#!/usr/bin/env python3
"""Candidate tables, ridge scorer fitting, and final charpost decoding."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import char_lattice as CL  # noqa: E402

WORK = Path(os.environ.get("SQUEEZE_WORK_DIR", "data"))
CHARPOST_DIR = WORK / "character_posterior"
OOF_DIR = CHARPOST_DIR / "oof_ranker"
FEATURE_VERSION = "lattice_length_delta_v2"


def log(message: str) -> None:
    print(f"\n=== {time.strftime('%H:%M:%S')} {message} ===", flush=True)


def atomic_write_json(path: Path, obj: Any, **json_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, **json_kwargs), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def fold_tag(prefix: str, fold_id: int) -> str:
    return f"{prefix}_f{fold_id}"


def fold_lm_dir(order: int, fold_id: int) -> Path:
    return OOF_DIR / f"train_lm_order{order}_exclude_f{fold_id}"


def fold_lm_path(order: int, fold_id: int, template: str = "") -> Path:
    if template:
        return Path(template.format(order=order, fold=fold_id, fold_id=fold_id))
    return fold_lm_dir(order, fold_id) / "lm.json"


def row_groups(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(inputs["bases"].tolist(), inputs["row_idx"].tolist())):
        grouped[(str(key[0]), int(key[1]))].append(i)
    rows = []
    for (base, ridx), indices in sorted(grouped.items()):
        indices.sort(key=lambda i: int(inputs["chunk_idx"][i]))
        rows.append({
            "base": base,
            "row_idx": ridx,
            "indices": indices,
            "truth": str(inputs["row_text"][indices[0]]),
        })
    return rows


def lattice_for_row(inputs: dict[str, Any], indices: list[int], char_topk: int = 0) -> list[dict[str, float]]:
    alphabet = inputs["alphabet"]
    log_probs = inputs["log_probs"]
    out = []
    for i in indices:
        vals = [(alphabet[j], float(log_probs[i, j])) for j in range(len(alphabet))]
        if char_topk and char_topk > 0:
            vals.sort(key=lambda item: item[1], reverse=True)
            vals = vals[:char_topk]
        out.append(dict(vals))
    return out


def visual_sum(text: str, lattice: list[dict[str, float]]) -> float:
    total = 0.0
    for ch, pos in zip(text, lattice):
        total += float(pos.get(ch, -1e9))
    return total


def lm_sum(text: str, lm: Any) -> float:
    if not text:
        return 0.0
    return float(lm.mean_logprob(text)) * len(text)


def candidate_features(text: str, lattice: list[dict[str, float]], lms: dict[int, Any]) -> dict[str, float]:
    v_sum = visual_sum(text, lattice)
    rec: dict[str, float] = {
        "visual_sum": v_sum,
        "mean_char_logprob": v_sum / max(len(text), 1),
        "length": float(len(text)),
        "length_delta": float(len(text) - len(lattice)),
    }
    for order, lm in lms.items():
        score = lm_sum(text, lm)
        rec[f"lm{order}_sum"] = score
        rec[f"lm{order}_mean"] = score / max(len(text), 1)
    return rec


def candidate_pool(lattice: list[dict[str, float]], proposal_lms: dict[int, Any], weights: list[float], beam: int,
                   char_topk: int) -> set[str]:
    texts: set[str] = set()
    for text, _score in CL.decode_lattice(lattice, None, 0.0, beam, char_topk=char_topk):
        texts.add(text)
    for lm in proposal_lms.values():
        for weight in weights:
            for text, _score in CL.decode_lattice(lattice, lm, weight, beam, char_topk=char_topk):
                texts.add(text)
    return texts


def command_build_candidates(args: argparse.Namespace) -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    folds = parse_csv_ints(args.folds_ids)
    orders = parse_csv_ints(args.orders)
    proposal_weights = parse_csv_floats(args.proposal_weights)
    if args.row_shard_count < 1:
        raise SystemExit("--row-shard-count must be >= 1")
    if not (0 <= args.row_shard_id < args.row_shard_count):
        raise SystemExit("--row-shard-id must satisfy 0 <= id < --row-shard-count")
    if args.row_shard_count > 1 and args.out == OOF_DIR / "candidates.json":
        args.out = OOF_DIR / f"candidates_shard{args.row_shard_id:02d}-of-{args.row_shard_count:02d}.json"
    records: list[dict[str, Any]] = []
    manifest = {
        "tag_prefix": args.tag_prefix,
        "folds": folds,
        "orders": orders,
        "proposal_weights": proposal_weights,
        "beam": args.beam,
        "char_topk": args.char_topk,
        "row_shard_count": args.row_shard_count,
        "row_shard_id": args.row_shard_id,
        "feature_version": FEATURE_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for fold_id in folds:
        tag = fold_tag(args.tag_prefix, fold_id)
        split = f"fold{fold_id}"
        log(f"building candidates fold={fold_id} tag={tag} split={split}")
        inputs = CL.load_decode_inputs(tag, split, args.device)
        lms = {order: CL.load_charpost_lm(str(fold_lm_path(order, fold_id, args.fold_lm_template))) for order in orders}
        fold_rows = row_groups(inputs)
        selected = [
            (row_no, row)
            for row_no, row in enumerate(fold_rows, 1)
            if (row_no - 1) % args.row_shard_count == args.row_shard_id
        ]
        row_limit = int(getattr(args, "row_limit", 0) or 0)
        if row_limit > 0:
            selected = selected[:row_limit]
        print(
            f"  fold {fold_id}: selected_rows={len(selected)}/{len(fold_rows)} "
            f"row_shard={args.row_shard_id}/{args.row_shard_count}",
            flush=True,
        )
        for done_no, (row_no, row) in enumerate(selected, 1):
            lattice = lattice_for_row(inputs, row["indices"], char_topk=0)
            proposal_lattice = lattice_for_row(inputs, row["indices"], char_topk=args.char_topk)
            texts = candidate_pool(proposal_lattice, lms, proposal_weights, args.beam, args.char_topk)
            truth = row["truth"]
            if truth not in texts and args.include_truth:
                texts.add(truth)
            for cand_idx, text in enumerate(sorted(texts)):
                err = CL.edit_distance(text, truth)
                feats = candidate_features(text, lattice, lms)
                rec: dict[str, Any] = {
                    "fold": fold_id,
                    "tag": tag,
                    "base": row["base"],
                    "row_idx": row["row_idx"],
                    "row_key": f"f{fold_id}:{row['base']}:{row['row_idx']}",
                    "candidate_index": cand_idx,
                    "text": text,
                    "truth": truth,
                    "chars": len(truth),
                    "edit_distance": err,
                    "target_neg_edit": -float(err),
                    "target_neg_cer": -float(err) / max(len(truth), 1),
                    **feats,
                }
                records.append(rec)
            if done_no % 50 == 0 or done_no == len(selected):
                print(
                    f"  fold {fold_id}: shard_rows {done_no}/{len(selected)} "
                    f"source_row={row_no}/{len(fold_rows)} records={len(records)}",
                    flush=True,
                )

    by_row: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_row[str(rec["row_key"])].append(idx)
    manifest["rows"] = len(by_row)
    manifest["candidates"] = len(records)

    out = {"manifest": manifest, "records": records}
    atomic_write_json(args.out, out, indent=2, sort_keys=True)
    print(f"wrote {args.out} rows={len(by_row)} candidates={len(records)}")


def command_merge_candidates(args: argparse.Namespace) -> None:
    paths = sorted({Path(path) for pattern in args.inputs.split(",") for path in glob.glob(pattern.strip())})
    if not paths:
        raise SystemExit(f"no candidate shard files matched: {args.inputs}")
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in paths:
        manifest, shard_records = load_candidate_records(path)
        manifests.append({"path": str(path), **manifest})
        records.extend(shard_records)
        print(f"read {path} records={len(shard_records)}", flush=True)
    shard_counts = {int(m["row_shard_count"]) for m in manifests if int(m.get("row_shard_count", 1)) > 1}
    expected_shards = int(args.expect_shards or 0)
    if shard_counts:
        if len(shard_counts) != 1:
            raise SystemExit(f"inconsistent row_shard_count values: {sorted(shard_counts)}")
        expected_shards = expected_shards or next(iter(shard_counts))
    if expected_shards:
        ids = {int(m.get("row_shard_id", -1)) for m in manifests}
        expected_ids = set(range(expected_shards))
        if ids != expected_ids:
            raise SystemExit(f"candidate shards incomplete: got ids={sorted(ids)} expected={sorted(expected_ids)}")
        if len(paths) != expected_shards:
            raise SystemExit(f"candidate shard count mismatch: got {len(paths)} expected {expected_shards}")

    by_row: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_row[str(rec["row_key"])].append(idx)
    if args.expect_rows and len(by_row) != args.expect_rows:
        raise SystemExit(f"merged row count mismatch: got {len(by_row)} expected {args.expect_rows}")

    out = {
        "manifest": {
            "kind": "merged_candidate_shards",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": manifests,
            "feature_version": FEATURE_VERSION,
            "rows": len(by_row),
            "candidates": len(records),
        },
        "records": records,
    }
    atomic_write_json(args.out, out, indent=2, sort_keys=True)
    print(f"wrote {args.out} rows={len(by_row)} candidates={len(records)}")


def load_candidate_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("manifest", {}), list(data.get("records", []))


def candidate_manifest_feature_version_ok(manifest: dict[str, Any]) -> bool:
    if manifest.get("kind") == "merged_candidate_shards":
        inputs = list(manifest.get("inputs") or [])
        return (
            manifest.get("feature_version") == FEATURE_VERSION
            and bool(inputs)
            and all(candidate_manifest_feature_version_ok(item) for item in inputs)
        )
    return manifest.get("feature_version") == FEATURE_VERSION


def feature_columns(records: list[dict[str, Any]], requested: str) -> list[str]:
    if requested:
        return [part.strip() for part in requested.split(",") if part.strip()]
    cols = ["visual_sum", "mean_char_logprob", "length", "length_delta"]
    lm_cols = sorted({key for rec in records for key in rec if key.startswith("lm") and key.endswith("_sum")})
    return cols + lm_cols


def matrix(records: list[dict[str, Any]], cols: list[str]) -> np.ndarray:
    return np.asarray([[float(rec.get(col, 0.0)) for col in cols] for rec in records], dtype=np.float64)


def pick_by_scores(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    by_row: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_row[str(rec["row_key"])].append(idx)
    total_err = 0
    total_chars = 0
    chosen = []
    oracle_err = 0
    for row_key, indices in by_row.items():
        best_idx = max(indices, key=lambda i: float(scores[i]))
        oracle_idx = min(indices, key=lambda i: int(records[i]["edit_distance"]))
        rec = records[best_idx]
        total_err += int(rec["edit_distance"])
        total_chars += int(rec["chars"])
        oracle_err += int(records[oracle_idx]["edit_distance"])
        chosen.append({
            "row_key": row_key,
            "fold": rec["fold"],
            "base": rec["base"],
            "row_idx": rec["row_idx"],
            "pred": rec["text"],
            "truth": rec["truth"],
            "err": rec["edit_distance"],
            "chars": rec["chars"],
            "score": float(scores[best_idx]),
            "oracle_err": records[oracle_idx]["edit_distance"],
        })
    return {
        "cer": total_err / total_chars if total_chars else 0.0,
        "errors": total_err,
        "chars": total_chars,
        "rows": len(by_row),
        "oracle_errors": oracle_err,
        "oracle_cer": oracle_err / total_chars if total_chars else 0.0,
        "predictions": chosen,
    }


def fit_score_model(method: str, train_records: list[dict[str, Any]], X_train: np.ndarray, cols: list[str], args: argparse.Namespace) -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if method != "ridge":
        raise ValueError(f"unknown method: {method}")
    y_neg = np.asarray([float(r[args.target]) for r in train_records], dtype=np.float64)
    return make_pipeline(StandardScaler(), Ridge(alpha=args.alpha)).fit(X_train, y_neg)


def predict_scores(model: Any, method: str, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def serializable_model(model: Any, method: str, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"method": method, "features": cols}
    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["ridge"]
    out["mean"] = scaler.mean_.tolist()
    out["std"] = scaler.scale_.tolist()
    coef = estimator.coef_
    out["coef"] = coef[0].tolist() if getattr(coef, "ndim", 1) == 2 and coef.shape[0] == 1 else coef.tolist()
    intercept = estimator.intercept_
    out["intercept"] = float(intercept[0]) if hasattr(intercept, "__len__") else float(intercept)
    out["alpha"] = float(estimator.alpha)
    return out


def load_ranker_fit(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fit = data.get("final_fit") or {}
    if not fit:
        raise SystemExit(f"ranker JSON has no final_fit: {path}")
    if fit.get("failed"):
        raise SystemExit(f"ranker final_fit failed for {fit.get('method')}: {fit.get('failed')}")
    if "coef" not in fit:
        raise SystemExit(f"ranker final_fit is not a supported linear scorer: {path}")
    return fit


def score_serialized_fit(fit: dict[str, Any], features: dict[str, float]) -> float:
    cols = list(fit.get("features") or fit.get("cols") or [])
    if not cols:
        raise SystemExit("ranker final_fit has no feature list")
    x = np.asarray([float(features.get(col, 0.0)) for col in cols], dtype=np.float64)
    mean = np.asarray(fit.get("mean", [0.0] * len(cols)), dtype=np.float64)
    std = np.asarray(fit.get("std", [1.0] * len(cols)), dtype=np.float64)
    coef = np.asarray(fit["coef"], dtype=np.float64)
    if coef.ndim != 1:
        raise SystemExit(f"unsupported ranker coef shape: {coef.shape}")
    if not (len(mean) == len(std) == len(coef) == len(cols)):
        raise SystemExit(
            f"ranker shape mismatch: features={len(cols)} mean={len(mean)} std={len(std)} coef={len(coef)}"
        )
    z = (x - mean) / np.maximum(std, 1e-8)
    return float(z @ coef + float(fit.get("intercept", 0.0)))


def lm_paths(orders: list[int], template: str) -> dict[int, Path]:
    paths = {order: Path(template.format(order=order)) for order in orders}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing LM artifact(s): " + ", ".join(missing))
    return paths


def command_decode(args: argparse.Namespace) -> None:
    orders = parse_csv_ints(args.orders)
    proposal_weights = parse_csv_floats(args.proposal_weights)
    if args.row_shard_count < 1:
        raise SystemExit("--row-shard-count must be >= 1")
    if not (0 <= args.row_shard_id < args.row_shard_count):
        raise SystemExit("--row-shard-id must satisfy 0 <= id < --row-shard-count")
    fit = load_ranker_fit(args.ranker)
    paths = lm_paths(orders, args.lm_template)
    lms = {order: CL.load_charpost_lm(str(path)) for order, path in paths.items()}
    inputs = CL.load_decode_inputs(args.tag, args.split, args.device)
    all_rows = row_groups(inputs)
    rows = [
        row
        for row_no, row in enumerate(all_rows, 1)
        if (row_no - 1) % args.row_shard_count == args.row_shard_id
    ]
    row_limit = int(getattr(args, "row_limit", 0) or 0)
    if row_limit > 0:
        rows = rows[:row_limit]
    print(
        f"{args.split}: selected_rows={len(rows)}/{len(all_rows)} "
        f"row_shard={args.row_shard_id}/{args.row_shard_count}",
        flush=True,
    )
    predictions: list[dict[str, Any]] = []
    total_err = 0
    total_chars = 0
    for row_no, row in enumerate(rows, 1):
        lattice = lattice_for_row(inputs, row["indices"], char_topk=0)
        proposal_lattice = lattice_for_row(inputs, row["indices"], char_topk=args.char_topk)
        texts = candidate_pool(proposal_lattice, lms, proposal_weights, args.beam, args.char_topk)
        if not texts:
            texts = {""}
        scored = []
        for cand_idx, text in enumerate(sorted(texts)):
            feats = candidate_features(text, lattice, lms)
            cand = {
                "candidate_index": cand_idx,
                "text": text,
                "score": score_serialized_fit(fit, feats),
            }
            if args.keep_features:
                cand["features"] = feats
            scored.append(cand)
        best = max(scored, key=lambda rec: float(rec["score"]))
        truth = str(row.get("truth", ""))
        pred: dict[str, Any] = {
            "base": row["base"],
            "row_idx": row["row_idx"],
            "pred": best["text"],
            "score": best["score"],
            "candidates": len(scored),
        }
        if args.keep_candidates:
            pred["candidate_scores"] = scored
        if truth and not args.no_score:
            err = CL.edit_distance(str(best["text"]), truth)
            total_err += err
            total_chars += len(truth)
            pred.update({"truth": truth, "err": err, "chars": len(truth)})
        predictions.append(pred)
        if row_no % 50 == 0 or row_no == len(rows):
            print(f"  decoded rows {row_no}/{len(rows)}", flush=True)

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        by_base[str(pred["base"])].append(pred)
    transcripts = []
    for base, base_rows in sorted(by_base.items()):
        base_rows.sort(key=lambda rec: int(rec["row_idx"]))
        transcripts.append({"base": base, "text": "\n".join(str(row["pred"]) for row in base_rows)})
    scored_summary = None
    if total_chars:
        scored_summary = {
            "cer": total_err / total_chars,
            "errors": total_err,
            "chars": total_chars,
            "rows": len(predictions),
        }
    out = {
        "tag": args.tag,
        "split": args.split,
        "source": inputs.get("source"),
        "ranker": str(args.ranker),
        "ranker_method": fit.get("method"),
        "features": list(fit.get("features") or fit.get("cols") or []),
        "orders": orders,
        "lm_paths": {str(order): str(path) for order, path in paths.items()},
        "proposal_weights": proposal_weights,
        "beam": args.beam,
        "char_topk": args.char_topk,
        "row_shard_count": args.row_shard_count,
        "row_shard_id": args.row_shard_id,
        "scored": scored_summary,
        "predictions": predictions,
        "transcripts": transcripts,
    }
    out_path = args.out or (CHARPOST_DIR / f"{args.tag}__{args.split}_decode.json")
    atomic_write_json(out_path, out, indent=2, sort_keys=True)
    if scored_summary:
        print(
            f"{args.split}: CER={scored_summary['cer']:.6f} "
            f"errors={total_err}/{total_chars} rows={len(predictions)}"
        )
    else:
        print(f"{args.split}: decoded rows={len(predictions)} bases={len(transcripts)}")
    print(f"wrote {out_path}")


def command_merge_decodes(args: argparse.Namespace) -> None:
    paths = sorted({Path(path) for pattern in args.inputs.split(",") for path in glob.glob(pattern.strip())})
    if not paths:
        raise SystemExit(f"no decode shard files matched: {args.inputs}")
    expected_shards = int(args.expect_shards or 0)
    shard_ids: set[int] = set()
    predictions: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    total_err = 0
    total_chars = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if first is None:
            first = data
        if data.get("tag") != first.get("tag") or data.get("split") != first.get("split"):
            raise SystemExit(f"decode shard does not match tag/split: {path}")
        shard_id = int(data.get("row_shard_id", -1))
        shard_count = int(data.get("row_shard_count", 1))
        if shard_id < 0:
            raise SystemExit(f"decode shard missing row_shard_id: {path}")
        shard_ids.add(shard_id)
        expected_shards = expected_shards or shard_count
        predictions.extend(list(data.get("predictions") or []))
        scored = data.get("scored") or {}
        if scored:
            total_err += int(scored.get("errors") or 0)
            total_chars += int(scored.get("chars") or 0)
        print(f"read {path} rows={len(data.get('predictions') or [])}", flush=True)
    if first is None:
        raise SystemExit("no decode shards loaded")
    if expected_shards and shard_ids != set(range(expected_shards)):
        raise SystemExit(f"decode shards incomplete: got ids={sorted(shard_ids)} expected={list(range(expected_shards))}")
    predictions.sort(key=lambda rec: (str(rec.get("base")), int(rec.get("row_idx", 0))))
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        by_base[str(pred["base"])].append(pred)
    transcripts = []
    for base, base_rows in sorted(by_base.items()):
        base_rows.sort(key=lambda rec: int(rec["row_idx"]))
        transcripts.append({"base": base, "text": "\n".join(str(row["pred"]) for row in base_rows)})
    scored_summary = None
    if total_chars:
        scored_summary = {
            "cer": total_err / total_chars,
            "errors": total_err,
            "chars": total_chars,
            "rows": len(predictions),
        }
    out = {
        **{key: first.get(key) for key in (
            "tag", "split", "source", "ranker", "ranker_method", "features",
            "orders", "lm_paths", "proposal_weights", "beam", "char_topk",
        )},
        "row_shard_count": expected_shards or len(paths),
        "row_shard_paths": [str(path) for path in paths],
        "scored": scored_summary,
        "predictions": predictions,
        "transcripts": transcripts,
    }
    atomic_write_json(args.out, out, indent=2, sort_keys=True)
    if scored_summary:
        print(
            f"{first.get('split')}: CER={scored_summary['cer']:.6f} "
            f"errors={total_err}/{total_chars} rows={len(predictions)}"
        )
    else:
        print(f"{first.get('split')}: decoded rows={len(predictions)} bases={len(transcripts)}")
    print(f"wrote {args.out}")


def command_fit_rankers(args: argparse.Namespace) -> None:
    manifest, records = load_candidate_records(args.candidates)
    if not args.allow_legacy_candidates and not candidate_manifest_feature_version_ok(manifest):
        raise SystemExit(
            f"candidate table is missing feature_version={FEATURE_VERSION}; "
            "rebuild candidates so length_delta does not depend on ground truth"
        )
    cols = feature_columns(records, args.features)
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    folds = sorted({int(rec["fold"]) for rec in records})
    all_results = []
    for method in methods:
        log(f"nested scorer eval: {method}")
        fold_predictions = []
        fold_scores = []
        for fold_id in folds:
            train_records = [rec for rec in records if int(rec["fold"]) != fold_id]
            eval_records = [rec for rec in records if int(rec["fold"]) == fold_id]
            X_train = matrix(train_records, cols)
            X_eval = matrix(eval_records, cols)
            try:
                model = fit_score_model(method, train_records, X_train, cols, args)
                scores = predict_scores(model, method, X_eval)
            except Exception as exc:
                print(f"  fold {fold_id}: {method} failed: {exc}", flush=True)
                fold_scores.append({"fold": fold_id, "failed": str(exc)})
                continue
            picked = pick_by_scores(eval_records, scores)
            fold_predictions.extend(picked["predictions"])
            fold_scores.append({k: v for k, v in picked.items() if k != "predictions"} | {"fold": fold_id})
            print(f"  fold {fold_id}: CER={picked['cer']:.6f} errors={picked['errors']}/{picked['chars']} oracle={picked['oracle_cer']:.6f}", flush=True)
        total_err = sum(int(row["err"]) for row in fold_predictions)
        total_chars = sum(int(row["chars"]) for row in fold_predictions)
        total_oracle = sum(int(row["oracle_err"]) for row in fold_predictions)
        rec = {
            "method": method,
            "cer": total_err / total_chars if total_chars else None,
            "errors": total_err,
            "chars": total_chars,
            "rows": len(fold_predictions),
            "oracle_cer": total_oracle / total_chars if total_chars else None,
            "oracle_errors": total_oracle,
            "folds": fold_scores,
        }
        all_results.append(rec)
        print(f"TOTAL {method}: CER={rec['cer']:.6f} errors={total_err}/{total_chars} oracle={rec['oracle_cer']:.6f}", flush=True)
    all_results.sort(key=lambda r: (math.inf if r["cer"] is None else r["cer"], r["method"]))
    final_fit = None
    if all_results:
        best_method = str(all_results[0]["method"])
        try:
            final_model = fit_score_model(best_method, records, matrix(records, cols), cols, args)
            final_fit = serializable_model(final_model, best_method, cols)
        except Exception as exc:
            final_fit = {"method": best_method, "failed": str(exc)}
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_manifest": manifest,
        "candidates": str(args.candidates),
        "features": cols,
        "target": args.target,
        "methods": methods,
        "scores": all_results,
        "best": all_results[0] if all_results else None,
        "final_fit": final_fit,
    }
    atomic_write_json(
        args.out,
        out,
        indent=2,
        sort_keys=True,
        default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
    )
    print(f"wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    build = sub.add_parser("build-candidates", help="build OOF row-candidate feature table from fold logits")
    build.add_argument("--folds-ids", default="0,1,2,3,4")
    build.add_argument("--orders", default="6,10,12")
    build.add_argument("--fold-lm-template", default="",
                       help="optional LM path template with {order} and {fold}; default uses the OOF ranker cache")
    build.add_argument("--tag-prefix", default="oof")
    build.add_argument("--proposal-weights", default="0,0.25,0.5,0.75,1.0")
    build.add_argument("--beam", type=int, default=256)
    build.add_argument("--char-topk", type=int, default=8)
    build.add_argument("--device", default="auto")
    build.add_argument("--row-shard-count", type=int, default=1,
                       help="split candidate building by source row modulo this count")
    build.add_argument("--row-shard-id", type=int, default=0,
                       help="which row shard to build; valid range is [0, row_shard_count)")
    build.add_argument("--row-limit", type=int, default=0,
                       help="debug/smoke only: process at most this many selected rows per fold")
    build.add_argument("--include-truth", action=argparse.BooleanOptionalAction, default=False,
                       help="debug/oracle only; do not enable for scorer fitting")
    build.add_argument("--out", type=Path, default=OOF_DIR / "candidates.json")
    build.set_defaults(func=command_build_candidates)

    merge = sub.add_parser("merge-candidates", help="merge row-sharded candidate JSON files")
    merge.add_argument("--inputs", default=str(OOF_DIR / "candidates_shard*.json"),
                       help="comma-separated glob(s) for candidate shard JSON files")
    merge.add_argument("--expect-shards", type=int, default=0,
                       help="fail unless exactly these shard ids [0..N) are present; inferred from shard manifests when possible")
    merge.add_argument("--expect-rows", type=int, default=0,
                       help="fail unless merged candidate rows equal this count")
    merge.add_argument("--out", type=Path, default=OOF_DIR / "candidates.json")
    merge.set_defaults(func=command_merge_candidates)

    fit = sub.add_parser("fit-rankers", help="nested fold-heldout evaluation of the ridge row-candidate scorer")
    fit.add_argument("--candidates", type=Path, default=OOF_DIR / "candidates.json")
    fit.add_argument("--out", type=Path, default=OOF_DIR / "ranker_scores.json")
    fit.add_argument("--features", default="",
                     help="comma-separated feature list; default uses visual + train-only n-gram features")
    fit.add_argument("--target", choices=("target_neg_edit", "target_neg_cer"), default="target_neg_edit")
    fit.add_argument("--methods", default="ridge")
    fit.add_argument("--alpha", type=float, default=1.0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--allow-legacy-candidates", action="store_true",
                     help="allow candidate tables without the current length_delta feature marker")
    fit.set_defaults(func=command_fit_rankers)

    dec = sub.add_parser("decode", help="apply the selected charpost ranker to cached final logits")
    dec.add_argument("--tag", default="all_train")
    dec.add_argument("--split", default="test")
    dec.add_argument("--ranker", type=Path, default=OOF_DIR / "clean_ranker_scores.json")
    dec.add_argument("--orders", default="6,10,12")
    dec.add_argument("--lm-template", default=str(CHARPOST_DIR / "train_lm_order{order}_train" / "lm.json"))
    dec.add_argument("--proposal-weights", default="0,0.25,0.5,0.75,1.0")
    dec.add_argument("--beam", type=int, default=256)
    dec.add_argument("--char-topk", type=int, default=8)
    dec.add_argument("--device", default="auto")
    dec.add_argument("--row-limit", type=int, default=0,
                     help="debug/smoke only: decode at most this many rows")
    dec.add_argument("--row-shard-count", type=int, default=1,
                     help="split final decode rows by row index modulo this count")
    dec.add_argument("--row-shard-id", type=int, default=0,
                     help="which decode row shard to run; valid range is [0, row_shard_count)")
    dec.add_argument("--no-score", action="store_true", help="skip CER calculation even if cached truth text is present")
    dec.add_argument("--keep-candidates", action="store_true", help="include every candidate score in output JSON")
    dec.add_argument("--keep-features", action="store_true", help="include per-candidate feature maps when --keep-candidates is used")
    dec.add_argument("--out", type=Path, default=None)
    dec.set_defaults(func=command_decode)

    merge_dec = sub.add_parser("merge-decodes", help="merge row-sharded final decode JSON files")
    merge_dec.add_argument("--inputs", required=True,
                           help="comma-separated glob(s) for row-sharded decode JSON files")
    merge_dec.add_argument("--expect-shards", type=int, default=0,
                           help="fail unless exactly these shard ids [0..N) are present")
    merge_dec.add_argument("--out", type=Path, required=True)
    merge_dec.set_defaults(func=command_merge_decodes)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
