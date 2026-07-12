#!/usr/bin/env python3
"""Candidate tables, OOF interpolation tuning, and final charpost decoding."""
from __future__ import annotations

import argparse
import glob
import json
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
FEATURE_VERSION = "ppm_interpolation_v2"
SELECTOR_METHOD = "visual_ppm_interpolation"
SELECTOR_FEATURES = ["visual_sum", "ppm_sum"]


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
    # The one-parameter selector uses exactly the two quantities that vary
    # among candidates in a row. The same PPM-C model also drives proposals.
    ppm = CL.ppm_c_sum(lms[max(lms)], text)
    return {"visual_sum": v_sum, "ppm_sum": ppm}


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
        lms = {order: CL.CharPostPPMLM(CL.load_charpost_lm(str(fold_lm_path(order, fold_id, args.fold_lm_template))))
               for order in orders}
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
    cols = [part.strip() for part in requested.split(",") if part.strip()] if requested else SELECTOR_FEATURES
    if cols != SELECTOR_FEATURES:
        raise ValueError(f"selector features must be {','.join(SELECTOR_FEATURES)}; got {','.join(cols)}")
    missing = [col for col in cols if not all(col in rec for rec in records)]
    if missing:
        raise ValueError(f"candidate table is missing selector feature(s): {', '.join(missing)}")
    return cols


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


def interpolation_scores(records: list[dict[str, Any]], lm_weight: float) -> np.ndarray:
    """Score candidates as visual_sum + lm_weight * ppm_sum."""
    return np.asarray(
        [float(rec["visual_sum"]) + lm_weight * float(rec["ppm_sum"]) for rec in records],
        dtype=np.float64,
    )


def interpolation_grid_metrics(records: list[dict[str, Any]], lm_weights: np.ndarray) -> dict[str, Any]:
    """Evaluate a weight grid efficiently, retaining corpus-level CER counts."""
    by_row: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_row[str(rec["row_key"])].append(idx)
    visual = np.asarray([float(rec["visual_sum"]) for rec in records], dtype=np.float64)
    ppm = np.asarray([float(rec["ppm_sum"]) for rec in records], dtype=np.float64)
    edit = np.asarray([int(rec["edit_distance"]) for rec in records], dtype=np.int64)
    errors = np.zeros(len(lm_weights), dtype=np.int64)
    oracle_errors = 0
    chars = 0
    for indices in by_row.values():
        idx = np.asarray(indices, dtype=np.int64)
        row_scores = visual[idx, None] + ppm[idx, None] * lm_weights[None, :]
        winners = idx[np.argmax(row_scores, axis=0)]
        errors += edit[winners]
        oracle_errors += int(np.min(edit[idx]))
        chars += int(records[indices[0]]["chars"])
    return {
        "errors": errors,
        "chars": chars,
        "rows": len(by_row),
        "oracle_errors": oracle_errors,
        "oracle_cer": oracle_errors / chars if chars else 0.0,
    }


def best_grid_index(errors: np.ndarray) -> int:
    """Choose the lowest weight among equal-CER grid points."""
    if errors.size == 0:
        raise ValueError("empty interpolation grid")
    return int(np.flatnonzero(errors == np.min(errors))[0])


def load_selector(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selector = data.get("final_selector") or {}
    if not selector:
        raise SystemExit(f"selector JSON has no final_selector: {path}")
    if selector.get("method") != SELECTOR_METHOD:
        raise SystemExit(f"unsupported selector method {selector.get('method')!r}: {path}")
    if selector.get("features") != SELECTOR_FEATURES or "lm_weight" not in selector:
        raise SystemExit(f"selector JSON has an invalid interpolation specification: {path}")
    return selector


def score_serialized_selector(selector: dict[str, Any], features: dict[str, float]) -> float:
    return float(features["visual_sum"]) + float(selector["lm_weight"]) * float(features["ppm_sum"])


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
    selector = load_selector(args.ranker)
    paths = lm_paths(orders, args.lm_template)
    lms = {order: CL.CharPostPPMLM(CL.load_charpost_lm(str(path))) for order, path in paths.items()}
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
                "score": score_serialized_selector(selector, feats),
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
        "ranker_method": selector.get("method"),
        "features": list(selector["features"]),
        "lm_weight": float(selector["lm_weight"]),
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
            "tag", "split", "source", "ranker", "ranker_method", "features", "lm_weight",
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
            "rebuild candidates for the interpolation selector"
        )
    cols = feature_columns(records, args.features)
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    if methods != [SELECTOR_METHOD]:
        raise SystemExit(f"--methods must be {SELECTOR_METHOD}")
    if args.lambda_steps < 2 or args.lambda_max <= args.lambda_min or args.lambda_min < 0:
        raise SystemExit("lambda grid requires 0 <= --lambda-min < --lambda-max and --lambda-steps >= 2")
    lm_weights = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps, dtype=np.float64)
    folds = sorted({int(rec["fold"]) for rec in records})
    if len(folds) < 2:
        raise SystemExit("nested OOF tuning requires at least two folds")

    log(f"nested selector eval: {SELECTOR_METHOD}")
    fold_records = {fold: [rec for rec in records if int(rec["fold"]) == fold] for fold in folds}
    fold_curves = {fold: interpolation_grid_metrics(fold_records[fold], lm_weights) for fold in folds}
    fold_predictions: list[dict[str, Any]] = []
    fold_scores = []
    for fold_id in folds:
        train_errors = sum(
            (fold_curves[fold]["errors"] for fold in folds if fold != fold_id),
            start=np.zeros(len(lm_weights), dtype=np.int64),
        )
        train_chars = sum(int(fold_curves[fold]["chars"]) for fold in folds if fold != fold_id)
        grid_idx = best_grid_index(train_errors)
        lm_weight = float(lm_weights[grid_idx])
        eval_records = fold_records[fold_id]
        picked = pick_by_scores(eval_records, interpolation_scores(eval_records, lm_weight))
        fold_predictions.extend(picked["predictions"])
        fold_scores.append({
            **{k: v for k, v in picked.items() if k != "predictions"},
            "fold": fold_id,
            "lm_weight": lm_weight,
            "train_errors": int(train_errors[grid_idx]),
            "train_chars": train_chars,
            "train_cer": float(train_errors[grid_idx] / train_chars) if train_chars else None,
        })
        print(
            f"  fold {fold_id}: lambda={lm_weight:.4f} CER={picked['cer']:.6f} "
            f"errors={picked['errors']}/{picked['chars']} oracle={picked['oracle_cer']:.6f}",
            flush=True,
        )

    total_err = sum(int(row["err"]) for row in fold_predictions)
    total_chars = sum(int(row["chars"]) for row in fold_predictions)
    total_oracle = sum(int(row["oracle_err"]) for row in fold_predictions)
    result = {
        "method": SELECTOR_METHOD,
        "cer": total_err / total_chars if total_chars else None,
        "errors": total_err,
        "chars": total_chars,
        "rows": len(fold_predictions),
        "oracle_cer": total_oracle / total_chars if total_chars else None,
        "oracle_errors": total_oracle,
        "folds": fold_scores,
    }
    print(
        f"TOTAL {SELECTOR_METHOD}: CER={result['cer']:.6f} "
        f"errors={total_err}/{total_chars} oracle={result['oracle_cer']:.6f}",
        flush=True,
    )

    all_errors = sum(
        (fold_curves[fold]["errors"] for fold in folds),
        start=np.zeros(len(lm_weights), dtype=np.int64),
    )
    final_idx = best_grid_index(all_errors)
    final_weight = float(lm_weights[final_idx])
    final_selector = {
        "method": SELECTOR_METHOD,
        "features": cols,
        "formula": "visual_sum + lm_weight * ppm_sum",
        "lm_weight": final_weight,
        "objective": "OOF corpus CER",
        "tie_break": "lowest lm_weight",
    }
    tuning_curve = [
        {
            "lm_weight": float(weight),
            "errors": int(errors),
            "chars": total_chars,
            "cer": float(errors / total_chars) if total_chars else None,
        }
        for weight, errors in zip(lm_weights, all_errors)
    ]
    out = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_manifest": manifest,
        "candidates": str(args.candidates),
        "features": cols,
        "methods": methods,
        "objective": "OOF corpus CER",
        "lambda_grid": {
            "min": float(args.lambda_min),
            "max": float(args.lambda_max),
            "steps": int(args.lambda_steps),
        },
        "tuning_curve": tuning_curve,
        "scores": [result],
        "best": result,
        "final_selector": final_selector,
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
    build.add_argument("--orders", default="12")
    build.add_argument("--fold-lm-template", default="",
                       help="optional LM path template with {order} and {fold}; default uses the OOF ranker cache")
    build.add_argument("--tag-prefix", default="oof")
    build.add_argument("--proposal-weights", default="1.0")
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

    fit = sub.add_parser("fit-rankers", help="nested OOF tuning of visual/PPM interpolation")
    fit.add_argument("--candidates", type=Path, default=OOF_DIR / "candidates.json")
    fit.add_argument("--out", type=Path, default=OOF_DIR / "ranker_scores.json")
    fit.add_argument("--features", default=",".join(SELECTOR_FEATURES),
                     help="must be visual_sum,ppm_sum")
    fit.add_argument("--methods", default=SELECTOR_METHOD)
    fit.add_argument("--lambda-min", type=float, default=0.0)
    fit.add_argument("--lambda-max", type=float, default=4.0)
    fit.add_argument("--lambda-steps", type=int, default=401,
                     help="inclusive linear grid; equal-CER ties choose the lowest weight")
    fit.add_argument("--allow-legacy-candidates", action="store_true",
                     help="allow candidate tables without the current feature-version marker")
    fit.set_defaults(func=command_fit_rankers)

    dec = sub.add_parser("decode", help="apply the selected charpost ranker to cached final logits")
    dec.add_argument("--tag", default="all_train")
    dec.add_argument("--split", default="test")
    dec.add_argument("--ranker", type=Path, default=OOF_DIR / "clean_ranker_scores.json")
    dec.add_argument("--orders", default="12")
    dec.add_argument("--lm-template", default=str(CHARPOST_DIR / "train_lm_order{order}_train" / "lm.json"))
    dec.add_argument("--proposal-weights", default="1.0")
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
