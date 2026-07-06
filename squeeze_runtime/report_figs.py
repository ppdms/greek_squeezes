#!/usr/bin/env python3
"""Report figure builders for the charpost mechanism walkthrough.

Every function writes a PNG under FIGS_DIR (default ``report/figs``) so the
LaTeX report can pick the file up on the next compile. Every builder here
reads the cached `data/` tree and raises FileNotFoundError when the required
artifact is missing; use ``maybe(fn)`` in the notebook so a missing artifact
skips the figure instead of failing the run.

`find_example_row` scans the cached final decode for a real row where the
ranker overturns the visual-only read, and its result is shared by
`mechanism_walkthrough`, `fig_lattice_graph`, and `fig_lattice_example` so the
report's prose, the schematic lattice graph, and the full-alphabet heatmap
all describe the same concrete example.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import char_lattice as CL  # noqa: E402
import charpost_ranker as CR  # noqa: E402

WORK = Path(os.environ.get("SQUEEZE_WORK_DIR", "data"))
FIGS_DIR = Path("report") / "figs"

# Frozen production candidate-search recipe (see ArtifactChecks defaults).
PROPOSAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BEAM = 256
CHAR_TOPK = 8


def _out_path(name: str, out: Path | str | None) -> Path:
    path = Path(out) if out else FIGS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: Any, name: str, out: Path | str | None) -> Path:
    path = _out_path(name, out)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")
    return path


def maybe(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a figure builder; report and skip when its artifacts are missing."""
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        print(f"[skip] {getattr(fn, '__name__', fn)}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Artifact-backed figures.
# ---------------------------------------------------------------------------

def _charpost_dir() -> Path:
    return Path(CR.CHARPOST_DIR)


def _ranker_dir() -> Path:
    return Path(CR.OOF_DIR)


def _position_ruler(n_pos: int, indent: int = 0) -> list[str]:
    """Two monospace lines (tens digit, ones digit) so a reader can read off
    which row position a character in an aligned block of candidate/lattice
    text sits at, e.g. position 10 under a two-digit column."""
    tens = "".join(str(i // 10) if i >= 10 else " " for i in range(n_pos))
    ones = "".join(str(i % 10) for i in range(n_pos))
    pad = " " * indent
    return [pad + tens, pad + ones]


def _prod_run_dir(slug: str) -> Path:
    """``--upload-prod-result-only`` stages summary.json/logits.npz/etc. here."""
    return WORK / "prod_runs" / slug


def _load_prod_cer(slug: str = "trogs26-test") -> float | None:
    """Read the real production CER from the staged run summary."""
    path = _prod_run_dir(slug) / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # _prod_summary (modal_app.py) nests the score under "scored", the same
    # {"cer", "errors", "chars", "rows"} shape command_decode writes.
    cer = (data.get("scored") or {}).get("cer")
    return float(cer) if cer is not None else None


def _load_prod_visual_cer(slug: str = "trogs26-test") -> float | None:
    """Compute the production visual-only-argmax CER from the staged logits.

    Needs no separate "ranker disabled" production run: charpost_ranker's
    command_decode builds each row's full lattice before the ranker ever
    runs, so the visual-only argmax is recoverable from the same logits
    already staged for the deployed decode -- the same dataset as
    ``_load_prod_cer``, so the two bars are apples-to-apples.
    """
    path = _prod_run_dir(slug) / "logits.npz"
    if not path.exists():
        return None
    inputs = _load_logits_npz_path(path)
    err = chars = 0
    for row in CR.row_groups(inputs):
        truth = row["truth"]
        if not truth:
            continue
        lattice = CR.lattice_for_row(inputs, row["indices"], char_topk=0)
        visual = "".join(max(pos.items(), key=lambda kv: kv[1])[0] for pos in lattice)
        err += CL.edit_distance(visual, truth)
        chars += len(truth)
    return err / chars if chars else None


def _load_logits_npz_path(path: Path) -> dict[str, Any]:
    """np.load cached logits from an explicit path (log-softmax in numpy, no torch)."""
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(f"missing cached logits: {path}")
    data = np.load(path, allow_pickle=True)
    logits = data["logits"].astype(np.float64)
    log_probs = logits - np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) - logits.max(axis=1, keepdims=True)
    return {
        "log_probs": log_probs,
        "labels": data["labels"].astype(int),
        "bases": data["bases"],
        "row_idx": data["row_idx"],
        "chunk_idx": data["chunk_idx"],
        "row_text": data["row_text"],
        "alphabet": "".join(str(x) for x in data["alphabet"].tolist()),
    }


def _load_logits_npz(tag: str, split: str) -> dict[str, Any]:
    """np.load the cached logits for a tag/split, without importing torch."""
    return _load_logits_npz_path(Path(CL.logits_path(tag, split)))


def fig_cer_ladder(candidates_json: Path | str | None = None,
                   ranker_json: Path | str | None = None,
                   production_cer: float | None = "auto",  # type: ignore[assignment]
                   prod_slug: str = "trogs26-test",
                   out: Path | str | None = None) -> Path:
    """OOF decode ladder: visual-only pick vs fitted ridge pick vs lattice
    oracle, computed from the frozen candidate table; plus, when available,
    the production visual-only and ranker CERs on the same external dataset.

    ``production_cer="auto"`` (the default) reads the real CER from
    ``data/prod_runs/<prod_slug>/summary.json``. If that artifact has not
    been synced locally, the production-ranker bar is omitted entirely --
    there is no hardcoded stand-in value. Pass an explicit float to override,
    or ``None`` to omit that bar unconditionally. The production visual-only
    bar is computed from the cached production logits (see
    ``_load_prod_visual_cer``) and is likewise omitted when unavailable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if production_cer == "auto":
        production_cer = _load_prod_cer(prod_slug)
    prod_visual_cer = _load_prod_visual_cer(prod_slug)

    cand_path = Path(candidates_json) if candidates_json else _ranker_dir() / "candidates.json"
    rank_path = Path(ranker_json) if ranker_json else _ranker_dir() / "ranker_scores.json"
    if not cand_path.exists():
        raise FileNotFoundError(str(cand_path))
    if not rank_path.exists():
        raise FileNotFoundError(str(rank_path))
    _, records = CR.load_candidate_records(cand_path)
    by_row: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_row.setdefault(str(rec["row_key"]), []).append(rec)
    vis_err = chars = oracle_err = 0
    for rows in by_row.values():
        vis = max(rows, key=lambda r: float(r["visual_sum"]))
        vis_err += int(vis["edit_distance"])
        oracle_err += min(int(r["edit_distance"]) for r in rows)
        chars += int(rows[0]["chars"])
    rank = json.loads(rank_path.read_text(encoding="utf-8"))
    best = rank.get("best") or {}
    labels = ["visual-only\nargmax (OOF)", "ridge ranker\n(OOF)", "lattice oracle\n(OOF, best candidate)"]
    values = [vis_err / max(chars, 1), float(best.get("cer") or 0.0), oracle_err / max(chars, 1)]
    colors = ["#4878A8", "#C8961E", "#7A9A6D"]
    if prod_visual_cer is not None:
        labels.append("visual-only\nargmax (prod)")
        values.append(float(prod_visual_cer))
        colors.append("#7793B0")
    if production_cer is not None:
        labels.append("ridge ranker\n(prod)")
        values.append(float(production_cer))
        colors.append("#8A5A83")
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CER")
    ax.set_title("Where the charpost layer earns its CER "
                 "(first 3 bars: OOF folds; last bars: trogs26-test production)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = _save(fig, "charpost_cer_ladder.png", out)
    plt.close(fig)
    return path


def fig_ranker_weights(ranker_json: Path | str | None = None,
                       out: Path | str | None = None) -> Path:
    """Standardized ridge coefficients of the production row ranker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank_path = Path(ranker_json) if ranker_json else _ranker_dir() / "ranker_scores.json"
    if not rank_path.exists():
        raise FileNotFoundError(str(rank_path))
    fit = CR.load_ranker_fit(rank_path)
    cols = list(fit["features"])
    coef = [float(c) for c in fit["coef"]]
    order = sorted(range(len(cols)), key=lambda i: coef[i])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.barh([cols[i] for i in order], [coef[i] for i in order],
            color=["#4878A8" if coef[i] >= 0 else "#A85454" for i in order])
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("standardized ridge coefficient (target: -edit distance)")
    ax.set_title("Learned row-ranker weights over candidate features", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = _save(fig, "charpost_ranker_weights.png", out)
    plt.close(fig)
    return path


def fig_confusion(tag_prefix: str = "oof", folds: tuple[int, ...] = (0, 1, 2, 3, 4),
                  out: Path | str | None = None) -> Path:
    """Aggregate OOF classifier confusion matrix over the 27-symbol alphabet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    alphabet = None
    mat = None
    for fold in folds:
        inputs = _load_logits_npz(f"{tag_prefix}_f{fold}", f"fold{fold}")
        alphabet = inputs["alphabet"]
        n = len(alphabet)
        if mat is None:
            mat = np.zeros((n, n), dtype=np.int64)
        pred = inputs["log_probs"].argmax(axis=1)
        for t, p in zip(inputs["labels"], pred):
            mat[int(t), int(p)] += 1
    assert mat is not None and alphabet is not None
    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    norm = mat / row_sums
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(np.log10(norm + 1e-4), cmap="viridis")
    ax.set_xticks(range(len(alphabet)), list(alphabet), fontsize=8)
    ax.set_yticks(range(len(alphabet)), list(alphabet), fontsize=8)
    ax.set_xlabel("predicted proxy character")
    ax.set_ylabel("true proxy character")
    ax.set_title("OOF charpost classifier confusion (row-normalized, log scale)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log10 P(pred | true)")
    off = mat.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:8], off.shape))[0]
    pairs = ", ".join(f"{alphabet[t]}->{alphabet[p]} ({off[t, p]})" for t, p in top)
    top1 = mat.trace() / mat.sum()
    fig.tight_layout()
    # Reserve a dedicated strip below the (already laid-out) xlabel/ticks and
    # place the diagnostic line there in figure coordinates. tight_layout()
    # doesn't know about this manually added fig.text, so the margin has to
    # be widened after the fact or this line lands on top of the tick labels.
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.07)
    fig.text(0.02, 0.015, f"top-1 {top1:.4f}; top confusions: {pairs}",
             ha="left", va="bottom", fontsize=7.5, transform=fig.transFigure)
    path = _save(fig, "charpost_confusion.png", out)
    plt.close(fig)
    return path


def find_example_row(tag: str = "all_train", split: str = "test",
                     ranker_json: Path | str | None = None,
                     lm_template: str | None = None,
                     max_rows_scanned: int = 200) -> dict[str, Any]:
    """Scan cached final-decode logits for a real row that makes the
    mechanism visible: the ranker overturns the visual-only argmax and lands
    on the ground truth. Falls back to any row where they disagree, then to
    the first scanned row.

    The returned dict (lattice, candidate pool, features, scores, the three
    read-outs, and the LMs used) is shared by `mechanism_walkthrough`,
    `fig_lattice_graph`, and `fig_lattice_example` so the report's narrative
    and both figures describe the same concrete row.
    """
    rank_path = Path(ranker_json) if ranker_json else _ranker_dir() / "ranker_scores.json"
    if not rank_path.exists():
        raise FileNotFoundError(str(rank_path))
    fit = CR.load_ranker_fit(rank_path)
    template = lm_template or str(_charpost_dir() / "train_lm_order{order}_train" / "lm.json")
    orders = (6, 10, 12)
    lms = {order: CL.load_charpost_lm(template.format(order=order)) for order in orders}
    inputs = _load_logits_npz(tag, split)
    rows = CR.row_groups(inputs)[:max_rows_scanned]
    if not rows:
        raise FileNotFoundError(f"no rows found in cached logits for {tag}/{split}")

    def score_row(row: dict[str, Any]) -> dict[str, Any]:
        lattice = CR.lattice_for_row(inputs, row["indices"], char_topk=0)
        proposal = CR.lattice_for_row(inputs, row["indices"], char_topk=CHAR_TOPK)
        pool = CR.candidate_pool(proposal, lms, list(PROPOSAL_WEIGHTS), BEAM, CHAR_TOPK)
        feats = {t: CR.candidate_features(t, lattice, lms) for t in pool}
        scores = {t: CR.score_serialized_fit(fit, feats[t]) for t in pool}
        chosen = max(pool, key=lambda t: scores[t])
        visual = "".join(max(pos.items(), key=lambda kv: kv[1])[0] for pos in lattice)
        return {"row": row, "lattice": lattice, "pool": pool, "feats": feats,
                "scores": scores, "chosen": chosen, "visual": visual,
                "truth": str(row["truth"]), "alphabet": inputs["alphabet"], "lms": lms}

    fallback = None
    for row in rows:
        rec = score_row(row)
        if rec["chosen"] != rec["visual"]:
            fallback = fallback or rec
            if rec["truth"] and rec["chosen"] == rec["truth"]:
                return rec
    return fallback or score_row(rows[0])


def mechanism_walkthrough(rec: dict[str, Any] | None = None, **find_kwargs: Any) -> dict[str, Any]:
    """Print the full charpost decode mechanism on a real row.

    Uses `find_example_row(**find_kwargs)` when `rec` is not supplied; pass a
    `rec` from a prior call to narrate the same row used by the figures.
    """
    rec = rec or find_example_row(**find_kwargs)
    lattice, truth, visual, chosen = rec["lattice"], rec["truth"], rec["visual"], rec["chosen"]
    base, row_idx = rec["row"]["base"], rec["row"]["row_idx"]
    print(f"Row: {base} row {row_idx}  (truth: {truth!r})")

    print("\nStep 1 - visual lattice (per-position classifier log-probs, top 4):")
    for i, pos in enumerate(lattice):
        ranked = sorted(pos.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"  pos {i}: " + "  ".join(f"{ch}:{lp:+.2f}" for ch, lp in ranked))

    print("\nStep 2 - visual-only decode (weight 0): argmax per position")
    visual_sum = sum(pos[visual[i]] for i, pos in enumerate(lattice))
    print(f"  -> {visual!r}  visual_sum={visual_sum:+.2f}")

    print("\nStep 3 - LM-weighted beam proposals (order-6 train-only LM):")
    lm6 = rec["lms"][6]
    for w in PROPOSAL_WEIGHTS:
        beams = CL.decode_lattice(lattice, lm6, w, BEAM, char_topk=CHAR_TOPK)
        top = beams[0]
        print(f"  weight {w:>4}: top beam {top[0]!r} (score {top[1]:+.2f})")

    print("\nStep 4 - candidate pool (union over weights and LM orders, deduplicated):")
    pool = rec["pool"]
    shown = sorted({t for t in pool if t in (truth, visual, chosen)})
    print(f"  {len(pool)} candidates; the ones that matter here: {shown}")

    # All candidates share the row's length (one beam step per lattice
    # position), so a two-line position ruler indented to match "  {text:<12}"
    # lines up with every candidate string printed below it.
    ruler = _position_ruler(len(lattice), indent=2)

    print("\nStep 5 - features per candidate (scored against the full lattice):")
    for line in ruler:
        print(line)
    feats = rec["feats"]
    cols = ["visual_sum", "mean_char_logprob", "length", "length_delta", "lm6_sum"]
    header = f"  {'candidate':<12}" + "".join(f"{c:>20}" for c in cols)
    print(header)
    for text in sorted(pool, key=lambda t: feats[t]["visual_sum"], reverse=True)[:6]:
        f = feats[text]
        print(f"  {text:<12}" + "".join(f"{f[c]:>20.3f}" for c in cols))

    print("\nStep 6 - ridge scores (production final_fit from ranker_scores.json):")
    for line in ruler:
        print(line)
    scores = rec["scores"]
    for text in sorted(pool, key=lambda t: scores[t], reverse=True)[:6]:
        marker = " <- chosen" if text == chosen else ""
        print(f"  {text:<12} score={scores[text]:+.3f}{marker}")

    print(f"\nResult: visual-only read {visual!r}; the ranker selects {chosen!r} (truth {truth!r}).")
    return rec


def fig_lattice_graph(rec: dict[str, Any] | None = None, top_k: int = 4,
                      out: Path | str | None = None, context: int = 2,
                      max_positions: int = 6, **find_kwargs: Any) -> Path:
    """Draw a real row's lattice as a graph, windowed to where the
    visual-only argmax and the ranker-selected (or true) row actually
    disagree, plus a small context margin, so long rows stay legible instead
    of shrinking to an illegible strip; the full row is always visible in the
    companion `fig_lattice_example` heatmap. Top-k candidate letters per
    position are shown (plus the visual/ranker/truth letters if they fall
    outside that top-k), all beam transitions faint, the visual-only path
    dashed, the ranker-chosen path solid, and the ground truth marked when it
    differs from the chosen path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    rec = rec or find_example_row(**find_kwargs)
    lattice, visual, chosen, truth = rec["lattice"], rec["visual"], rec["chosen"], rec["truth"]
    n = len(lattice)

    diverge = [i for i in range(min(n, len(visual), len(chosen))) if visual[i] != chosen[i]]
    diverge += [i for i in range(min(n, len(truth), len(chosen))) if truth and truth[i] != chosen[i]]
    if diverge and n > max_positions:
        lo, hi = max(0, min(diverge) - context), min(n - 1, max(diverge) + context)
        while hi - lo + 1 < min(max_positions, n):
            if lo > 0:
                lo -= 1
            elif hi < n - 1:
                hi += 1
            else:
                break
    else:
        lo, hi = 0, n - 1
    positions = list(range(lo, hi + 1))
    windowed = len(positions) < n

    display: list[dict[str, float]] = []
    for i in positions:
        pos = lattice[i]
        ranked = sorted(pos.items(), key=lambda kv: kv[1], reverse=True)
        keep = dict(ranked[:top_k])
        for text in (visual, chosen, truth):
            if i < len(text) and text[i] in pos and text[i] not in keep:
                keep[text[i]] = pos[text[i]]
        display.append(dict(sorted(keep.items(), key=lambda kv: kv[1], reverse=True)))

    def windowed_text(full_text: str) -> str:
        return "".join(full_text[i] for i in positions if i < len(full_text))

    visual_w, chosen_w = windowed_text(visual), windowed_text(chosen)
    truth_w = windowed_text(truth) if truth else ""

    m = len(positions)
    n_rank = max(len(pos) for pos in display)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * m), 4.6))
    coord: dict[tuple[int, str], tuple[float, float]] = {}
    for li, pos in enumerate(display):
        for r, ch in enumerate(pos):
            coord[(li, ch)] = (float(li), float(-r))
    for li in range(m - 1):
        for ch_a in display[li]:
            for ch_b in display[li + 1]:
                (xa, ya), (xb, yb) = coord[(li, ch_a)], coord[(li + 1, ch_b)]
                ax.plot([xa + 0.18, xb - 0.18], [ya, yb], color="0.86", lw=0.6, zorder=1)

    paths = [(visual_w, "--", "#4878A8", f"visual-only argmax: {visual}")]
    if truth_w and truth_w != chosen_w:
        paths.append((truth_w, ":", "#3F9142", f"ground truth: {truth}"))
    paths.append((chosen_w, "-", "#C8961E", f"ranker-selected row: {chosen}"))
    for text, style, color, label in paths:
        xs = [coord[(li, text[li])][0] for li in range(min(m, len(text))) if (li, text[li]) in coord]
        ys = [coord[(li, text[li])][1] for li in range(min(m, len(text))) if (li, text[li]) in coord]
        ax.plot(xs, ys, style, color=color, lw=2.6, zorder=3, label=label, solid_capstyle="round")

    for (li, ch), (x, y) in coord.items():
        lp = display[li][ch]
        strength = max(0.0, min(1.0, math.exp(lp)))
        on_chosen = li < len(chosen_w) and chosen_w[li] == ch
        box = FancyBboxPatch((x - 0.18, y - 0.22), 0.36, 0.44,
                             boxstyle="round,pad=0.02",
                             facecolor=plt.cm.YlOrBr(0.15 + 0.5 * strength) if on_chosen
                             else plt.cm.Blues(0.10 + 0.55 * strength),
                             edgecolor="#C8961E" if on_chosen else "#40608A",
                             lw=1.6 if on_chosen else 0.8, zorder=4)
        ax.add_patch(box)
        ax.text(x, y + 0.05, ch, ha="center", va="center", fontsize=12,
                fontweight="bold", zorder=5)
        ax.text(x, y - 0.13, f"{lp:+.1f}", ha="center", va="center", fontsize=7,
                color="0.25", zorder=5)
    for li, abs_i in enumerate(positions):
        ax.text(li, 0.62, f"position {abs_i}", ha="center", va="center", fontsize=9, color="0.35")
    ax.set_xlim(-0.55, m - 0.45)
    ax.set_ylim(-n_rank + 0.45, 0.95)
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=len(paths), frameon=False, fontsize=9)
    base, row_idx = rec["row"]["base"], rec["row"]["row_idx"]
    window_note = f", positions {lo}-{hi} of {n} shown" if windowed else ""
    ax.set_title(f"charpost row lattice: {base} row {row_idx}{window_note}\n"
                 f"(per-position character posteriors and decoded paths)", fontsize=10)
    fig.tight_layout()
    path = _save(fig, "charpost_lattice_graph.png", out)
    plt.close(fig)
    return path


def fig_lattice_example(rec: dict[str, Any] | None = None, out: Path | str | None = None,
                        **find_kwargs: Any) -> Path:
    """Real-data decode walkthrough for one row: full-alphabet lattice
    heatmap with the truth / visual-only / ranker-selected paths, plus the
    top-scored candidates and their features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec = rec or find_example_row(**find_kwargs)
    alphabet = rec["alphabet"]
    fit_cols = ["visual_sum", "mean_char_logprob", "length", "length_delta", "lm6_sum", "lm10_sum", "lm12_sum"]
    lattice = rec["lattice"]
    n_pos = len(lattice)
    grid = np.full((len(alphabet), n_pos), -12.0)
    for i, pos in enumerate(lattice):
        for ch, lp in pos.items():
            grid[alphabet.index(ch), i] = lp
    fig = plt.figure(figsize=(max(7.0, 0.62 * n_pos), 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=(2.4, 1.0), hspace=0.28)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=-10, vmax=0)
    ax.set_yticks(range(len(alphabet)), list(alphabet), fontsize=7)
    ax.set_xticks(range(n_pos), [str(i) for i in range(n_pos)], fontsize=7)
    ax.set_xlabel("row position (one tile1 crop per letter box)")
    ax.set_ylabel("proxy character")
    for i in range(n_pos):
        if rec["truth"] and i < len(rec["truth"]) and rec["truth"][i] in alphabet:
            ax.plot(i, alphabet.index(rec["truth"][i]), "s", ms=11, mfc="none",
                    mec="#7CFC00", mew=1.8)
        if rec["visual"][i] in alphabet:
            ax.plot(i, alphabet.index(rec["visual"][i]), "o", ms=7, mfc="none",
                    mec="#63B0E3", mew=1.6)
        if i < len(rec["chosen"]) and rec["chosen"][i] in alphabet:
            ax.plot(i, alphabet.index(rec["chosen"][i]), "x", ms=8, c="#FFD24A", mew=2.0)
    base, ridx = rec["row"]["base"], rec["row"]["row_idx"]
    ax.set_title(f"row lattice {base} row {ridx}  "
                 f"(square=truth, circle=visual argmax, x=ranker choice)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="log P(char | crop)")

    axt = fig.add_subplot(gs[1])
    axt.axis("off")
    cols = fit_cols
    prefix_w = len("truth : ")  # "truth : "/"visual: "/"chosen: " share this width
    lines = [*_position_ruler(n_pos, indent=prefix_w),
             f"truth : {rec['truth']}", f"visual: {rec['visual']}", f"chosen: {rec['chosen']}", ""]
    # Candidate strings start at column 0 and are all exactly n_pos characters
    # (one beam step per lattice position), so an unindented ruler lines up
    # with every row below it regardless of the ridge/feature columns' width.
    lines.extend(_position_ruler(n_pos))
    lines.append(f"{'candidate':<{max(10, n_pos + 2)}}{'ridge':>9}  " +
                 "".join(f"{c:>{max(len(c) + 2, 9)}}" for c in cols))
    for text in sorted(rec["pool"], key=lambda t: rec["scores"][t], reverse=True)[:6]:
        f = rec["feats"][text]
        lines.append(f"{text:<{max(10, n_pos + 2)}}{rec['scores'][text]:>9.3f}  " +
                     "".join(f"{f.get(c, 0.0):>{max(len(c) + 2, 9)}.2f}" for c in cols))
    axt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=7.2,
             va="top", ha="left", transform=axt.transAxes)
    path = _save(fig, "charpost_lattice_example.png", out)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Line-model view figures (dual-light input, patch grid, cross-attention).
# ---------------------------------------------------------------------------

def _sample_dual_chunks(n: int = 3, tile: int = 4, min_chars: int = 3,
                        ckpt: Path | str | None = None) -> list[dict[str, Any]]:
    """Aligned (crop1, crop2, text) chunks from the first readable train bases."""
    import rerank as R

    ckpt = str(ckpt or R.CKPT)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"missing line checkpoint dir: {ckpt}")
    g = R.load(ckpt=ckpt, tile=tile, load_model=False, verbose=False)
    idf = g["index_df"]
    chunks: list[dict[str, Any]] = []
    for _, row in idf[idf["split"] == "train"].iterrows():
        if not (row["rot1_img"] and row.get("rot2_img")):
            continue
        rows1 = g["extract_rows"](row["rot1_img"], clean=True, max_chars=tile)
        rows2 = g["extract_rows"](row["rot2_img"], clean=True, max_chars=tile)
        if len(rows1) != len(rows2):
            continue
        for r1, r2 in zip(rows1, rows2):
            if len(r1) != len(r2):
                continue
            for (c1, t1), (c2, t2) in zip(r1, r2):
                if t1 == t2 and len(t1) >= min_chars:
                    chunks.append({"crop1": c1, "crop2": c2, "text": t1})
                    if len(chunks) >= n:
                        return chunks
    if not chunks:
        raise FileNotFoundError("no aligned readable tile chunks found under data/")
    return chunks


def fig_duallight_steps(n: int = 3, out: Path | str | None = None) -> Path:
    """Dual-light construction: raw rotations, registered channels, min-union,
    and the packed false-color RGB tensor the encoder actually sees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import duallight as DL

    chunks = _sample_dual_chunks(n=n)
    cols = ["Rotation1 (R)", "Rotation2 raw", "registered Rot2 (G)", "min(R,G) union (B)", "packed RGB"]
    fig, axes = plt.subplots(len(chunks), len(cols), figsize=(2.6 * len(cols), 1.5 * len(chunks) + 0.7))
    axes = np.atleast_2d(axes)
    for r, chunk in enumerate(chunks):
        packed = np.asarray(DL.pack(chunk["crop1"], chunk["crop2"], "min").convert("RGB"))
        views = [np.asarray(chunk["crop1"].convert("L")), np.asarray(chunk["crop2"].convert("L")),
                 packed[:, :, 1], packed[:, :, 2], packed]
        for c, (ax, img) in enumerate(zip(axes[r], views)):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(chunk["text"], fontsize=9)
    fig.suptitle("Dual-light packing on real tile4 chunks", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = _save(fig, "duallight_steps.png", out)
    plt.close(fig)
    return path


def fig_patch_grid(n: int = 3, size: int = 384, patch: int = 16,
                   out: Path | str | None = None) -> Path:
    """Packed crops resized to the processor square with the ViT patch grid
    overlaid: what one token sees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import duallight as DL
    from PIL import Image

    chunks = _sample_dual_chunks(n=n)
    side = size // patch
    # Chunks side by side (not stacked) so the figure prints as a landscape
    # strip rather than a tall column that dwarfs whatever it's paired with.
    fig, axes = plt.subplots(1, len(chunks), figsize=(2.3 * len(chunks), 2.6))
    axes = np.atleast_1d(axes)
    for ax, chunk in zip(axes, chunks):
        packed = DL.pack(chunk["crop1"], chunk["crop2"], "min").convert("RGB")
        ax.imshow(np.asarray(packed.resize((size, size), Image.Resampling.BILINEAR)))
        for k in range(0, size + 1, patch):
            ax.axhline(k - 0.5, color="w", lw=0.25, alpha=0.6)
            ax.axvline(k - 0.5, color="w", lw=0.25, alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(chunk["text"], fontsize=9)
    fig.suptitle(f"Packed tile4 crops over the {side}x{side} ViT patch grid", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = _save(fig, "patch_grid.png", out)
    plt.close(fig)
    return path


def fig_cross_attention(ckpt: Path | str | None = None, n: int = 2,
                        out: Path | str | None = None) -> Path:
    """Decoder cross-attention over encoder patches for each generated token
    of the all-train line recognizer (interpretability diagnostic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import rerank as R
    from PIL import Image
    import duallight as DL

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    ckpt = str(ckpt or R.CKPT)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"missing line checkpoint dir: {ckpt}")
    chunks = _sample_dual_chunks(n=n, ckpt=ckpt)
    # load_model=False: rerank.load(load_model=True) would pull the *base*
    # pretrained TrOCR processor from the HF hub (needs sentencepiece) before
    # switching to the checkpoint's saved one. We only need the dataset/device
    # context here, so load the checkpoint's own processor and model directly.
    g = R.load(ckpt=ckpt, tile=4, load_model=False, verbose=False)
    device = g["DEVICE"]
    prep = CL.make_default_prep_image()

    from trocr_model import fix_trocr_meta

    processor = TrOCRProcessor.from_pretrained(ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model = fix_trocr_meta(model, device).eval()
    alphabet = R._alphabet_from_gt(g)
    logits_processor = R.make_charset_logits_processor(processor, alphabet, device, mode="charset")

    panels: list[tuple[Any, list[tuple[str, np.ndarray]]]] = []
    for chunk in chunks:
        packed = DL.pack(chunk["crop1"], chunk["crop2"], "min").convert("RGB")
        pix = processor(images=prep(packed), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen = model.generate(pix, max_new_tokens=8, num_beams=1,
                                 output_attentions=True, return_dict_in_generate=True,
                                 logits_processor=logits_processor)
        toks = [processor.tokenizer.decode([t], skip_special_tokens=True) for t in gen.sequences[0][1:]]
        maps: list[tuple[str, np.ndarray]] = []
        for step, tok in enumerate(toks):
            if not tok.strip() or step >= len(gen.cross_attentions):
                continue
            att = gen.cross_attentions[step][-1][0].mean(dim=0)[-1].float().cpu().numpy()
            side = int(math.isqrt(att.shape[-1]))
            maps.append((tok, att[-side * side:].reshape(side, side)))
        panels.append((packed, maps))
    n_tok = max((len(m) for _, m in panels), default=1)
    n_rows = len(panels)
    n_cols = n_tok + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.1 * n_rows))
    # np.atleast_2d always prepends a new axis, which is wrong whenever it's
    # n_cols (not n_rows) that collapsed to 1 -- e.g. every sampled chunk
    # generates zero non-blank tokens, so n_tok=0. reshape is unambiguous for
    # every combination of n_rows/n_cols instead.
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for r, (packed, maps) in enumerate(panels):
        img = np.asarray(packed.resize((384, 384), Image.Resampling.BILINEAR))
        axes[r, 0].imshow(img)
        axes[r, 0].set_title("input", fontsize=8)
        for c in range(n_tok):
            ax = axes[r, c + 1]
            if c < len(maps):
                tok, amap = maps[c]
                ax.imshow(img)
                ax.imshow(np.kron(amap, np.ones((384 // amap.shape[0], 384 // amap.shape[1]))),
                          cmap="inferno", alpha=0.55)
                ax.set_title(repr(tok), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("Decoder cross-attention per generated proxy token (last layer, head mean)",
                 fontsize=10)
    # h_pad reserves extra vertical space between rows; without it tight_layout
    # packs rows tightly enough that row r+1's per-axes titles sit on top of
    # row r's images.
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2)
    path = _save(fig, "cross_attention.png", out)
    plt.close(fig)
    return path


def build_all(production_cer: float | None = "auto") -> None:  # type: ignore[assignment]
    """Build every report figure that the local artifact tree supports."""
    maybe(fig_cer_ladder, production_cer=production_cer)
    maybe(fig_ranker_weights)
    maybe(fig_confusion)
    example = maybe(find_example_row)
    if example is not None:
        mechanism_walkthrough(rec=example)
        fig_lattice_graph(rec=example)
        fig_lattice_example(rec=example)
    maybe(fig_duallight_steps)
    maybe(fig_patch_grid)
    maybe(fig_cross_attention)


if __name__ == "__main__":
    build_all()
