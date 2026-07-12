#!/usr/bin/env python3
"""Official TROGS-26 scoring (contest_evaluation_v2) for finished prod runs.

The pipeline's own CER (per-row, no newline credit, all letters mandatory --
see report Section 2) is used for every fitted or reported internal number.
This module grades a prod run's submission-style transcripts the way the
organizers grade entries, by calling the verbatim v2 scorer, and reports the
two facts that decide how those numbers relate to the pipeline's:

* whether the ground-truth annotation files carry the v2 per-box confidence
  section at all (without it, `cer` == `all_cer` and the masked variant is
  vacuous), and
* whether v1 (used by frozen staging) and v2 row grouping agree on these
  annotation files -- v2 activated two neighbor-mismatch resets in
  getRowBoxes that v1 left commented out, so grouping can differ on files
  with inconsistent left/right neighbor links.

Nothing here feeds back into decoding, ranking, or artifact builds.

CLI (after syncing or staging the run locally):
    python squeeze_runtime/official_scoring.py \
        --pred data/prod_runs/<slug>/transcripts \
        --gt   data/prod_inputs/<slug>/Annotations/Annotations \
        [--json scores.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contest_evaluation as CE1  # noqa: E402  (frozen staging parser)
import contest_evaluation_v2 as CE2  # noqa: E402  (official scorer)


def official_scores(pred_dir: Path | str, gt_dir: Path | str) -> dict:
    """Run the official v2 scorer over a transcripts dir and a letters dir.

    Returns the three official CERs. Note the official tool iterates over
    every letters file, so each squeeze is scored once per rotation file
    (identical transcript and confidence union both times); the micro-average
    ratios are unaffected.
    """
    (cer, all_cer, opt_cer,
     n_err, n_all_err, n_opt_err,
     n_char, n_all_char, n_opt_char, gt_files) = CE2.run_evaluations(str(pred_dir), str(gt_dir))
    return {
        'scorer': 'contest_evaluation_v2.run_evaluations',
        'cer_confident_only': cer,
        'all_cer': all_cer,
        'opt_cer': opt_cer,
        'errors': {'confident_only': int(sum(n_err)), 'all': int(sum(n_all_err)),
                   'opt': int(sum(n_opt_err))},
        'chars': {'confident_only': int(sum(n_char)), 'all': int(sum(n_all_char)),
                  'opt': int(sum(n_opt_char))},
        'gt_files': len(gt_files),
    }


def confidence_coverage(gt_dir: Path | str) -> dict:
    """Report whether/how much the letters files use the v2 confidence section.

    A file "has" the section when it contains a fifth `#` header; per-box
    flags are then read through the official parser. Without any sections the
    official confident-only CER degenerates to the all-characters CER.
    """
    gt_dir = Path(gt_dir)
    files = sorted(p for p in gt_dir.iterdir() if p.is_file())
    with_section = 0
    boxes = 0
    obscured = 0
    for path in files:
        headers = sum(1 for line in path.read_text(encoding='utf-8', errors='replace').splitlines()
                      if line.startswith('#'))
        conf = CE2.readBoxFile(str(path), all_conf=True, get_conf=True)
        boxes += len(conf)
        if headers >= 5:
            with_section += 1
            obscured += sum(1 for c in conf if not c)
    return {
        'files': len(files),
        'files_with_confidence_section': with_section,
        'boxes': boxes,
        'boxes_marked_obscured': obscured,
    }


def grouping_check(gt_dir: Path | str) -> dict:
    """Compare v1 vs v2 row grouping on every letters file.

    The frozen pipeline stages rows with v1's getRowBoxes; the official v2
    scorer builds its reference rows with v2's. If they disagree on a file,
    the pipeline's transcript row order can differ from the official
    reference row order on that squeeze.
    """
    gt_dir = Path(gt_dir)
    differing: list[str] = []
    files = sorted(p for p in gt_dir.iterdir() if p.is_file())
    for path in files:
        def idlist(mod, **kw):
            gangle, boxes, transcript, lines = mod.readBoxFile(str(path), **kw)
            boxes2 = mod.orderBoxes(boxes, gangle, lines)
            _rowlist, ids = mod.getRowBoxes(boxes2, gangle=gangle)
            return [[int(i) for i in row] for row in ids]
        if idlist(CE1) != idlist(CE2, all_conf=True):
            differing.append(path.name)
    return {
        'files': len(files),
        'files_where_v1_v2_grouping_differs': len(differing),
        'differing_files': differing,
    }


def score_run(pred_dir: Path | str, gt_dir: Path | str) -> dict:
    return {
        **official_scores(pred_dir, gt_dir),
        'confidence': confidence_coverage(gt_dir),
        'grouping_v1_vs_v2': grouping_check(gt_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pred', required=True, help='transcripts dir ({base}_transcript.txt files)')
    ap.add_argument('--gt', required=True, help='letters dir (*_letters.txt, both rotations)')
    ap.add_argument('--json', default='', help='optional path to write the full result as JSON')
    args = ap.parse_args()
    result = score_run(args.pred, args.gt)
    conf = result['confidence']
    grp = result['grouping_v1_vs_v2']
    print(f"official v2 scores over {result['gt_files']} letters files:")
    print(f"  cer (confident-only): {result['cer_confident_only']:.6f}")
    print(f"  all_cer (all chars) : {result['all_cer']:.6f}")
    print(f"  opt_cer (masked)    : {result['opt_cer']:.6f}")
    print(f"confidence sections: {conf['files_with_confidence_section']}/{conf['files']} files; "
          f"{conf['boxes_marked_obscured']}/{conf['boxes']} boxes marked obscured")
    print(f"v1 vs v2 row grouping differs on {grp['files_where_v1_v2_grouping_differs']}"
          f"/{grp['files']} files"
          + (f": {', '.join(grp['differing_files'][:5])}" if grp['differing_files'] else ''))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
        print('wrote', args.json)


if __name__ == '__main__':
    main()
