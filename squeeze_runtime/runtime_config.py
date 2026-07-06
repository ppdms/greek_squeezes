#!/usr/bin/env python3
"""One place to point every runtime module at an artifact tree.

The runtime modules (rerank, char_lattice, charpost_ranker, report_figs) keep
their paths in module globals. Callers used to overwrite those globals one by
one from notebook cells and modal_app, which made it hard to tell which paths
were actually in effect. ``configure(work_dir)`` is now the single sanctioned
way to do it: it sets every path global consistently and returns the derived
layout so the caller can build a PipelineConfig from the same values.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def configure(work_dir: Path | str, figs_dir: Path | str | None = None) -> dict[str, Path]:
    """Point rerank/char_lattice/charpost_ranker/report_figs at `work_dir`.

    `work_dir` is the artifact tree root (the directory that contains
    line_recognizer/, character_posterior/, splits/, prod_runs/). `figs_dir`
    optionally redirects report_figs output (default: report/figs under the
    current working directory).

    Returns the derived layout: work, line_dir, charpost_dir, splits_dir,
    folds_json, ranker_dir, image_cache_dir.
    """
    import rerank as R
    import char_lattice as CL
    import charpost_ranker as CR
    import report_figs as RF

    work = Path(work_dir)
    line_dir = work / 'line_recognizer'
    charpost_dir = work / 'character_posterior'
    splits_dir = work / 'splits'
    folds_json = splits_dir / 'oof_folds.json'
    ranker_dir = charpost_dir / 'oof_ranker'
    image_cache_dir = charpost_dir / 'image_cache'

    R.WORK = str(work.resolve())
    R.CKPT = str(line_dir.resolve() / 'all_train')
    CL.WORK = work
    CL.CHARPOST_DIR = charpost_dir
    CL.DEFAULT_FOLDS = folds_json
    CR.WORK = work
    CR.CHARPOST_DIR = charpost_dir
    CR.OOF_DIR = ranker_dir
    RF.WORK = work
    if figs_dir is not None:
        RF.FIGS_DIR = Path(figs_dir)

    return {
        'work': work,
        'line_dir': line_dir,
        'charpost_dir': charpost_dir,
        'splits_dir': splits_dir,
        'folds_json': folds_json,
        'ranker_dir': ranker_dir,
        'image_cache_dir': image_cache_dir,
    }
