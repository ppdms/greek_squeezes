from __future__ import annotations

import random
from typing import Any

RANK_FOLD_COUNT = 5
RANK_FOLD_SEED = 42


def rank_folds_from_index(index_df: Any, n_folds: int = RANK_FOLD_COUNT, seed: int = RANK_FOLD_SEED) -> list[list[str]]:
    bases = sorted(index_df[index_df['split'] == 'train']['base'].tolist())
    rng = random.Random(seed)
    rng.shuffle(bases)
    return [sorted(bases[i::n_folds]) for i in range(n_folds)]
