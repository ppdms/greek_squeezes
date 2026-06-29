from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURE_VERSION = 'lattice_length_delta_v2'


@dataclass(frozen=True)
class PipelineConfig:
    repo: Path
    charpost_dir: Path
    folds: tuple[int, ...]
    tile: int
    folds_json: Path
    line_all_train: Path
    line_oof_fold: dict[int, Path]
    charpost_all_train_tag: str
    charpost_oof_prefix: str
    charpost_orders: tuple[int, ...]
    image_cache_dir: Path
    ranker_dir: Path
    candidates_json: Path
    ranker_json: Path
    ranker_methods: str
    charpost_ranker_features: str
    rank_fold_seed: int


class ArtifactChecks:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def file_ready(path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0

    @staticmethod
    def dir_ready(path: Path) -> bool:
        return path.exists() and path.is_dir()

    @staticmethod
    def newer_than(path: Path, deps: list[Path]) -> bool:
        try:
            if not path.exists():
                return False
            ready_deps = [dep for dep in deps if dep.exists()]
            if len(ready_deps) != len(deps):
                return False
            return path.stat().st_mtime >= max(dep.stat().st_mtime for dep in ready_deps)
        except OSError:
            return False

    @staticmethod
    def read_json_or_none(path: Path) -> Any | None:
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None

    @staticmethod
    def same_pathish(left: str | Path, right: str | Path) -> bool:
        def data_suffix(value: str | Path) -> tuple[str, ...] | None:
            parts = Path(str(value)).expanduser().parts
            if 'data' not in parts:
                return None
            idx = len(parts) - 1 - parts[::-1].index('data')
            suffix = parts[idx:]
            return suffix if len(suffix) > 1 else None

        try:
            if Path(left).expanduser().resolve() == Path(right).expanduser().resolve():
                return True
        except Exception:
            pass
        left_suffix = data_suffix(left)
        right_suffix = data_suffix(right)
        if left_suffix is not None and right_suffix is not None:
            return left_suffix == right_suffix
        return str(left) == str(right)

    def json_file_ready(self, path: Path) -> bool:
        return isinstance(self.read_json_or_none(path), (dict, list))

    def require_artifact_layout(self) -> None:
        nested = self.cfg.repo / self.cfg.repo.name
        if nested.exists():
            raise RuntimeError(f'unexpected nested artifact directory: {nested}')

    def checkpoint_ready(self, path: Path) -> bool:
        required = ('config.json', 'generation_config.json', 'run_config.json')
        if not self.dir_ready(path) or any(not self.file_ready(path / name) for name in required):
            return False
        if not any(self.file_ready(path / name) for name in ('model.safetensors', 'pytorch_model.bin')):
            return False
        if not any(self.file_ready(path / name) for name in ('processor_config.json', 'preprocessor_config.json', 'image_processor_config.json')):
            return False
        if not any(self.file_ready(path / name) for name in ('tokenizer.json', 'tokenizer_config.json')):
            return False
        return isinstance(self.read_json_or_none(path / 'run_config.json'), dict)

    def folds_manifest_ready(self, path: Path) -> bool:
        data = self.read_json_or_none(path)
        if not isinstance(data, dict):
            return False
        folds = data.get('folds') or []
        ids = sorted(int(item.get('id', -1)) for item in folds if isinstance(item, dict))
        return data.get('kind') == 'charpost_oof_folds' and data.get('seed') == self.cfg.rank_fold_seed and ids == list(self.cfg.folds)

    def image_cache_ready(self) -> bool:
        manifest = self.read_json_or_none(self.cfg.image_cache_dir / 'dual_pack__min' / 'manifest.json')
        if not isinstance(manifest, dict):
            return False
        expected = {
            'kind': 'charpost_image_cache',
            'init_ckpt': str(self.cfg.line_all_train),
            'tile': 1,
            'splits': 'train,val,test',
            'input_mode': 'dual_pack',
            'dual_mode': 'min',
            'cache_key': 'dual_pack__min',
        }
        for key, value in expected.items():
            if key == 'init_ckpt':
                if not self.same_pathish(str(manifest.get(key, '')), value):
                    return False
            elif manifest.get(key) != value:
                return False
        example_index = self.cfg.image_cache_dir / 'dual_pack__min' / str(manifest.get('example_index') or 'examples.jsonl')
        return (
            int(manifest.get('total_files') or 0) > 0
            and int(manifest.get('examples') or 0) > 0
            and self.file_ready(example_index)
        )

    def charpost_classifier_ready(
        self,
        model_dir: Path,
        *,
        tag: str | None = None,
        init_ckpt: Path | None = None,
        train_exclude_fold_id: int | None = None,
        val_fold_id: int | None = None,
    ) -> bool:
        if not (
            self.file_ready(model_dir / 'head.pt')
            and self.dir_ready(model_dir / 'encoder')
            and self.file_ready(model_dir / 'metadata.json')
            and self.file_ready(model_dir / 'train_state.pt')
        ):
            return False
        meta = self.read_json_or_none(model_dir / 'metadata.json')
        if not isinstance(meta, dict):
            return False
        checks: dict[str, Any] = {
            'input_mode': 'dual_pack',
            'dual_mode': 'min',
            'tile': 1,
            'train_splits': 'train',
            'batch': 96,
            'grad_accum': 1,
            'max_steps': 3000,
            'workers': 6,
            'prefetch_factor': 4,
            'grad_checkpoint': True,
            'fp16': True,
        }
        if tag is not None:
            checks['tag'] = tag
        for key, value in checks.items():
            if meta.get(key) != value:
                return False
        if init_ckpt is not None and not self.same_pathish(str(meta.get('init_ckpt', '')), init_ckpt):
            return False
        if meta.get('train_exclude_fold_id') != train_exclude_fold_id:
            return False
        if meta.get('val_fold_id') != val_fold_id:
            return False
        if meta.get('image_cache_dir') and not self.same_pathish(str(meta.get('image_cache_dir')), self.cfg.image_cache_dir):
            return False
        if not bool(meta.get('train_complete')):
            return False
        if int(meta.get('global_step') or 0) < int(meta.get('max_steps') or 3000):
            return False
        return True

    def charpost_logits_ready(self, path: Path, *, tag: str, split: str, model_dir: Path, include_fold_id: int | None = None) -> bool:
        if not self.file_ready(path):
            return False
        init_ckpt = self.cfg.line_oof_fold[include_fold_id] if include_fold_id is not None else self.cfg.line_all_train
        if not self.charpost_classifier_ready(
            model_dir,
            tag=tag,
            init_ckpt=init_ckpt,
            train_exclude_fold_id=include_fold_id,
            val_fold_id=include_fold_id,
        ):
            return False
        try:
            logits_mtime = path.stat().st_mtime
            classifier_mtime = max(
                (model_dir / 'head.pt').stat().st_mtime,
                (model_dir / 'metadata.json').stat().st_mtime,
                (model_dir / 'train_state.pt').stat().st_mtime,
            )
            if logits_mtime < classifier_mtime:
                return False
        except OSError:
            return False
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as data:
                required = {'logits', 'labels', 'bases', 'row_idx', 'chunk_idx', 'row_text', 'alphabet'}
                if not required <= set(data.files):
                    return False
                logits = data['logits']
                n = int(logits.shape[0]) if getattr(logits, 'ndim', 0) == 2 else -1
                alphabet = data['alphabet']
                if n <= 0 or int(logits.shape[1]) != len(alphabet):
                    return False
                for key in ('labels', 'bases', 'row_idx', 'chunk_idx', 'row_text'):
                    if len(data[key]) != n:
                        return False
                if np.any(data['labels'] < 0) or np.any(data['labels'] >= len(alphabet)):
                    return False
        except Exception:
            return False
        return True

    def candidate_deps(self) -> list[Path]:
        deps: list[Path] = []
        for fold in self.cfg.folds:
            tag = f'{self.cfg.charpost_oof_prefix}_f{fold}'
            deps.append(self.cfg.charpost_dir / f'{tag}__fold{fold}_logits.npz')
            for order in self.cfg.charpost_orders:
                deps.append(self.cfg.ranker_dir / f'train_lm_order{order}_exclude_f{fold}' / 'lm.json')
        return deps

    def final_decode_deps(self, split: str) -> list[Path]:
        deps = [
            self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__{split}_logits.npz',
            self.cfg.ranker_json,
        ]
        for order in self.cfg.charpost_orders:
            deps.append(self.cfg.charpost_dir / f'train_lm_order{order}_train' / 'lm.json')
        return deps

    def _candidate_manifest_matches(self, manifest: dict[str, Any], *, row_shard_count: int, row_shard_id: int | None = None) -> bool:
        expected = {
            'tag_prefix': self.cfg.charpost_oof_prefix,
            'folds': list(self.cfg.folds),
            'orders': list(self.cfg.charpost_orders),
            'proposal_weights': [0.0, 0.25, 0.5, 0.75, 1.0],
            'beam': 256,
            'char_topk': 8,
            'feature_version': FEATURE_VERSION,
            'row_shard_count': row_shard_count,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                return False
        if row_shard_id is not None and manifest.get('row_shard_id') != row_shard_id:
            return False
        return int(manifest.get('rows') or 0) > 0 and int(manifest.get('candidates') or 0) > 0

    def candidate_manifest_ok(self, path: Path, *, shards: int | None = None) -> bool:
        data = self.read_json_or_none(path)
        if not isinstance(data, dict):
            return False
        records = data.get('records') or []
        manifest = data.get('manifest') or {}
        if not isinstance(manifest, dict) or not isinstance(records, list) or not records:
            return False
        if manifest.get('kind') == 'merged_candidate_shards':
            inputs = manifest.get('inputs') or []
            if not inputs:
                return False
            row_shard_ids = sorted(item.get('row_shard_id') for item in inputs)
            expected_shards = len(inputs) if shards is None else shards
            if len(inputs) != expected_shards or row_shard_ids != list(range(expected_shards)):
                return False
            shard_paths = [Path(str(item.get('path') or '')) for item in inputs]
            if not all(self.file_ready(shard_path) for shard_path in shard_paths):
                return False
            if not all(
                self.candidate_shard_ready(
                    shard_path,
                    int(item.get('row_shard_id')),
                    expected_shards,
                )
                for item, shard_path in zip(inputs, shard_paths)
            ):
                return False
            if not self.newer_than(path, shard_paths):
                return False
            return int(manifest.get('rows') or 0) > 0 and int(manifest.get('candidates') or 0) == len(records)
        expected_shards = 1 if shards is None else shards
        return (
            self._candidate_manifest_matches(manifest, row_shard_count=expected_shards, row_shard_id=0)
            and int(manifest.get('candidates') or 0) == len(records)
            and self.newer_than(path, self.candidate_deps())
        )

    def candidate_shard_ready(self, path: Path, sid: int, shards: int) -> bool:
        data = self.read_json_or_none(path)
        if not isinstance(data, dict):
            return False
        manifest = data.get('manifest') or {}
        records = data.get('records') or []
        return (
            isinstance(manifest, dict)
            and isinstance(records, list)
            and bool(records)
            and self._candidate_manifest_matches(manifest, row_shard_count=shards, row_shard_id=sid)
            and self.newer_than(path, self.candidate_deps())
        )

    def ranker_ready(self, path: Path) -> bool:
        data = self.read_json_or_none(path)
        if not isinstance(data, dict):
            return False
        final_fit = data.get('final_fit') or {}
        best = data.get('best') or {}
        return (
            data.get('features') == self.cfg.charpost_ranker_features.split(',')
            and data.get('methods') == [part.strip() for part in self.cfg.ranker_methods.split(',') if part.strip()]
            and self.candidate_manifest_ok(self.cfg.candidates_json)
            and isinstance(best, dict)
            and best.get('cer') is not None
            and isinstance(final_fit, dict)
            and not final_fit.get('failed')
            and set(final_fit.get('features') or final_fit.get('cols') or []) == set(self.cfg.charpost_ranker_features.split(','))
            and 'coef' in final_fit
            and self.newer_than(path, [self.cfg.candidates_json])
        )

    def final_charpost_lm_ready(self, path: Path, order: int) -> bool:
        data = self.read_json_or_none(path)
        return isinstance(data, dict) and int(data.get('order') or order) == order

    def final_charpost_logits_ready(self, split: str, model_dir: Path) -> bool:
        logits = self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__{split}_logits.npz'
        return self.charpost_logits_ready(logits, tag=self.cfg.charpost_all_train_tag, split=split, model_dir=model_dir, include_fold_id=None)

    def decode_ready(self, path: Path, split: str) -> bool:
        data = self.read_json_or_none(path)
        if not isinstance(data, dict):
            return False
        predictions = data.get('predictions') or []
        transcripts = data.get('transcripts') or []
        scored = data.get('scored')
        return (
            data.get('tag') == self.cfg.charpost_all_train_tag
            and data.get('split') == split
            and data.get('orders') == list(self.cfg.charpost_orders)
            and data.get('proposal_weights') == [0.0, 0.25, 0.5, 0.75, 1.0]
            and data.get('beam') == 256
            and data.get('char_topk') == 8
            and data.get('features') == self.cfg.charpost_ranker_features.split(',')
            and self.same_pathish(str(data.get('ranker', '')), self.cfg.ranker_json)
            and isinstance(predictions, list)
            and isinstance(transcripts, list)
            and len(predictions) > 0
            and len(transcripts) > 0
            and (scored is None or (isinstance(scored, dict) and int(scored.get('rows') or 0) == len(predictions)))
            and self.newer_than(path, self.final_decode_deps(split))
        )

    def decode_shard_ready(self, path: Path, split: str, shard: int, shards: int) -> bool:
        data = self.read_json_or_none(path)
        if not isinstance(data, dict):
            return False
        predictions = data.get('predictions') or []
        return (
            data.get('tag') == self.cfg.charpost_all_train_tag
            and data.get('split') == split
            and int(data.get('row_shard_count') or -1) == shards
            and int(data.get('row_shard_id') or -1) == shard
            and data.get('orders') == list(self.cfg.charpost_orders)
            and data.get('proposal_weights') == [0.0, 0.25, 0.5, 0.75, 1.0]
            and data.get('beam') == 256
            and data.get('char_topk') == 8
            and data.get('features') == self.cfg.charpost_ranker_features.split(',')
            and self.same_pathish(str(data.get('ranker', '')), self.cfg.ranker_json)
            and isinstance(predictions, list)
            and len(predictions) > 0
            and self.newer_than(path, self.final_decode_deps(split))
        )

    def status(self) -> dict[str, Any]:
        self.require_artifact_layout()
        oof_folds: dict[str, dict[str, bool]] = {}
        for fold in self.cfg.folds:
            tag = f'{self.cfg.charpost_oof_prefix}_f{fold}'
            model_dir = self.cfg.charpost_dir / tag
            logits = self.cfg.charpost_dir / f'{tag}__fold{fold}_logits.npz'
            oof_folds[str(fold)] = {
                'classifier': self.charpost_classifier_ready(
                    model_dir,
                    tag=tag,
                    init_ckpt=self.cfg.line_oof_fold[fold],
                    train_exclude_fold_id=fold,
                    val_fold_id=fold,
                ),
                'logits': self.charpost_logits_ready(
                    logits,
                    tag=tag,
                    split=f'fold{fold}',
                    model_dir=model_dir,
                    include_fold_id=fold,
                ),
                'lms': all(
                    self.json_file_ready(self.cfg.ranker_dir / f'train_lm_order{order}_exclude_f{fold}' / 'lm.json')
                    for order in self.cfg.charpost_orders
                ),
            }
        return {
            'line_checkpoints': {
                'all_train': self.checkpoint_ready(self.cfg.line_all_train),
                'oof_folds': {str(fold): self.checkpoint_ready(path) for fold, path in self.cfg.line_oof_fold.items()},
                'folds_json': self.folds_manifest_ready(self.cfg.folds_json),
            },
            'image_cache': self.image_cache_ready(),
            'oof_folds': oof_folds,
            'candidates': self.candidate_manifest_ok(self.cfg.candidates_json),
            'ranker': self.ranker_ready(self.cfg.ranker_json),
            'final_classifier': self.charpost_classifier_ready(
                self.cfg.charpost_dir / self.cfg.charpost_all_train_tag,
                tag=self.cfg.charpost_all_train_tag,
                init_ckpt=self.cfg.line_all_train,
            ),
            'val_decode': self.decode_ready(
                self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__val_decode.json',
                'val',
            ),
            'test_decode': self.decode_ready(
                self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__test_decode.json',
                'test',
            ),
        }
