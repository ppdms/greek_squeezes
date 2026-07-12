from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from artifact_checks import ArtifactChecks, PipelineConfig


class CharpostPipeline:
    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        artifacts: ArtifactChecks,
        rerank: Any,
        line_folds: Any,
        char_lattice: Any,
        charpost_ranker: Any,
        run_training_if_missing: bool,
    ) -> None:
        self.cfg = cfg
        self.artifacts = artifacts
        self.R = rerank
        self.LF = line_folds
        self.CL = char_lattice
        self.CR = charpost_ranker
        self.run_training_if_missing = run_training_if_missing

    @property
    def order_arg(self) -> str:
        return ','.join(str(order) for order in self.cfg.charpost_orders)

    def require_or_train(self, label: str) -> None:
        if not self.run_training_if_missing:
            raise FileNotFoundError(f'{label} is missing and RUN_TRAINING_IF_MISSING=False')

    def create_oof_folds_if_missing(self) -> None:
        if self.artifacts.folds_manifest_ready(self.cfg.folds_json):
            print('fold manifest ready ->', self.cfg.folds_json)
            return
        g = self.R.load(tile=self.cfg.tile, load_model=False, verbose=False)
        folds = self.LF.rank_folds_from_index(g['index_df'], n_folds=len(self.cfg.folds), seed=self.cfg.rank_fold_seed)
        manifest = {
            'kind': 'charpost_oof_folds',
            'seed': self.cfg.rank_fold_seed,
            'folds': [{'id': i, 'bases': list(bases)} for i, bases in enumerate(folds)],
        }
        self.cfg.folds_json.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.folds_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
        print('wrote fold manifest ->', self.cfg.folds_json)

    def require_line_model(self, kind: str, fold: int | None = None) -> Path:
        if kind == 'all_train':
            target = self.cfg.line_all_train
        elif kind == 'oof_fold' and fold is not None:
            target = self.cfg.line_oof_fold[fold]
        else:
            raise ValueError((kind, fold))
        if self.artifacts.checkpoint_ready(target):
            print(f'line model ready -> {target}')
            return target
        raise FileNotFoundError(f'line model missing or incomplete at {target}')

    def ensure_line_models(self) -> None:
        self.artifacts.require_artifact_layout()
        self.create_oof_folds_if_missing()
        self.require_line_model('all_train')
        for fold in self.cfg.folds:
            self.require_line_model('oof_fold', fold)

    def run_char_lattice(self, argv: list[str]) -> None:
        parser = self.CL.build_parser()
        args = parser.parse_args(argv)
        args.func(args)

    def run_charpost_ranker(self, argv: list[str]) -> None:
        parser = self.CR.build_parser()
        args = parser.parse_args(argv)
        args.func(args)

    def charpost_fold_tag(self, fold: int) -> str:
        return f'{self.cfg.charpost_oof_prefix}_f{fold}'

    def ensure_image_cache(self, workers: int = 1) -> None:
        manifest = self.cfg.image_cache_dir / 'dual_pack__min' / 'manifest.json'
        if self.artifacts.image_cache_ready():
            print('charpost image cache ready ->', manifest.parent)
            return
        self.require_or_train('charpost image cache')
        self.run_char_lattice([
            'cache-images',
            '--init-ckpt', str(self.cfg.line_all_train),
            '--input-mode', 'dual_pack',
            '--dual-mode', 'min',
            '--splits', 'train,val,test',
            '--image-cache-dir', str(self.cfg.image_cache_dir),
            '--workers', str(workers),
        ])
        if not self.artifacts.image_cache_ready():
            raise RuntimeError(f'charpost image cache failed validation: {manifest}')

    def ensure_oof_classifier(self, fold: int) -> None:
        tag = self.charpost_fold_tag(fold)
        model_dir = self.cfg.charpost_dir / tag
        init_ckpt = self.cfg.line_oof_fold[fold]
        if not self.artifacts.charpost_classifier_ready(model_dir, tag=tag, init_ckpt=init_ckpt, train_exclude_fold_id=fold, val_fold_id=fold):
            self.require_or_train(f'charpost OOF classifier fold {fold}')
            self.run_char_lattice([
                'train-classifier',
                '--tag', tag,
                '--init-ckpt', str(init_ckpt),
                '--out-dir', str(model_dir),
                '--input-mode', 'dual_pack',
                '--dual-mode', 'min',
                '--batch', '96',
                '--grad-accum', '1',
                '--max-steps', '3000',
                '--eval-steps', '0',
                '--log-steps', '5',
                '--state-steps', '250',
                '--lr', '1e-5',
                '--weight-decay', '1e-4',
                '--class-weight', 'sqrt',
                '--workers', '6',
                '--prefetch-factor', '4',
                '--image-cache-dir', str(self.cfg.image_cache_dir),
                '--require-image-cache',
                '--no-save-during-training',
                '--folds', str(self.cfg.folds_json),
                '--train-exclude-fold-id', str(fold),
                '--val-fold-id', str(fold),
            ])
            if not self.artifacts.charpost_classifier_ready(model_dir, tag=tag, init_ckpt=init_ckpt, train_exclude_fold_id=fold, val_fold_id=fold):
                raise RuntimeError(f'charpost OOF classifier failed validation: {model_dir}')
        else:
            print('charpost OOF classifier ready ->', model_dir)

    def ensure_oof_logits(self, fold: int) -> None:
        tag = self.charpost_fold_tag(fold)
        model_dir = self.cfg.charpost_dir / tag
        logits = self.CL.logits_path(tag, f'fold{fold}')
        if not self.artifacts.charpost_classifier_ready(
            model_dir,
            tag=tag,
            init_ckpt=self.cfg.line_oof_fold[fold],
            train_exclude_fold_id=fold,
            val_fold_id=fold,
        ):
            raise FileNotFoundError(f'charpost OOF classifier missing before logits: {model_dir}')
        if not self.artifacts.charpost_logits_ready(logits, tag=tag, split=f'fold{fold}', model_dir=model_dir, include_fold_id=fold):
            self.require_or_train(f'charpost OOF logits fold {fold}')
            self.run_char_lattice([
                'cache-logits',
                '--tag', tag,
                '--model-dir', str(model_dir),
                '--splits', 'train',
                '--output-split', f'fold{fold}',
                '--include-fold-id', str(fold),
                '--folds', str(self.cfg.folds_json),
                '--batch', '64',
                '--tta', 'shift5',
                '--tta-pixels', '4',
                '--tta-scale', '0.04',
                '--image-cache-dir', str(self.cfg.image_cache_dir),
                '--require-image-cache',
            ])
            if not self.artifacts.charpost_logits_ready(logits, tag=tag, split=f'fold{fold}', model_dir=model_dir, include_fold_id=fold):
                raise RuntimeError(f'charpost OOF logits failed validation: {logits}')
        else:
            print('charpost OOF logits ready ->', logits)

    def ensure_oof_lms(self, fold: int) -> None:
        for order in self.cfg.charpost_orders:
            lm_dir = self.cfg.ranker_dir / f'train_lm_order{order}_exclude_f{fold}'
            if self.artifacts.json_file_ready(lm_dir / 'lm.json'):
                print('charpost OOF LM ready ->', lm_dir / 'lm.json')
                continue
            self.require_or_train(f'charpost OOF LM order {order} fold {fold}')
            self.run_char_lattice([
                'train-lm',
                '--order', str(order),
                '--splits', 'train',
                '--folds', str(self.cfg.folds_json),
                '--exclude-fold-id', str(fold),
                '--out-dir', str(lm_dir),
            ])
            if not self.artifacts.json_file_ready(lm_dir / 'lm.json'):
                raise RuntimeError(f'charpost OOF LM failed validation: {lm_dir / "lm.json"}')

    def ensure_oof_fold(self, fold: int) -> None:
        self.ensure_oof_classifier(fold)
        self.ensure_oof_logits(fold)
        self.ensure_oof_lms(fold)

    def ensure_oof_artifacts(self, image_workers: int = 1) -> None:
        self.ensure_image_cache(workers=image_workers)
        for fold in self.cfg.folds:
            self.ensure_oof_fold(fold)

    def candidate_shard_path(self, sid: int, shards: int) -> Path:
        return self.cfg.ranker_dir / f'candidates_shard{sid:02d}-of-{shards:02d}.json'

    def ensure_candidate_shard(self, sid: int, shards: int) -> Path:
        if not 0 <= sid < shards:
            raise ValueError(f'candidate shard id must be in [0, {shards}); got {sid}')
        shard = self.candidate_shard_path(sid, shards)
        if self.artifacts.candidate_shard_ready(shard, sid, shards):
            print('candidate shard ready ->', shard)
            return shard
        self.require_or_train(f'charpost candidate shard {sid}/{shards}')
        self.run_charpost_ranker([
            'build-candidates',
            '--tag-prefix', self.cfg.charpost_oof_prefix,
            '--orders', self.order_arg,
            '--row-shard-count', str(shards),
            '--row-shard-id', str(sid),
            '--out', str(shard),
        ])
        if not self.artifacts.candidate_shard_ready(shard, sid, shards):
            raise RuntimeError(f'candidate shard failed validation: {shard}')
        return shard

    def merge_candidate_shards(self, shards: int) -> None:
        if self.artifacts.candidate_manifest_ok(self.cfg.candidates_json, shards=shards):
            print('charpost candidates ready ->', self.cfg.candidates_json)
            return
        shard_paths = [self.candidate_shard_path(sid, shards) for sid in range(shards)]
        missing = [
            str(path)
            for sid, path in enumerate(shard_paths)
            if not self.artifacts.candidate_shard_ready(path, sid, shards)
        ]
        if missing:
            raise FileNotFoundError('candidate shards missing before merge: ' + ', '.join(missing))
        self.run_charpost_ranker([
            'merge-candidates',
            '--inputs', ','.join(str(path) for path in shard_paths),
            '--expect-shards', str(shards),
            '--out', str(self.cfg.candidates_json),
        ])
        if not self.artifacts.candidate_manifest_ok(self.cfg.candidates_json, shards=shards):
            raise RuntimeError(f'charpost candidates failed validation: {self.cfg.candidates_json}')

    def ensure_candidates(self, shards: int = 1) -> None:
        if self.artifacts.candidate_manifest_ok(self.cfg.candidates_json, shards=shards):
            print('charpost candidates ready ->', self.cfg.candidates_json)
            return
        self.require_or_train('charpost candidate table')
        if shards == 1:
            self.run_charpost_ranker([
                'build-candidates',
                '--tag-prefix', self.cfg.charpost_oof_prefix,
                '--orders', self.order_arg,
                '--out', str(self.cfg.candidates_json),
            ])
        else:
            for sid in range(shards):
                self.ensure_candidate_shard(sid, shards)
            self.merge_candidate_shards(shards)
        if not self.artifacts.candidate_manifest_ok(self.cfg.candidates_json, shards=shards):
            raise RuntimeError(f'charpost candidates failed validation: {self.cfg.candidates_json}')

    def ensure_ranker(self, candidate_shards: int | None = None) -> None:
        if self.artifacts.ranker_ready(self.cfg.ranker_json):
            print('charpost ranker ready ->', self.cfg.ranker_json)
            return
        if self.artifacts.candidate_manifest_ok(self.cfg.candidates_json):
            print('charpost candidates ready ->', self.cfg.candidates_json)
        else:
            self.ensure_candidates(shards=candidate_shards or 1)
        self.run_charpost_ranker([
            'fit-rankers',
            '--candidates', str(self.cfg.candidates_json),
            '--out', str(self.cfg.ranker_json),
            '--features', self.cfg.charpost_ranker_features,
            '--methods', self.cfg.ranker_methods,
        ])
        if not self.artifacts.ranker_ready(self.cfg.ranker_json):
            raise RuntimeError(f'charpost ranker failed validation: {self.cfg.ranker_json}')

    def ensure_final_lm(self, order: int) -> None:
        lm_dir = self.cfg.charpost_dir / f'train_lm_order{order}_train'
        if self.artifacts.final_charpost_lm_ready(lm_dir / 'lm.json', order):
            print('final charpost LM ready ->', lm_dir / 'lm.json')
            return
        self.require_or_train(f'final charpost LM order {order}')
        self.run_char_lattice([
            'train-lm',
            '--order', str(order),
            '--splits', 'train',
            '--out-dir', str(lm_dir),
        ])
        if not self.artifacts.final_charpost_lm_ready(lm_dir / 'lm.json', order):
            raise RuntimeError(f'final charpost LM failed validation: {lm_dir / "lm.json"}')

    def ensure_final_lms(self) -> None:
        for order in self.cfg.charpost_orders:
            self.ensure_final_lm(order)

    def ensure_final_classifier(self) -> None:
        model_dir = self.cfg.charpost_dir / self.cfg.charpost_all_train_tag
        if not self.artifacts.charpost_classifier_ready(model_dir, tag=self.cfg.charpost_all_train_tag, init_ckpt=self.cfg.line_all_train):
            self.require_or_train('final all-train charpost classifier')
            self.run_char_lattice([
                'train-classifier',
                '--tag', self.cfg.charpost_all_train_tag,
                '--init-ckpt', str(self.cfg.line_all_train),
                '--out-dir', str(model_dir),
                '--input-mode', 'dual_pack',
                '--dual-mode', 'min',
                '--batch', '96',
                '--grad-accum', '1',
                '--max-steps', '3000',
                '--eval-steps', '0',
                '--log-steps', '5',
                '--state-steps', '250',
                '--lr', '1e-5',
                '--weight-decay', '1e-4',
                '--class-weight', 'sqrt',
                '--workers', '6',
                '--prefetch-factor', '4',
                '--image-cache-dir', str(self.cfg.image_cache_dir),
                '--require-image-cache',
                '--no-save-during-training',
            ])
            if not self.artifacts.charpost_classifier_ready(model_dir, tag=self.cfg.charpost_all_train_tag, init_ckpt=self.cfg.line_all_train):
                raise RuntimeError(f'final charpost classifier failed validation: {model_dir}')
        else:
            print('final charpost classifier ready ->', model_dir)

    def ensure_final_logits(self, split: str) -> None:
        if split not in {'val', 'test'}:
            raise ValueError(f'final logits split must be val or test; got {split!r}')
        model_dir = self.cfg.charpost_dir / self.cfg.charpost_all_train_tag
        if not self.artifacts.charpost_classifier_ready(model_dir, tag=self.cfg.charpost_all_train_tag, init_ckpt=self.cfg.line_all_train):
            raise FileNotFoundError(f'final charpost classifier missing before logits: {model_dir}')
        logits = self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__{split}_logits.npz'
        if not self.artifacts.final_charpost_logits_ready(split, model_dir):
            self.require_or_train(f'final charpost {split} logits')
            self.run_char_lattice([
                'cache-logits',
                '--tag', self.cfg.charpost_all_train_tag,
                '--model-dir', str(model_dir),
                '--splits', split,
                '--batch', '64',
                '--tta', 'shift5',
                '--tta-pixels', '4',
                '--tta-scale', '0.04',
                '--image-cache-dir', str(self.cfg.image_cache_dir),
                '--require-image-cache',
            ])
            if not self.artifacts.final_charpost_logits_ready(split, model_dir):
                raise RuntimeError(f'final charpost logits failed validation: {logits}')
        else:
            print('final charpost logits ready ->', logits)

    def ensure_final_artifacts(self) -> None:
        self.ensure_final_lms()
        self.ensure_final_classifier()
        for split in ('val', 'test'):
            self.ensure_final_logits(split)

    def ensure_decode(self, split: str, *, ensure_inputs: bool = True, device: str = 'auto') -> None:
        if split not in {'val', 'test'}:
            raise ValueError(f'decode split must be val or test; got {split!r}')
        if ensure_inputs:
            self.ensure_final_artifacts()
        else:
            model_dir = self.cfg.charpost_dir / self.cfg.charpost_all_train_tag
            if not self.artifacts.final_charpost_logits_ready(split, model_dir):
                raise FileNotFoundError(f'final charpost logits missing before decode: {split}')
            for order in self.cfg.charpost_orders:
                lm_path = self.cfg.charpost_dir / f'train_lm_order{order}_train' / 'lm.json'
                if not self.artifacts.final_charpost_lm_ready(lm_path, order):
                    raise FileNotFoundError(f'final charpost LM missing before decode: {lm_path}')
            if not self.artifacts.ranker_ready(self.cfg.ranker_json):
                raise FileNotFoundError(f'charpost ranker missing before decode: {self.cfg.ranker_json}')
        out = self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__{split}_decode.json'
        if self.artifacts.decode_ready(out, split):
            print('decode ready ->', out)
            return
        args = argparse.Namespace(
            tag=self.cfg.charpost_all_train_tag,
            split=split,
            ranker=self.cfg.ranker_json,
            orders=self.order_arg,
            lm_template=str(self.cfg.charpost_dir / 'train_lm_order{order}_train' / 'lm.json'),
            proposal_weights='1.0',
            beam=256,
            char_topk=8,
            device=device,
            row_shard_count=1,
            row_shard_id=0,
            no_score=False,
            keep_candidates=False,
            keep_features=False,
            out=out,
        )
        self.CR.command_decode(args)
        if not self.artifacts.decode_ready(out, split):
            raise RuntimeError(f'decode failed validation: {out}')

    def ensure_decodes(self) -> None:
        self.ensure_final_artifacts()
        for split in ('val', 'test'):
            self.ensure_decode(split, ensure_inputs=False)

    def summarize(self) -> None:
        print(json.dumps(self.artifacts.status(), indent=2, sort_keys=True))
        if self.cfg.ranker_json.exists():
            data = json.loads(self.cfg.ranker_json.read_text(encoding='utf-8'))
            best = data.get('best') or {}
            print('Charpost OOF best:', {
                'method': best.get('method'),
                'cer': best.get('cer'),
                'errors': best.get('errors'),
                'chars': best.get('chars'),
                'oracle_cer': best.get('oracle_cer'),
            })
            selector = data.get('final_selector') or {}
            print('Charpost selector:', {
                'method': selector.get('method'),
                'lm_weight': selector.get('lm_weight'),
            })
        for split in ('val', 'test'):
            path = self.cfg.charpost_dir / f'{self.cfg.charpost_all_train_tag}__{split}_decode.json'
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                print(f'Charpost {split} decode:', data.get('scored'))
