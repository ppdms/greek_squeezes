#!/usr/bin/env python3
"""Loader for the shipped TrOCR line recognizer.

``load()`` builds the context dict (``g``) that char_lattice,
charpost_pipeline, report_figs, and modal_app consume: the dataset index,
crop extractors, device, and optionally the fine-tuned checkpoint or the
line-training helpers. All pieces are real functions in ``line_runtime`` /
``contest_evaluation``; this module just composes them and handles
checkpoint-specific configuration (run_config knobs, saved processors,
charset-constrained decoding).
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contest_evaluation as CE
import line_runtime as LR
import prep_cache

WORK = os.path.abspath(os.environ.get('SQUEEZE_WORK_DIR', 'data'))
CKPT = os.path.join(WORK, 'line_recognizer', 'all_train')


def _load_run_config(ckpt):
    """Find a checkpoint run_config.json (checkpoint dir or up to two parents)."""
    candidates = [
        os.path.join(ckpt, 'run_config.json'),
        os.path.join(os.path.dirname(ckpt), 'run_config.json'),
        os.path.join(os.path.dirname(os.path.dirname(ckpt)), 'run_config.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return json.load(open(path))
    return {}


def _apply_dual_run_config(run_config):
    """Mirror training-time dual packing knobs for rerank cache generation."""
    if not run_config.get('dual'):
        return
    os.environ['DUAL_MODE'] = str(run_config.get('dual_mode', 'mean'))
    os.environ['DUAL_DIVERGENT'] = str(run_config.get('dual_divergent', 'both'))
    os.environ['DUAL_REGISTER'] = '1' if bool(run_config.get('dual_register', False)) else '0'
    os.environ['DUAL_REGISTER_METHOD'] = str(run_config.get('dual_register_method', 'phase'))
    os.environ['DUAL_REGISTER_MAX_SHIFT'] = str(run_config.get('dual_register_max_shift', 0.15))
    os.environ['DUAL_CHANNEL_SWAP'] = '1' if bool(run_config.get('dual_channel_swap', False)) else '0'
    os.environ['DUAL_MONO_DROPOUT'] = str(float(run_config.get('dual_mono_dropout', 0.0)))


def _has_saved_processor(path):
    return ((os.path.exists(os.path.join(path, 'preprocessor_config.json'))
             or os.path.exists(os.path.join(path, 'processor_config.json'))
             or os.path.exists(os.path.join(path, 'image_processor_config.json')))
            and (os.path.exists(os.path.join(path, 'tokenizer_config.json'))
                 or os.path.exists(os.path.join(path, 'tokenizer.json'))))


def _processor_path_for_ckpt(ckpt):
    for path in (ckpt, os.path.dirname(ckpt), os.path.dirname(os.path.dirname(ckpt))):
        if path and _has_saved_processor(path):
            return path
    return ''


def _apply_input_run_config(g, run_config):
    """Mirror image-size / aspect-ratio knobs saved with the checkpoint."""
    size = int(run_config.get('image_size') or 0)
    if size:
        ip = g['processor'].image_processor
        ip.size = {'height': size, 'width': size}
        if hasattr(ip, 'crop_size'):
            ip.crop_size = {'height': size, 'width': size}
    if bool(run_config.get('keep_aspect', False)):
        old = g['prep_image']
        use_size = size or 384

        def prep_image_sized(pil):
            return old(pil, size=use_size)

        g['prep_image'] = prep_image_sized


def _alphabet_from_gt(g, splits=('train', 'val')):
    alpha = set()
    for _, r in g['index_df'][g['index_df']['split'].isin(list(splits))].iterrows():
        for ch in g['getRowTranscript'](r['ann_path']).upper():
            if not ch.isspace():
                alpha.add(ch)
    return sorted(alpha)


def _build_char_tokenizer(alphabet):
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    specials = ['<pad>', '</s>', '<s>', '<unk>']
    vocab = {tok: i for i, tok in enumerate(specials)}
    for ch in alphabet:
        vocab.setdefault(ch, len(vocab))
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    tok.pre_tokenizer = Split('', behavior='isolated')
    tok.decoder = decoders.Fuse()
    tok.post_processor = TemplateProcessing(
        single='<s> $A </s>',
        pair='<s> $A </s> $B:1 </s>:1',
        special_tokens=[('<s>', vocab['<s>']), ('</s>', vocab['</s>'])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token='<pad>',
        eos_token='</s>',
        sep_token='</s>',
        bos_token='<s>',
        cls_token='<s>',
        unk_token='<unk>',
    )


class _AllowedTokensLogitsProcessor:
    def __init__(self, allowed_ids, vocab_size, device):
        import torch
        mask = torch.full((vocab_size,), float('-inf'))
        mask[torch.tensor(sorted(allowed_ids), dtype=torch.long)] = 0.0
        self.mask = mask.to(device)

    def __call__(self, input_ids, scores):
        if scores.shape[-1] != self.mask.shape[-1]:
            raise RuntimeError(
                f'logits processor vocab mismatch: scores={scores.shape[-1]} mask={self.mask.shape[-1]}'
            )
        return scores + self.mask


def make_charset_logits_processor(processor, allowed_chars, device, mode='charset'):
    from transformers import LogitsProcessorList

    tok = processor.tokenizer
    allowed = set(c.upper() for c in allowed_chars)
    ids = set(int(i) for i in (getattr(tok, 'all_special_ids', []) or []) if i is not None)
    for tid in range(len(tok)):
        core = tok.decode([tid], skip_special_tokens=True).strip()
        if core == '':
            ids.add(tid)
            continue
        if not all(c.upper() in allowed for c in core):
            continue
        if mode == 'singlechar' and len(core) != 1:
            continue
        ids.add(tid)
    return LogitsProcessorList([_AllowedTokensLogitsProcessor(ids, len(tok), device)])


def load(ckpt=None, tile=8, verbose=True, load_model=True, train_context=False):
    """Build the line-recognizer context dict.

    ckpt = checkpoint dir (default: the module-level CKPT, resolved at call
    time so runtime_config.configure() can retarget it); tile = the
    LINE_MAX_CHARS the model was trained with
    (recorded as g['TILE'] — crop extraction callers pass max_chars explicitly;
    see the tiling note in line_runtime's module docstring).

    load_model=False skips the fine-tuned checkpoint (cache-only consumers that
    never run the model forward). train_context=True additionally provides the
    base processor, prep_image, and the training helpers (_augment, _AUG_RNG,
    MAX_LEN, compute_metrics) that the Modal line-training job needs — without
    loading any model weights.

    Returned keys (always): RUN_CONFIG, DEVICE, DTYPE, USE_FP16, empty_cache,
    index_df, images_df, ANN_DIR, IMG_DIR, SPLIT_SOURCE, getRowTranscript,
    to_gray, preprocess, preprocess_steps, extract_rows, extract_lines,
    LINE_MAX_CHARS, TILE, ALPHABET, ft_model (None unless load_model).
    With load_model or train_context: device, processor, prep_image, BASE_CKPT,
    KEEP_ASPECT, PROC_SIZE. With load_model: ft_model, CKPT_PATH,
    processor_path, LP_CHARSET. With train_context: MAX_LEN, _AUG_RNG,
    _augment, compute_metrics.
    """
    ckpt = ckpt or CKPT
    run_config = _load_run_config(ckpt)
    _apply_dual_run_config(run_config)

    g = {'RUN_CONFIG': run_config}
    g.update(LR.setup_environment())
    g.update(LR.build_index(WORK))
    g['getRowTranscript'] = CE.getRowTranscript
    g['to_gray'] = LR.to_gray
    g['preprocess'] = LR.preprocess
    g['preprocess_steps'] = LR.preprocess_steps
    extract_rows, extract_lines, lmc = LR.make_extractors(g['ANN_DIR'], g)
    g['extract_rows'], g['extract_lines'], g['LINE_MAX_CHARS'] = extract_rows, extract_lines, lmc
    # Optional on-disk preprocess cache (PREP_CACHE=1): replaces g['preprocess'],
    # which extract_rows resolves through g at call time.
    prep_cache.maybe_wrap(g, os.path.join(WORK, 'prep_cache'))
    g['TILE'] = tile
    g['ALPHABET'] = _alphabet_from_gt(g)

    if load_model or train_context:
        g.update(LR.build_trocr_context(g['DEVICE']))
    if train_context:
        g.update(LR.make_train_helpers(g['processor']))
    if not load_model:
        g['ft_model'] = None
        return g

    # Fine-tuned checkpoint: prefer its saved processor; reconstruct a
    # char-level tokenizer when the run trained with one but didn't save it.
    proc_path = _processor_path_for_ckpt(ckpt)
    if proc_path:
        from transformers import TrOCRProcessor
        g['processor'] = TrOCRProcessor.from_pretrained(proc_path)
        g['processor_path'] = proc_path
    elif run_config.get('char_vocab'):
        g['processor'].tokenizer = _build_char_tokenizer(g['ALPHABET'])
        g['processor_path'] = '<reconstructed-char-vocab>'
    _apply_input_run_config(g, run_config)

    from transformers import VisionEncoderDecoderModel
    from trocr_model import fix_trocr_meta

    proc = g['processor']
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False).to(g['device'])
    model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.generation_config.pad_token_id = proc.tokenizer.pad_token_id
    model.generation_config.eos_token_id = proc.tokenizer.sep_token_id
    model = fix_trocr_meta(model, g['device'])  # trocr-large pos-emb meta buffers
    g['ft_model'] = model.eval()
    g['CKPT_PATH'] = ckpt

    tok_vocab = len(proc.tokenizer)
    dec_vocab = int(getattr(model.config.decoder, 'vocab_size', model.config.vocab_size))
    if tok_vocab != dec_vocab:
        raise RuntimeError(f'tokenizer/model vocab mismatch for {ckpt}: tokenizer={tok_vocab} decoder={dec_vocab}')
    g['LP_CHARSET'] = make_charset_logits_processor(proc, g['ALPHABET'], g['device'], mode='charset')
    if verbose:
        print(
            f'line recognizer loaded: {os.path.basename(str(ckpt).rstrip(os.sep))} '
            f'tile={tile} device={g["device"]}',
            flush=True,
        )
    return g
