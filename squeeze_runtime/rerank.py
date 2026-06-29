#!/usr/bin/env python3
"""Notebook-backed loader for the shipped TrOCR line recognizer."""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK = os.path.abspath(os.environ.get('SQUEEZE_WORK_DIR', 'data'))
CKPT = os.path.join(WORK, 'line_recognizer', 'all_train')


def _load_run_config(ckpt):
    """Find a checkpoint run_config.json, if present."""
    candidates = [
        os.path.join(ckpt, 'run_config.json'),
        os.path.join(os.path.dirname(ckpt), 'run_config.json'),
        os.path.join(os.path.dirname(os.path.dirname(ckpt)), 'run_config.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return json.load(open(path))
            except Exception:
                return {}
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


def _analysis_find(marker):
    from analysis_cells import ANALYSIS_CODE_CELLS

    for source in ANALYSIS_CODE_CELLS:
        if marker in source:
            return (
                source
                .replace("os.path.abspath('data')", repr(os.path.abspath(WORK)))
            )
    raise RuntimeError(f'helper cell not found: {marker!r}')


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


def load(ckpt=CKPT, tile=8, verbose=True, load_model=True):
    """Exec the minimal notebook cells (no slow build_pairs) and load a FT checkpoint.
    ckpt = path to a checkpoint dir; tile = LINE_MAX_CHARS the model was trained with (the crops
    at inference must match training tiling). Works for trocr-base and trocr-large checkpoints.
    load_model=False skips the FT checkpoint entirely (for cache-only analysis like viz_errors
    compare, which reranks from cached candidates and never runs the model forward)."""
    import shutil
    run_config = _load_run_config(ckpt)
    _apply_dual_run_config(run_config)
    g = {'__name__': '__main__', 'shutil': shutil}
    g['RUN_CONFIG'] = run_config
    g['DUAL_STREAM'] = False
    os.environ.setdefault('USE_SYNTH', '0')
    exec(_analysis_find('WORK_DIR  = os.path.abspath'), g)
    exec(_analysis_find('Shared setup: dataset index'), g)
    exec(_analysis_find('def to_gray').split('# Show the SAME samples')[0], g)  # defs only (skip slow demo)
    exec(_analysis_find('def run_evaluations'), g)                              # scorer + box readers
    g['LINE_MAX_CHARS'] = tile                                          # tile size for extract_rows/extract_lines
    line_extract = _analysis_find('# === Line-image extraction').split('# === Build the line-image dataset')[0]
    exec(line_extract, g)                                               # extract_rows/extract_lines only
    try:                                # optional on-disk preprocess cache (PREP_CACHE=1)
        import prep_cache as _pc        # scripts dir already on sys.path (top of file)
        _pc.maybe_wrap(g, os.path.join(WORK, 'prep_cache'))
    except Exception:
        pass
    if load_model:
        exec(_analysis_find('Baseline model: pretrained TrOCR'), g)     # processor, prep_image, make_logits_processor, baseline_model
        proc_path = _processor_path_for_ckpt(ckpt)
        if proc_path:
            g['processor'] = g['TrOCRProcessor'].from_pretrained(proc_path)
            g['processor_path'] = proc_path
        elif run_config.get('char_vocab'):
            alpha = _alphabet_from_gt(g)
            g['processor'].tokenizer = _build_char_tokenizer(alpha)
            g['processor_path'] = '<reconstructed-char-vocab>'
        _apply_input_run_config(g, run_config)
        exec(_analysis_find('Character n-gram LM for shallow-fusion'), g)  # char_lm, trocr_decode
    else:
        char_lm_defs = _analysis_find('Character n-gram LM for shallow-fusion').split('# Train on TRAIN+VAL row transcripts')[0]
        exec(char_lm_defs, g)                                             # CharNGramLM only

    # FT checkpoint.
    if not load_model:
        g['ft_model'] = None
        g['TILE'] = tile
        idf0 = g['index_df']
        alpha0 = set()
        for _, r in idf0[idf0['split'].isin(['train', 'val'])].iterrows():
            for ch in g['getRowTranscript'](r['ann_path']).upper():
                if not ch.isspace():
                    alpha0.add(ch)
        g['ALPHABET'] = sorted(alpha0)
        return g
    import torch
    from transformers import VisionEncoderDecoderModel
    model_cls = VisionEncoderDecoderModel

    proc = g['processor']
    model = model_cls.from_pretrained(ckpt, low_cpu_mem_usage=False).to(g['device'])
    model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.generation_config.pad_token_id = proc.tokenizer.pad_token_id
    model.generation_config.eos_token_id = proc.tokenizer.sep_token_id
    try:                                  # the validated meta fix (needed for trocr-large pos-emb)
        from trocr_model import fix_trocr_meta
        model = fix_trocr_meta(model, g['device'])
    except Exception as e:
        if verbose:
            print(f'(TrOCR meta-buffer fix unavailable: {e}; minimal fallback)', flush=True)
        for name, buf in list(model.named_buffers()):
            if getattr(buf, 'is_meta', False):
                parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
                parent.register_buffer(name.rsplit('.', 1)[-1],
                                       torch.zeros(buf.shape, dtype=buf.dtype, device=g['device']),
                                       persistent=False)
    g['ft_model'] = model.eval()
    g['TILE'] = tile
    g['CKPT_PATH'] = ckpt

    # Alphabet from GT (train+val) transcripts -> charset constraint (avoids build_pairs).
    g['ALPHABET'] = _alphabet_from_gt(g)
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
