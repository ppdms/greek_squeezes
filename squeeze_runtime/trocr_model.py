from __future__ import annotations


def fix_trocr_meta(model, device):
    """Materialize TrOCR meta buffers created by newer transformers loaders."""
    import torch

    for name, buf in list(model.named_buffers()):
        if getattr(buf, 'is_meta', False):
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            parent.register_buffer(
                name.rsplit('.', 1)[-1],
                torch.zeros(buf.shape, dtype=buf.dtype, device=device),
                persistent=False,
            )
    for module in model.modules():
        weights = getattr(module, 'weights', None)
        if isinstance(weights, torch.Tensor) and getattr(weights, 'is_meta', False):
            get_emb = getattr(type(module), 'get_embedding', None)
            pad = getattr(module, 'padding_idx', None)
            try:
                module.weights = get_emb(weights.shape[0], weights.shape[1], pad).to(device)
            except Exception:
                module.weights = None
    return model
