"""Load a trained TIGER checkpoint and expose decoder layers for hooking.

The Tiger module (genrec/models/tiger.py) wraps the vendored minimal T5 in
genrec/modules/t5.py. Its decoder layers live at `model.model.decoder.block[l]`
(an nn.ModuleList of T5Block), which is where PopSteer captures / injects the
residual stream.
"""

from __future__ import annotations

from typing import Dict, Any, List

import torch
import torch.nn as nn

from genrec.models.tiger import Tiger


# Matches config/tiger/amazon/tiger.gin. vocab_size = codebook_size*sem_id_dim + 1.
DEFAULT_TIGER_CONFIG: Dict[str, Any] = dict(
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    vocab_size=769,
    pad_token_id=0,
    eos_token_id=0,
    feed_forward_proj="relu",
    sem_id_dim=3,
)


def load_tiger(
    checkpoint_path: str = "out/tiger/amazon/beauty/best_model.pt",
    config: Dict[str, Any] | None = None,
    device: str = "cuda",
) -> Tiger:
    """Instantiate Tiger with the training config and load the saved state_dict."""
    cfg = {**DEFAULT_TIGER_CONFIG, **(config or {})}
    model = Tiger(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Saved as model.state_dict() of the Tiger module.
    state = state.get("state_dict", state) if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state)
    model.eval()
    return model


def decoder_blocks(model: Tiger) -> List[nn.Module]:
    """The decoder T5Block list — hook targets for capture / injection."""
    return list(model.model.decoder.block)


def num_decoder_layers(model: Tiger) -> int:
    return len(decoder_blocks(model))
