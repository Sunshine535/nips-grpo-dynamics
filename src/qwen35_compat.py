"""
Qwen3.5 compatibility patch for TRL GRPOTrainer (text-only training).

Qwen3.5 uses 3D position encoding (compute_3d_position_ids) designed for
multimodal inputs (images/videos). When TRL's GRPOTrainer processes text-only
inputs during log-prob computation, the 3D position delta tensor can be empty,
causing a RuntimeError shape mismatch.

This module patches the class method to:
  1. Fix stale rope_deltas batch-dimension mismatches (from generation step)
  2. Catch the dimension-1 shape mismatch and fall back to standard 1D positions

Import and call apply_qwen35_text_only_patch() before creating any model instance.
"""

import math
import torch


def apply_qwen35_text_only_patch():
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        return False

    model_cls = getattr(modeling_qwen3_5, "Qwen3_5Model", None)
    if model_cls is None or not hasattr(model_cls, "compute_3d_position_ids"):
        return False

    if getattr(model_cls, "_text_only_patched", False):
        return True

    original_fn = model_cls.compute_3d_position_ids

    def _safe_compute_3d_position_ids(self, *args, **kwargs):
        # --- Fix stale rope_deltas from a prior generation step ---
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and len(args) > 1:
            inputs_embeds = args[1]

        if getattr(self, "rope_deltas", None) is not None and inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            rd = self.rope_deltas
            if rd.numel() == 0:
                self.rope_deltas = None
            elif rd.ndim > 0:
                delta_rows = rd.shape[0]
                if delta_rows != batch_size:
                    if delta_rows > batch_size:
                        self.rope_deltas = rd[:batch_size]
                    elif batch_size % delta_rows == 0:
                        pass
                    else:
                        repeats = math.ceil(batch_size / delta_rows)
                        self.rope_deltas = rd.repeat_interleave(repeats, dim=0)[
                            :batch_size
                        ]

        # --- Call original, catch dim-1 shape mismatch for text-only inputs ---
        try:
            return original_fn(self, *args, **kwargs)
        except RuntimeError as e:
            if "must match the size" not in str(e):
                raise

            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(args) > 4:
                attention_mask = args[4]

            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
                device = inputs_embeds.device
            elif input_ids is not None:
                batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device
            else:
                raise

            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = torch.arange(seq_len, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            self.rope_deltas = None
            return position_ids

    model_cls.compute_3d_position_ids = _safe_compute_3d_position_ids
    model_cls._text_only_patched = True
    print("[qwen35_compat] Applied Qwen3.5 text-only position ID fallback patch")
    return True
