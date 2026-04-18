"""
PyTorch 2.5 compatibility patch for TRL.

TRL >= 0.15 imports `FSDPModule` from `torch.distributed.fsdp`, which only
exists in PyTorch 2.6+. This module monkey-patches the import so TRL works
on PyTorch 2.5 (CUDA 12.1).

Usage: import this module BEFORE importing trl.

    from src.torch_compat import apply_torch_compat_patch
    apply_torch_compat_patch()

    from trl import GRPOTrainer  # now works on PyTorch 2.5
"""

import logging
import sys

logger = logging.getLogger(__name__)

_patched = False


def apply_torch_compat_patch():
    """Patch torch.distributed.fsdp + vllm.distributed.utils for cross-version support."""
    global _patched
    if _patched:
        return

    # --- FSDPModule stub for PyTorch < 2.6 ---
    try:
        from torch.distributed.fsdp import FSDPModule  # noqa: F401
        logger.debug("FSDPModule available natively (PyTorch >= 2.6)")
    except ImportError:
        import torch
        import torch.distributed.fsdp as fsdp_module

        class FSDPModule(torch.nn.Module):
            """Stub for PyTorch < 2.6 compatibility."""
            pass

        fsdp_module.FSDPModule = FSDPModule
        if hasattr(fsdp_module, '__all__'):
            if 'FSDPModule' not in fsdp_module.__all__:
                fsdp_module.__all__.append('FSDPModule')
        logger.info("Patched torch.distributed.fsdp.FSDPModule stub (PyTorch %s)", torch.__version__)

    # --- StatelessProcessGroup stub for vllm < 0.7 ---
    try:
        from vllm.distributed.utils import StatelessProcessGroup  # noqa: F401
        logger.debug("vllm StatelessProcessGroup available")
    except (ImportError, Exception):
        try:
            import vllm.distributed.utils as vllm_utils

            class StatelessProcessGroup:
                """Stub for vllm < 0.7 compatibility. Not functional — only for import."""
                def __init__(self, *a, **kw):
                    raise RuntimeError("StatelessProcessGroup stub — install vllm>=0.7 for actual use")

            vllm_utils.StatelessProcessGroup = StatelessProcessGroup
            logger.info("Patched vllm.distributed.utils.StatelessProcessGroup stub")
        except ImportError:
            # vllm not installed — DON'T inject fake (breaks trl's _is_package_available check).
            # Just leave it; trl 0.15.x handles missing vllm gracefully via OptionalDependency.
            logger.info("vllm not installed; trl will use its OptionalDependency path")

    _patched = True
