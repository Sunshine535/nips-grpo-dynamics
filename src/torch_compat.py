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
    """Patch torch.distributed.fsdp to provide FSDPModule stub if missing."""
    global _patched
    if _patched:
        return

    try:
        from torch.distributed.fsdp import FSDPModule  # noqa: F401
        logger.debug("FSDPModule available natively (PyTorch >= 2.6)")
    except ImportError:
        import torch
        import torch.distributed.fsdp as fsdp_module

        # Create a dummy FSDPModule class
        class FSDPModule(torch.nn.Module):
            """Stub for PyTorch < 2.6 compatibility."""
            pass

        # Inject into the fsdp module
        fsdp_module.FSDPModule = FSDPModule

        # Also patch the specific import path TRL uses
        if hasattr(fsdp_module, '__all__'):
            if 'FSDPModule' not in fsdp_module.__all__:
                fsdp_module.__all__.append('FSDPModule')

        logger.info(
            "Patched torch.distributed.fsdp.FSDPModule stub "
            "(PyTorch %s < 2.6, FSDP2 not available)", torch.__version__
        )

    _patched = True
