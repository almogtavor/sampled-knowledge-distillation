"""Compatibility shim for legacy imports of `sampledkd.training.distiller`."""

from sampledkd.distill.trainer import Distiller, kl_from_vocab_samples

__all__ = ["Distiller", "kl_from_vocab_samples"]
