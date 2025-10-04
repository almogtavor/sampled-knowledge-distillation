from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...training.offline_cache import TeacherOfflineCache


class CacheMixin:
    def _lookup_cache_batch(self, input_ids) -> Optional[List[Dict[str, Any]]]:
        if not self.cache:
            return None
        items = []
        for i in range(input_ids.size(0)):
            key = TeacherOfflineCache.key_from_ids(input_ids[i])
            if not self.cache.has(key):
                return None
            items.append(self.cache.read_item(key))
        return items

    def compute_cache_signature(self) -> Dict[str, Any]:
        """Compute a stable signature for the logits cache based on teacher/tokenizer/settings/dataset."""
        return {
            "teacher_name": getattr(getattr(self.teacher, "config", None), "_name_or_path", "unknown"),
            "tokenizer_name": getattr(self.tok, "name_or_path", "unknown"),
            "max_seq_len": int(self.config.max_seq_len),
            "entropy_approx_m": int(getattr(self.config, "entropy_approx_m", 12)),
            "rs_vocab_samples": int(getattr(self.config, "rs_vocab_samples", 12)),
            "rs_vocab_beta": float(getattr(self.config, "rs_vocab_beta", 1.0)),
            "entropy_approx_temperature": float(
                getattr(self.config, "entropy_approx_temperature", getattr(self.config, "cache_temperature", 1.0))
            ),
            "dataset_len": int(len(self.dataloader.dataset)) if hasattr(self.dataloader, "dataset") else -1,
        }
