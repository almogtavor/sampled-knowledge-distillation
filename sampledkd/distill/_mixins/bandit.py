from __future__ import annotations

from typing import Optional

from ...training.linucb_controller import LinUCBBanditController


class BanditMixin:
    bandit_manager: Optional[LinUCBBanditController]

    def _init_bandit_manager(self):
        self.bandit_manager = None
        if self.config.distill_type == "linucb":
            self.bandit_manager = LinUCBBanditController(
                tokenizer=self.tok,
                config=self.config,
                student_device=self.student_device,
                teacher_device=self.teacher_device,
                sanitize_logits_fn=self._sanitize_logits,
            )
