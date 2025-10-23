from __future__ import annotations

from torch.optim import AdamW
import torch


class OOMSetupHelper:
    @staticmethod
    def configure(self, config):
        # ===== OOM Reduction: 8-bit optimizer to save ~2-3x memory =====
        try:
            from bitsandbytes.optim import Adam8bit
            self.opt = Adam8bit(self.student.parameters(), lr=config.lr)
            if self.ddp_rank == 0:
                print("[OOM-opt] Using 8-bit Adam optimizer to reduce memory.")
        except Exception:
            self.opt = AdamW(self.student.parameters(), lr=config.lr)
            if self.ddp_rank == 0:
                print("[OOM-opt] bitsandbytes not available, using standard AdamW.")

        self._printed_cache_info = False

        # ===== OOM Reduction: Enable gradient checkpointing on student =====
        student_base = self._student_base
        if hasattr(student_base, "gradient_checkpointing_enable"):
            try:
                student_base.gradient_checkpointing_enable()
                if self.ddp_rank == 0:
                    print("[OOM-opt] Enabled gradient checkpointing on student model.")
            except Exception as e:
                if self.ddp_rank == 0:
                    print(f"[OOM-opt] Could not enable gradient checkpointing: {e}")
        
        # ===== OOM Reduction: Enable memory-efficient attention =====
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            if self.ddp_rank == 0:
                print("[OOM-opt] Enabled memory-efficient SDPA backends.")
        except Exception:
            pass
        
        # Try Flash Attention 2 on student if supported
        if hasattr(student_base, "config") and hasattr(student_base.config, "use_flash_attention_2"):
            try:
                student_base.config.use_flash_attention_2 = True
                if self.ddp_rank == 0:
                    print("[OOM-opt] Enabled Flash Attention 2 on student model.")
            except Exception:
                pass
        
        # ===== OOM Reduction: Mixed precision (AMP) setup =====
        # Initialize GradScaler for fp16 AMP (disabled for bfloat16 which doesn't need scaling)
        self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._use_amp = True
        if self._amp_dtype == torch.float16:
            from torch.cuda.amp import GradScaler
            self._scaler = GradScaler(enabled=True)
            if self.ddp_rank == 0:
                print(f"[OOM-opt] Using AMP with {self._amp_dtype} and GradScaler.")
        else:
            self._scaler = None
            if self.ddp_rank == 0:
                print(f"[OOM-opt] Using AMP with {self._amp_dtype} (no scaler needed).")
