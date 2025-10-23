from __future__ import annotations

import torch
import torch.distributed as dist


def teacher_forward_logits(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    if not self.teacher_available:
        raise RuntimeError("Teacher logits requested but no teacher is available on this rank.")
    if self.teacher is not None and not self.teacher_rank0_only:
        from torch.cuda.amp import autocast
        input_ids_t = input_ids.to(self.teacher_device)
        attn_t = attn_mask.to(self.teacher_device)
        with torch.no_grad():
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits = self.teacher(
                    input_ids_t,
                    attention_mask=attn_t,
                    output_hidden_states=False,
                ).logits
        return self._sanitize_logits(logits, "teacher")
    return self._teacher_forward_distributed(input_ids, attn_mask, amp_enabled, amp_dtype)


def teacher_forward_distributed(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, amp_enabled: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    if not (self.teacher_rank0_only and self.ddp_enabled and dist.is_initialized()):
        raise RuntimeError("Distributed teacher forwarding requested without active rank-0 ownership.")

    payload = (input_ids.cpu(), attn_mask.cpu())
    gather_list = [None] * self.ddp_world_size if self.ddp_rank == 0 else None
    dist.gather_object(payload, gather_list, dst=0)

    outputs = None
    if self.ddp_rank == 0:
        outputs = []
        from torch.cuda.amp import autocast
        with torch.no_grad():
            for ids_cpu, mask_cpu in gather_list:
                if ids_cpu is None:
                    outputs.append(None)
                    continue
                ids = ids_cpu.to(self.teacher_device, non_blocking=True)
                mask = mask_cpu.to(self.teacher_device, non_blocking=True)
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    logits = self.teacher(
                        ids,
                        attention_mask=mask,
                        output_hidden_states=False,
                    ).logits
                outputs.append(self._sanitize_logits(logits, "teacher").detach().cpu())

    recv = [None]
    dist.scatter_object_list(recv, outputs if self.ddp_rank == 0 else None, src=0)
    logits_cpu = recv[0]
    if logits_cpu is None:
        raise RuntimeError("Distributed teacher failed to return logits for current rank.")
    return logits_cpu.to(self.student_device)
