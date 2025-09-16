import time
import torch
import torch.nn.functional as F
from torch import device
from torch.optim import AdamW
import os
from pathlib import Path
import glob

from ..config import TrainingConfig, CheckpointData, TrainingMetrics


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy (the regular Entropy formula) in natural logarithm (base e).
    Used find high-entropy "fork" tokens / low-entropy certain tokens.

    logits: [seq_len, vocab]
    returns: [seq_len]"""
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(-1)  # [L, V] -> [L] where L is sequence length, V is vocabulary size


class Distiller:
    """Main distillation trainer class.
    It takes a teacher and a student model, a tokenizer, and a dataloader, then performs distillation -
    both EKD and both vanilla style."""

    def __init__(
            self,
            teacher_model,
            student_model,
            tokenizer,
            dataloader,
            config: TrainingConfig,
            teacher_device: device = "cuda",
            student_device: device = "cuda",
            logger=None,  # Combined logger for W&B + TensorBoard
    ):
        self.teacher = teacher_model.eval()  # frozen
        # self.teacher = teacher_model # todo: is it okay not to use the frozen? sounds like it should be frozen
        self.student = student_model.train()
        self.tok = tokenizer
        self.dataloader = dataloader
        self.config = config
        self.alpha_ce = 0.1  # Fixed value, could be added to config if needed
        self.opt = AdamW(self.student.parameters(), lr=config.lr)
        self.teacher_device = teacher_device
        self.student_device = student_device
        
        # Logging setup
        self.logger = logger
        self.global_step = 0
        
        # Checkpointing
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch: int, step: int):
        """Save a training checkpoint."""
        if self.config.checkpoint_steps <= 0:
            return
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'distill_type': self.config.distill_type,
            'top_k_percent': self.config.top_k_percent,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints."""
        if self.config.keep_checkpoints <= 0:
            return
        checkpoint_pattern = str(self.checkpoint_dir / "checkpoint_epoch*_step*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time

        # Remove old checkpoints beyond keep_checkpoints limit
        for old_checkpoint in checkpoint_files[self.config.keep_checkpoints:]:
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.student_device)
        
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint['epoch'], checkpoint['step']

    def _kl_loss(self, log_p: torch.Tensor, log_q: torch.Tensor):
        """KL(P||Q) where log_p are teacher log-probs, log_q are student log-probs."""
        return F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)

    @staticmethod
    def _sanitize_logits(x: torch.Tensor, name: str) -> torch.Tensor:
        # cast to fp32 for stability, clamp, and replace NaN/Inf
        x = x.float()
        x = torch.clamp(x, min=-1e4, max=1e4)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        if not torch.isfinite(x).all():
            print(f"[warn] non-finite after sanitize in {name}")
        return x
    
    def _forward_batch(self, batch):
        # Move inputs
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        input_ids_t = input_ids.to(self.teacher_device)
        attn_t = attn_mask.to(self.teacher_device)
        input_ids_s = input_ids.to(self.student_device)
        attn_s = attn_mask.to(self.student_device)

        T = 2.0  # distillation temperature; feel free to move to config
        T2 = T * T

        # ---- teacher ----
        with torch.no_grad():
            t_logits = self.teacher(input_ids_t, attention_mask=attn_t, output_hidden_states=False).logits
            t_logits = self._sanitize_logits(t_logits, "teacher")

        # ---- student ----
        s_logits = self.student(input_ids_s, attention_mask=attn_s).logits
        s_logits = self._sanitize_logits(s_logits, "student")

        # Align to next-token prediction
        t_pred = t_logits[:, :-1, :]  # [B, L-1, V]
        s_pred = s_logits[:, :-1, :]  # [B, L-1, V]
        valid_next = attn_s[:, 1:].bool()  # [B, L-1]

        # Softened log-probs for KD
        t_lp = torch.log_softmax(t_pred / T, dim=-1)  # [B, L-1, V]
        s_lp = torch.log_softmax(s_pred / T, dim=-1)  # [B, L-1, V]

        # KL per position (sum over vocab → [B, L-1])
        kl_pos = self._kl_loss(t_lp.to(self.student_device), s_lp)

        # --- KD loss ---
        if self.config.distill_type == "vanilla":
            denom = valid_next.sum().clamp(min=1)
            kd_loss = (kl_pos * valid_next).sum() / denom
        else:
            # EKD: top-k% entropy among valid positions only
            ent = token_entropy(t_pred).to(self.student_device)  # [B, L-1]
            kd_terms = []
            Bsz = kl_pos.size(0)
            for i in range(Bsz):
                vi = valid_next[i]
                if vi.sum() < 2:
                    continue
                ent_valid = ent[i][vi]
                thr = torch.quantile(ent_valid.float(), 1 - self.config.top_k_percent / 100.0)
                keep = torch.zeros_like(vi)
                keep[vi] = ent_valid >= thr
                if keep.any():
                    kd_terms.append(kl_pos[i][keep].mean())
            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else:
                denom = valid_next.sum().clamp(min=1)
                kd_loss = (kl_pos * valid_next).sum() / denom

        # Multiply by T^2 as in standard distillation
        kd_loss = kd_loss * T2

        # --- CE loss (only valid targets) ---
        logits = s_pred                      # [B, L-1, V]
        targets = input_ids_s[:, 1:]         # [B, L-1]
        targets = targets.masked_fill(~valid_next, -100)

        V = logits.size(-1)
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        keep = flat_targets.ne(-100)

        if keep.any():
            ce_loss = F.cross_entropy(flat_logits[keep], flat_targets[keep], reduction="mean")
        else:
            ce_loss = flat_logits.sum() * 0.0

        # Final loss
        total = kd_loss + self.alpha_ce * ce_loss

        # Last line of defense: skip bad batch
        if (not torch.isfinite(total)) or (not torch.isfinite(kd_loss)) or (not torch.isfinite(ce_loss)):
            print("[warn] skipping batch due to non-finite loss "
                f"(total={total.item()}, kd={kd_loss.item()}, ce={ce_loss.item()})")
            # Return a tiny zero-like loss with grad so autograd graph stays valid
            zero = (flat_logits.sum() * 0.0) + (flat_targets[keep][:0].sum() * 0.0)
            return zero + zero, 0.0, 0.0

        return total, kd_loss.item(), ce_loss.item()

    def train(self, epochs: int = 1, log_every: int = 100):
        """Run distillation training for specified number of epochs."""
        for epoch in range(epochs):
            step_start = time.time()
            running = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
            self.opt.zero_grad()  # Initialize gradients
            
            for step, batch in enumerate(self.dataloader):
                loss, kl_val, ce_val = self._forward_batch(batch)
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Only update weights after accumulation steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad()
                    self.global_step += 1
                    
                    # Save checkpoint if needed
                    if (self.config.checkpoint_steps > 0 and 
                        self.global_step % self.config.checkpoint_steps == 0):
                        self.save_checkpoint(epoch, step)

                # logging
                running["loss"] += loss.item() * self.config.gradient_accumulation_steps  # Unscale for logging
                running["kl"] += kl_val
                running["ce"] += ce_val
                
                # Logging every step using TrainingMetrics
                metrics = TrainingMetrics(
                    loss=loss.item() * self.config.gradient_accumulation_steps,
                    kl_loss=kl_val,
                    ce_loss=ce_val,
                    epoch=epoch + 1,
                    step=step + 1,
                    global_step=self.global_step
                )
                
                # Log metrics
                if self.logger:
                    log_metrics = {**metrics.to_dict(), "train/step": step + 1, "train/global_step": self.global_step}
                    self.logger.log(log_metrics, self.global_step)
                    
                if (step + 1) % log_every == 0:
                    n = log_every
                    avg_loss = running['loss'] / n
                    avg_kl = running['kl'] / n
                    avg_ce = running['ce'] / n
                    
                    elapsed = time.time() - step_start 
                    step_start = time.time()
                    print(
                        f"ep{epoch + 1} step{step + 1} | "
                        f"loss={avg_loss:.4f} kl={avg_kl:.4f} ce={avg_ce:.4f} "
                        f"| global_step={self.global_step} | {elapsed:.2f}s total, {elapsed/log_every:.2f}s/step"
                    )
                    
                    # Log averages using new combined logger or legacy loggers
                    avg_metrics = {
                        "train/avg_loss": avg_loss,
                        "train/avg_kl_loss": avg_kl,
                        "train/avg_ce_loss": avg_ce,
                        "train/elapsed_time": elapsed,
                        "train/steps_per_second": log_every / elapsed
                    }
                    
                    # Log averages
                    if self.logger:
                        self.logger.log(avg_metrics, self.global_step)
                        self.logger.flush()
                        
                    running = {k: 0.0 for k in running}
        
        # Final checkpoint and cleanup at the end
        if self.config.checkpoint_steps > 0:
            print("Training completed. Performing final cleanup of old checkpoints...")
            self._cleanup_old_checkpoints()
