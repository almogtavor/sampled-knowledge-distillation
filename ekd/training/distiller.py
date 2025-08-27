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
            writer=None,  # TensorBoard writer
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
        self.writer = writer
        self.global_step = 0  # For TensorBoard logging
        
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

    def _forward_batch(self, batch):
        # A 2D integer tensor of shape [batch_size, seq_len]
        input_ids_teacher = batch["input_ids"].to(self.teacher_device)
        # A 2D binary (0/1) tensor of shape [batch_size, seq_len]. 1 for real tokens, 0 for padding tokens
        attention_teacher = batch["attention_mask"].to(self.teacher_device)

        input_ids_student = batch["input_ids"].to(self.student_device)
        attention_student = batch["attention_mask"].to(self.student_device)

        with torch.no_grad():
            t_out = self.teacher(input_ids_teacher, attention_mask=attention_teacher, output_hidden_states=False)
            t_logits = t_out.logits  # [B, L, V]
            t_log_probs = torch.log_softmax(t_logits, dim=-1)
            ent = token_entropy(t_logits)  # [B, L]

        s_out = self.student(input_ids_student, attention_mask=attention_student)
        s_logits = s_out.logits
        s_log_probs = torch.log_softmax(s_logits, dim=-1)

        if self.config.distill_type == "vanilla":
            kl = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)  # [B, L]
            kd_loss = kl.mean()
        else:  # EKD
            # mask top-k% entropy tokens per example
            kd_losses = []
            for i in range(ent.size(0)):
                seq_ent = ent[i].float()  # Ensure float dtype for quantile
                threshold = torch.quantile(seq_ent, 1 - self.config.top_k_percent / 100.0)
                mask = seq_ent >= threshold  # high-entropy positions (fork tokens)
                kl_i = self._kl_loss(t_log_probs[i].to(self.student_device), s_log_probs[i])  # [L]
                if mask.any():
                    kd_losses.append(kl_i[mask].mean())
            kd_loss = torch.stack(kd_losses).mean()

        # optional small CE on every token to keep language ability
        ce_loss = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            input_ids_student.view(-1),
            ignore_index=self.tok.pad_token_id
        )
        loss = kd_loss + self.alpha_ce * ce_loss
        return loss, kd_loss.item(), ce_loss.item()

    def train(self, epochs: int = 1, log_every: int = 100):
        """Run distillation training for specified number of epochs."""
        for epoch in range(epochs):
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
                
                # TensorBoard logging every step using TrainingMetrics
                if self.writer is not None:
                    metrics = TrainingMetrics(
                        loss=loss.item() * self.config.gradient_accumulation_steps,
                        kl_loss=kl_val,
                        ce_loss=ce_val,
                        epoch=epoch + 1,
                        step=step + 1,
                        global_step=self.global_step
                    )
                    
                    # Log metrics to TensorBoard
                    for key, value in metrics.to_dict().items():
                        self.writer.add_scalar(key, value, self.global_step)
                    
                if (step + 1) % log_every == 0:
                    n = log_every
                    avg_loss = running['loss'] / n
                    avg_kl = running['kl'] / n
                    avg_ce = running['ce'] / n
                    
                    print(
                        f"ep{epoch + 1} step{step + 1} | loss={avg_loss:.4f} "
                        f"kl={avg_kl:.4f} ce={avg_ce:.4f} | global_step={self.global_step}"
                    )
                    
                    # Log averages to TensorBoard
                    if self.writer is not None:
                        self.writer.add_scalar("train/avg_loss", avg_loss, self.global_step)
                        self.writer.add_scalar("train/avg_kl_loss", avg_kl, self.global_step)
                        self.writer.add_scalar("train/avg_ce_loss", avg_ce, self.global_step)
                        self.writer.flush()
                        
                    running = {k: 0.0 for k in running}
        
        # Final checkpoint and cleanup at the end
        if self.config.checkpoint_steps > 0:
            print("Training completed. Performing final cleanup of old checkpoints...")
            self._cleanup_old_checkpoints()
