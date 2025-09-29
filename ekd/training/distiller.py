import time
import torch
import torch.nn.functional as F
from torch import device
from torch.optim import AdamW
import os
from pathlib import Path
import glob
import math
from typing import Dict, Any, List, Optional

from ..config import TrainingConfig, TrainingMetrics
from .offline_cache import TeacherOfflineCache, build_offline_cache_if_needed, init_offline_cache_for_trainer
from .entropy_utils import token_entropy


# importance-sampled KL over vocabulary using cached RS samples
def kl_from_vocab_samples(
    t_logp_sel: torch.Tensor, # [S] teacher log-probs at sampled tokens
    s_logp_sel: torch.Tensor, # [S] student log-probs at same tokens
    q_sel: torch.Tensor,  # [S] proposal probabilities used to draw the samples
    self_norm: bool = True,
) -> torch.Tensor:
    """
    Estimate KL(P||S) = sum_j p_j (log p_j - log s_j)
    with samples j ~ q. Use weights w_j = p_j / q_j.

    If self_norm=True (default): return sum_j (w_j / sum w) * (log p_j - log s_j)   (low-variance, slightly biased)
    Else (unbiased Horvitz-Thompson): return mean_j w_j * (log p_j - log s_j) with expectation over q.

    Inputs are 1D tensors for a single position. Returns a scalar tensor.
    """
    with torch.no_grad():
        p_sel = (t_logp_sel).exp()  # [S]
        w = p_sel / q_sel.clamp_min(1e-12)  # [S]
        if self_norm:
            w = w / w.sum().clamp_min(1e-12)
    # (log p - log s)
    diff = (t_logp_sel - s_logp_sel)  # [S]
    if self_norm:
        return (w * diff).sum()
    else:
        # Importance expectation E_q[w * diff] ≈ (1/S) * sum w * diff
        return (w * diff).mean()


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
        self.student = student_model.train()
        self.tok = tokenizer
        self.dataloader = dataloader
        self.config = config
        self.opt = AdamW(self.student.parameters(), lr=config.lr)
        self.teacher_device = teacher_device
        self.student_device = student_device
        self._printed_cache_info = False
        
        # Logging setup
        self.logger = logger
        self.global_step = 0
        
        # Checkpointing
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # offline teacher logits cache: centralized initialization
        self.cache = init_offline_cache_for_trainer(self)

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
            'k_percent': self.config.k_percent,
            # Record base/student/teacher identifiers to allow later export without extra CLI
            'base_model_dir': getattr(self.config, 'student_model', None),
            'student_model': getattr(self.config, 'student_model', None),
            'teacher_model': getattr(self.config, 'teacher_model', None),
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


    def _lookup_cache_batch(self, input_ids: torch.Tensor) -> Optional[List[Dict[str, Any]]]:
        if not self.cache:
            return None
        items = []
        for i in range(input_ids.size(0)):
            key = TeacherOfflineCache.key_from_ids(input_ids[i])
            if not self.cache.has(key):
                return None
            items.append(self.cache.read_item(key))
        return items

    def _entropy_for_selection(self, input_ids: torch.Tensor, t_pred: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return per-position entropy used only for *selection*.
        If cache is available for the whole batch, use cached H_hat (truncated entropy approximation).
        Otherwise, compute exact entropy online via t_pred.
        Output shape: [B, L-1]
        """
        if self.cache is not None:
            items = self._lookup_cache_batch(input_ids)
            if items is not None:
                # Stack cached per-example arrays
                H = [torch.as_tensor(it["H_hat"]) for it in items]  # each [L-1]
                return torch.stack(H, dim=0).to(self.student_device)  # [B, L-1]

        # need exact from t_pred
        assert t_pred is not None, "Exact entropy requested but teacher logits are unavailable."
        B = t_pred.size(0)
        ent_list = []
        for i in range(B):
            ent_list.append(token_entropy(t_pred[i]).to(self.student_device))  # [L-1]
        return torch.stack(ent_list, dim=0)  # [B, L-1]

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

    def _compute_kd_loss(self, t_pred: torch.Tensor, t_log_probs: torch.Tensor, 
                         s_log_probs: torch.Tensor, valid_next: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss based on the configured distillation type.
        
        Args:
            t_pred: Teacher predictions [B, L-1, V]
            t_log_probs: Teacher log probabilities [B, L-1, V] 
            s_log_probs: Student log probabilities [B, L-1, V]
            valid_next: Valid next token positions [B, L-1]
            
        Returns:
            kd_loss: Computed KD loss tensor
        """
        if self.config.distill_type == "vanilla":
            # KL for all positions for vanilla (sum over vocab so it becomes [B, L-1])
            kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)
            denom = valid_next.sum().clamp(min=1)
            kd_loss = (kl_pos * valid_next).sum() / denom
            
        elif self.config.distill_type == "top-k-tok":
            # top-k% tokens by entropy among valid positions only
            # selection by entropy (T=1 for selection is common) ----
            ent = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            ent = ent.masked_fill(~valid_next, float('-inf')) # ignore invalid

            pct = max(0.0, min(1.0, self.config.k_percent / 100.0)) # e.g. 0.2

            # top-k-tok: select positions first, then compute KL only on those
            kd_terms = []
            batch_size = valid_next.size(0)
            for i in range(batch_size):
                valid_next_i = valid_next[i]  # [L-1] - valid positions for sequence i
                n_valid = int(valid_next_i.sum().item())  # Count valid positions
                if n_valid < 3:
                    continue # Skip sequences with too few valid tokens
                k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                # Find indices of valid positions (all that aren't 0)
                valid_idx = torch.nonzero(valid_next_i, as_tuple=False).squeeze(-1)      # [n_valid]
                ent_valid_pos = ent[i][valid_idx] # [n_valid], entropy of valid positions
                _, selected_topk_pos = torch.topk(ent_valid_pos, k=k, largest=True, sorted=False)
                # Convert relative indices back to absolute sequence positions
                selected_topk_idx = valid_idx[selected_topk_pos] # [k]

                # Compute KL only on selected positions (sum over vocab, so [k])
                idx_t = selected_topk_idx.to(t_log_probs.device)
                kl_sel = self._kl_loss(t_log_probs[i, idx_t, :].to(self.student_device),
                        s_log_probs[i, selected_topk_idx, :])  # [k]
                kd_terms.append(kl_sel.mean())

            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else: # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient
        elif self.config.distill_type == "bucket":
            # Bucket: distill on tokens with entropy in [lower_bound, upper_bound] percentiles
            # e.g., if lower=70, upper=80: distill on tokens with 70th-80th percentile entropy
            # This excludes both the lowest 70% and highest 20% entropy tokens
            ent = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            kd_terms = []
            Bsz = t_log_probs.size(0)
            for i in range(Bsz):
                valid_next_i = valid_next[i]
                if valid_next_i.sum() < 3:  # Need at least 3 tokens for bucket selection
                    continue
                ent_valid = ent[i][valid_next_i]

                # Calculate both thresholds
                lower_thr = torch.quantile(ent_valid.float(), self.config.bucket_lower_percent / 100.0)
                upper_thr = torch.quantile(ent_valid.float(), self.config.bucket_upper_percent / 100.0)

                # Select tokens in the bucket [lower_thr, upper_thr]
                keep = torch.zeros_like(valid_next_i)
                keep[valid_next_i] = (ent_valid >= lower_thr) & (ent_valid <= upper_thr)
                if keep.any():
                    # Compute KL *only* for the selected tokens
                    kd_terms.append(
                        self._kl_loss(
                            t_log_probs[i][keep].to(self.student_device),
                            s_log_probs[i][keep]
                        ).mean()
                    )
            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else: # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient

        elif self.config.distill_type == "random":
            kd_terms = []
            Bsz = t_log_probs.size(0)
            for i in range(Bsz):
                valid_next_i = valid_next[i]
                valid_count = valid_next_i.sum().item()
                if valid_count < 2:
                    continue
                k_count = max(1, int(valid_count * self.config.k_percent / 100.0))
                valid_indices = torch.where(valid_next_i)[0]
                if len(valid_indices) >= k_count:
                    perm = torch.randperm(len(valid_indices), device=self.student_device)
                    selected_indices = valid_indices[perm[:k_count]]
                    keep = torch.zeros_like(valid_next_i)
                    keep[selected_indices] = True
                    if keep.any():
                        # Compute KL *only* for the selected tokens
                        kd_terms.append(
                            self._kl_loss(
                                t_log_probs[i][keep].to(self.student_device),
                                s_log_probs[i][keep]
                            ).mean()
                        )
            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else: # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient
        elif self.config.distill_type == "pos-rs-kd":
            # RS-KD over POSITIONS: sample K% positions by entropy-based distribution q(i)
            # q(i) ∝ H_i^alpha ;  q ← (1-ε) q + ε·uniform ; floor on q to avoid degeneracy
            ent = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            kd_terms = []
            Bsz = t_pred.size(0)
            alpha = float(getattr(self.config, "rs_alpha", 1.0))
            q_floor = float(getattr(self.config, "rs_floor", 1e-6))
            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
        
            for i in range(Bsz):
                valid_next_i = valid_next[i]  # [L-1] bool of valid positions
                valid_count = int(valid_next_i.sum().item())
                if valid_count < 3:
                    continue
                # Entropy over valid positions
                ent_valid = ent[i][valid_next_i].float() # [n_valid]
                ent_valid = torch.clamp(ent_valid, min=1e-8) # Make strictly positive for exponentiation
                # q_un ∝ H^alpha (Unnormalized proposal distribution). Higher H -> higher sampling prob
                if alpha == 0.0:
                    q_un = torch.ones_like(ent_valid) # uniform if alpha=0
                else:
                    q_un = ent_valid.pow(alpha)
        
                q_un_sum = q_un.sum()
                if q_un_sum <= 0:
                    # fallback to uniform if degenerate
                    q = torch.full_like(q_un, 1.0 / q_un.numel())
                else:
                    q = q_un / q_un_sum  # normalize
                    # floor to avoid zero probs
                    q = torch.clamp(q, min=q_floor)
                    q = q / q.sum()  # renormalize
        
                k_count = max(1, int(valid_count * pct))
                # Sample indices within the "valid positions" subspace
                # torch.multinomial expects probs sum to 1
                sel_rel = torch.multinomial(q, num_samples=k_count, replacement=False)  # [k]
        
                # Map relative -> absolute indices
                valid_idx = torch.where(valid_next_i)[0]  # absolute positions
                selected_abs = valid_idx[sel_rel]  # [k]
                # KL on selected positions (like top-k branch)
                idx_t = selected_abs.to(t_log_probs.device)
                kl_sel = self._kl_loss(
                    t_log_probs[i, idx_t, :].to(self.student_device),
                    s_log_probs[i, selected_abs, :]
                )  # [k]
        
                # Importance weights ∝ 1/q(i). Normalize to sum=1 to keep scale comparable.
                q_sel = q[sel_rel]  # [k]
                w = 1.0 / torch.clamp(q_sel, min=q_floor)
                w = w / w.sum()
                kd_terms.append((kl_sel * w).sum())
            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else: # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient
        else:
            raise ValueError(f"Unknown distill_type: {self.config.distill_type}")
            
        return kd_loss

    @staticmethod
    def _sanitize_logits(x: torch.Tensor, name: str) -> torch.Tensor:
        """Sanitize logits to prevent NaNs/Infs during training.
        We might train with lower precision (e.g., fp16), so instability might occur.
        """
        # cast to fp32 for stability, clamp, and replace NaN/Inf
        x = x.float()
        x = torch.clamp(x, min=-1e4, max=1e4)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        if not torch.isfinite(x).all():
            print(f"[warn] non-finite after sanitize in {name}")
        return x
    
    def _forward_batch(self, batch):
        # Move inputs
        input_ids = batch["input_ids"] # [B, L]
        attn_mask = batch["attention_mask"]
        input_ids_s = input_ids.to(self.student_device)
        attn_mask_s = attn_mask.to(self.student_device)

        T = 2.0  # distillation temperature
        T2 = T * T

        # --- student forward (always) ---
        s_logits = self.student(input_ids_s, attention_mask=attn_mask_s).logits
        s_logits = self._sanitize_logits(s_logits, "student")

        # Align to next-token prediction
        s_pred = s_logits[:, :-1, :]  # [B, L-1, V]
        valid_next = attn_mask_s[:, 1:].bool()  # [B, L-1]
        s_log_probs = torch.log_softmax(s_pred / T, dim=-1)  # [B, L-1, V]

        # Decide whether we need full teacher logits now
        cached_items = None
        use_vocab_rs_kd = bool(getattr(self.config, "offline_cache", False))
        need_teacher_full = self.config.distill_type in {"vanilla", "top-k-tok", "bucket", "pos-rs-kd"}
        if use_vocab_rs_kd:
            # try offline cache to avoid teacher forward; fall back to teacher if cache missing
            cached_items = self._lookup_cache_batch(input_ids)
            if cached_items is not None:
                need_teacher_full = False

        # Inform per-batch whether we hit the logits cache or run the teacher online
        if not self._printed_cache_info:
            if not need_teacher_full and cached_items is not None:
                print("[logits-cache] Using cached teacher logits/statistics (no online teacher forward).")
            elif not need_teacher_full:
                print("[logits-cache] Cache considered but empty for this batch; skipping teacher forward due to mode.")
            elif getattr(self.config, "offline_cache", False):
                print("[logits-cache] Cache miss or not applicable - running online teacher forward.")
            else:
                print("[logits-cache] Disabled - running online teacher forward.")
            self._printed_cache_info = True

        if need_teacher_full:
            input_ids_t = input_ids.to(self.teacher_device)
            attn_t = attn_mask.to(self.teacher_device)
            with torch.no_grad():
                t_logits = self.teacher(input_ids_t, attention_mask=attn_t, output_hidden_states=False).logits
                t_logits = self._sanitize_logits(t_logits, "teacher") # [B, L, V]
            t_pred = t_logits[:, :-1, :]  # [B, L-1, V]
            t_log_probs = torch.log_softmax(t_pred / T, dim=-1)  # [B, L-1, V]
        else:
            t_pred = t_log_probs = None
            # message already printed above

        # --- KD loss ---
        # Compute per-position base KD according to mode (positions subset and weights),
        # then replace vocab-sum with RS-KD estimator when enabled.
        if not use_vocab_rs_kd:
            kd_loss = self._compute_kd_loss(t_pred, t_log_probs, s_log_probs, valid_next, input_ids)
        else:
            kd_terms = []
            B = s_log_probs.size(0)
            for i in range(B):
                valid_i = valid_next[i]
                if valid_i.sum() == 0:
                    continue

                # Gather RS-KD lists from cache or online fallback using teacher logp
                if cached_items is not None:
                    rs = cached_items[i]["rs"]
                    idx_list = rs["idx"]
                    tlogp_list = rs["t_logp"]
                    q_list = rs["q"]
                else:
                    # Need teacher logprobs for online proposal
                    assert t_log_probs is not None, "Teacher logits required for online RS-KD when cache missing"
                    with torch.no_grad():
                        t_full_i = t_log_probs[i]
                        p_i = t_full_i.exp()
                    beta = float(getattr(self.config, "rs_vocab_beta", 1.0))
                    S_vocab = int(getattr(self.config, "rs_vocab_samples", 64))
                    idx_list, tlogp_list, q_list = [], [], []
                    for pos, is_valid in enumerate(valid_i.tolist()):
                        if not is_valid:
                            idx_list.append(torch.empty(0, dtype=torch.long))
                            tlogp_list.append(torch.empty(0))
                            q_list.append(torch.empty(0))
                            continue
                        p_pos = p_i[pos]
                        q_un = p_pos.clamp_min(1e-12).pow(beta) if beta != 1.0 else p_pos.clamp_min(1e-12)
                        q = q_un / q_un.sum()
                        S = min(S_vocab, q.numel())
                        idx = torch.multinomial(q, num_samples=S, replacement=False)
                        idx_list.append(idx.cpu())
                        tlogp_list.append(t_full_i[pos, idx].cpu())
                        q_list.append(q[idx].cpu())

                # For each position, compute KL estimate using only sampled tokens
                for pos, is_valid in enumerate(valid_i.tolist()):
                    if not is_valid:
                        continue
                    idx = idx_list[pos].to(self.student_device)
                    if idx.numel() == 0:
                        continue
                    t_logp_sel = tlogp_list[pos].to(self.student_device)
                    q_sel = q_list[pos].to(self.student_device)
                    s_logp_sel = s_log_probs[i, pos, idx]
                    kd_terms.append(kl_from_vocab_samples(t_logp_sel, s_logp_sel, q_sel, self_norm=True))

            kd_loss = torch.stack(kd_terms).mean() if kd_terms else s_pred.sum() * 0.0
        
        # Temperature factor (keep gradients comparable across T, as in standard distillation)
        kd_loss = kd_loss * T2

        # --- CE loss (only valid targets) ---
        if self.config.enable_ce:
            logits = s_pred  # [B, L-1, V]
            targets = input_ids_s[:, 1:] # [B, L-1]
            targets = targets.masked_fill(~valid_next, -100)

            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
            flat_targets = targets.reshape(-1)
            keep = flat_targets.ne(-100)

            if keep.any():
                ce_loss = F.cross_entropy(flat_logits[keep], flat_targets[keep], reduction="mean")
            else:
                ce_loss = flat_logits.sum() * 0.0
            total = (1.0 - self.config.alpha_ce) * kd_loss + self.config.alpha_ce * ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=self.student_device)
            # Pure KD loss when CE is disabled
            total = kd_loss

        # Last line of defense: skip bad batch
        if (not torch.isfinite(total)) or (not torch.isfinite(kd_loss)) or (not torch.isfinite(ce_loss)):
            print("[warn] skipping batch due to non-finite loss "
                f"(total={total.item()}, kd={kd_loss.item()}, ce={ce_loss.item()})")
            # Return a tiny zero-like loss with grad so autograd graph stays valid
            zero = s_pred.sum() * 0.0
            return zero + zero, 0.0, 0.0

        return total, kd_loss.item(), ce_loss.item()


    def train(self, epochs: int = 1, log_every: int = 100):
        """Run distillation training for specified number of epochs."""
        # make the offline pass once, if requested
        build_offline_cache_if_needed(self)
        for epoch in range(epochs):
            step_start = time.time()
            running = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
            self.opt.zero_grad(set_to_none=True)  # Initialize gradients
            
            for step, batch in enumerate(self.dataloader):
                loss, kl_val, ce_val = self._forward_batch(batch)
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Only update weights after accumulation steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
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
