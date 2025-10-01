import glob
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import device
from torch.optim import AdamW

from ..config import TrainingConfig, TrainingMetrics
from .entropy_utils import token_entropy
from .linucb_controller import LinUCBBanditController
from .offline_cache import (
    TeacherOfflineCache,
    build_offline_cache_if_needed,
    init_offline_cache_for_trainer,
)


# importance-sampled KL over vocabulary using cached RS samples
def kl_from_vocab_samples(
    t_logp_sel: torch.Tensor, # [S] teacher log-probs at sampled tokens
    s_logp_sel: torch.Tensor, # [S] student log-probs at same tokens
    q_sel: torch.Tensor,  # [S] proposal probabilities used to draw the samples
    self_norm: bool = True,
    T_kd: float = 1.0,
) -> torch.Tensor:
    """
    Estimate KL(P||S) = sum_j p_j (log p_j - log s_j)
    with samples j ~ q. Use weights w_j = p_j / q_j.

    If self_norm=True (default): return sum_j (w_j / sum w) * (log p_j - log s_j)   (low-variance, slightly biased)
    Else (unbiased Horvitz-Thompson): return mean_j w_j * (log p_j - log s_j) with expectation over q.

    Inputs are 1D tensors for a single position. Returns a scalar tensor.
    """
    with torch.no_grad():
        # teacher probabilities at T=1 (cache stores t_logp_sel at T=1)
        p1 = t_logp_sel.exp()
        # retemper to current KD temperature: p_T ∝ p_1^{1/T_kd}
        gamma = 1.0 / max(1e-12, T_kd)
        pT_un = p1.pow(gamma)
        w = pT_un / q_sel.clamp_min(1e-12)  # [S]
        if self_norm:
            w = w / w.sum().clamp_min(1e-12)
    # approximate log p_T on sampled ids up to a constant
    logpT_approx = torch.log(pT_un) - torch.log(pT_un.sum().clamp_min(1e-12))
    diff = (logpT_approx - s_logp_sel)  # [S]
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
        self.cache = None

        # LinUCB contextual bandit setup
        self.bandit_manager: Optional[LinUCBBanditController] = None
        if self.config.distill_type == "linucb":
            self.bandit_manager = LinUCBBanditController(
                tokenizer=self.tok,
                config=self.config,
                student_device=self.student_device,
                teacher_device=self.teacher_device,
                sanitize_logits_fn=self._sanitize_logits,
            )

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
        # If t_pred is None, we are *expecting* to use the cache. Hard-fail on any miss.
        if t_pred is None:
            if self.cache is None:
                raise RuntimeError("Entropy requested with t_pred=None but no cache is set.")
            items = self._lookup_cache_batch(input_ids)
            if items is None:
                raise RuntimeError("Cache miss: at least one example not present in the offline cache.")
            H_list = []
            for it in items:
                if "H_hat_u8" in it:
                    H_u8 = torch.as_tensor(it["H_hat_u8"], dtype=torch.uint8)
                    V = int(it.get("rs", {}).get("sentinel_id", 0))
                    H_cap = math.log(max(2, V)) if V > 0 else 1.0
                    H_f = (H_u8.float() / 255.0) * H_cap
                    H_list.append(H_f)
                elif "H_hat" in it:
                    H_list.append(torch.as_tensor(it["H_hat"]).float())
                else:
                    raise RuntimeError("Cache item lacks H_hat / H_hat_u8; cache is incomplete.")
            return torch.stack(H_list, dim=0).to(self.student_device)
        assert t_pred is not None, "Exact entropy requested but teacher logits are unavailable."
        B = t_pred.size(0)
        ent_list = []
        for i in range(B):
            ent_list.append(token_entropy(t_pred[i]).to(self.student_device))  # [L-1]
        return torch.stack(ent_list, dim=0)  # [B, L-1]

    @staticmethod
    def _normalize_component(values: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
        """Normalize `values` over the True positions of `mask` according to `mode`.

        Returns a tensor of the same shape as `values` (no masking applied).
        """
        if mode == "none":
            return values

        valid_vals = values[mask]
        if valid_vals.numel() == 0:
            return values

        if mode == "z":
            mean = valid_vals.mean()
            std = valid_vals.std(unbiased=False)
            if std < 1e-6:
                return values - mean
            return (values - mean) / std
        elif mode == "minmax":
            min_val = valid_vals.min()
            max_val = valid_vals.max()
            denom = (max_val - min_val).clamp_min(1e-6)
            return (values - min_val) / denom
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def _prepare_score_context(
        self,
        ent_raw: torch.Tensor,
        kl_pos: torch.Tensor,
        s_log_probs: Optional[torch.Tensor],
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        student_ce_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        weights = (
            float(getattr(self.config, "score_entropy_weight", 1.0)),
            float(getattr(self.config, "score_ce_weight", 1.0)),
            float(getattr(self.config, "score_kl_weight", 1.0)),
        )
        if all(abs(w) < 1e-8 for w in weights):
            raise ValueError("Score-based selection requires at least one non-zero component weight.")

        norm_mode = getattr(self.config, "score_normalize", "z")

        # entropy / kl components
        ent_for_score = ent_raw.detach().masked_fill(~valid_next, 0.0)
        kl_for_score = kl_pos.detach().masked_fill(~valid_next, 0.0)

        # student CE component
        if student_ce_override is not None:
            student_ce = student_ce_override.detach().masked_fill(~valid_next, 0.0)
        else:
            if s_log_probs is None:
                raise ValueError("s_log_probs is required unless student_ce_override is provided.")
            targets = input_ids[:, 1:].to(self.student_device)
            targets = targets.masked_fill(~valid_next, 0)
            target_logp = s_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            student_ce = (-target_logp).detach().masked_fill(~valid_next, 0.0)

        return {
            "weights": weights,
            "norm_mode": norm_mode,
            "entropy": ent_for_score,
            "student_ce": student_ce,
            "kl": kl_for_score,
        }

    def _build_score_vector(
        self,
        score_ctx: Dict[str, Any],
        example_idx: int,
        mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Construct combined score for one example or return None if no components are active."""
        weights = score_ctx["weights"]
        norm_mode = score_ctx["norm_mode"]

        combined = torch.zeros_like(score_ctx["entropy"][example_idx])
        component_used = False

        if abs(weights[0]) > 0:
            comp = self._normalize_component(score_ctx["entropy"][example_idx].clone(), mask, norm_mode)
            combined = combined + weights[0] * comp
            component_used = True

        if abs(weights[1]) > 0:
            comp = self._normalize_component(score_ctx["student_ce"][example_idx].clone(), mask, norm_mode)
            combined = combined + weights[1] * comp
            component_used = True

        if abs(weights[2]) > 0:
            comp = self._normalize_component(score_ctx["kl"][example_idx].clone(), mask, norm_mode)
            combined = combined + weights[2] * comp
            component_used = True

        if not component_used:
            return None

        combined = combined.masked_fill(~mask, float('-inf'))
        return combined

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

    # === Sampled softmax helpers (used only in cached RS-KD path when eliminate_softmax=True) ===
    def _proposal_sample_negatives(self, V: int, M: int, device: torch.device) -> torch.Tensor:
        """Draw M uniform 'negatives' from [0, V). Overlap with U/avoid is allowed."""
        if M <= 0 or V <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.randint(low=0, high=V, size=(M,), device=device, dtype=torch.long)

    def _student_log_probs_sampled(self, z_sel: torch.Tensor, q_sel: torch.Tensor, T: float) -> torch.Tensor:
        """Return log softmax over the sampled set S with importance correction: log s(i) ∝ z_i/T - log q(i)."""
        z_corr = z_sel / T - torch.log(q_sel.clamp_min(1e-12))
        logZ = torch.logsumexp(z_corr, dim=-1, keepdim=False)
        return z_corr - logZ

    def _compute_kd_loss(
        self,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Compute knowledge distillation loss for the configured strategy.

        Args:
            t_pred: Teacher predictions [B, L-1, V]
            t_log_probs: Teacher log probabilities [B, L-1, V]
            s_pred: Student logits aligned to next-token positions [B, L-1, V]
            s_log_probs: Student log probabilities at KD temperature [B, L-1, V]
            valid_next: Boolean mask of valid next-token positions [B, L-1]
            input_ids: Input ids [B, L]
            attention_mask: Attention mask [B, L]
            temperature: KD temperature used for the current batch

        Returns:
            Tuple containing the KD loss tensor and optional auxiliary info (for LinUCB).
        """
        extra: Optional[Dict[str, Any]] = None

        if self.config.distill_type == "vanilla":
            kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)
            denom = valid_next.sum().clamp(min=1)
            kd_loss = (kl_pos * valid_next).sum() / denom
        elif self.config.distill_type == "top-k-tok":
            # top-k% tokens by entropy among valid positions only
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            ent = ent_raw.masked_fill(~valid_next, float('-inf'))  # ignore invalid

            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))  # e.g. 0.2

            # optional score-based selection context
            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            score_ctx = None
            if score_enabled:
                kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)  # [B, L-1]
                score_ctx = self._prepare_score_context(ent_raw, kl_pos, s_log_probs, valid_next, input_ids)

            # select positions first, then compute KL only on those
            kd_terms = []
            batch_size = valid_next.size(0)
            for i in range(batch_size):
                mask = valid_next[i]  # [L-1] - valid positions for sequence i
                n_valid = int(mask.sum().item())
                if n_valid < 3:
                    continue
                k = max(1, min(n_valid, math.ceil(pct * n_valid)))

                if score_enabled:
                    combined = self._build_score_vector(score_ctx, i, mask)
                    if combined is None:
                        continue
                    score_valid = combined[mask]
                    if score_valid.numel() == 0:
                        continue
                    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [n_valid]
                    _, rel_idx = torch.topk(score_valid, k=k, largest=True, sorted=False)
                    selected_topk_idx = valid_idx[rel_idx]  # [k]
                else:
                    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [n_valid]
                    ent_valid_pos = ent[i][valid_idx] # [n_valid], entropy of valid positions
                    _, selected_topk_pos = torch.topk(ent_valid_pos, k=k, largest=True, sorted=False)
                    # Convert relative indices back to absolute sequence positions
                    selected_topk_idx = valid_idx[selected_topk_pos]  # [k]
                # Compute KL only on selected positions (sum over vocab, so [k])
                idx_t = selected_topk_idx.to(t_log_probs.device)
                kl_sel = self._kl_loss(
                    t_log_probs[i, idx_t, :].to(self.student_device),
                    s_log_probs[i, selected_topk_idx, :]
                )  # [k]
                kd_terms.append(kl_sel.mean())

            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else:
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient

        elif self.config.distill_type == "bucket":
            # Bucket: distill on tokens with entropy in [lower_bound, upper_bound] percentiles
            # e.g., if lower=70, upper=80: distill on tokens with 70th-80th percentile entropy
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]

            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            score_ctx = None
            kl_pos = None
            if score_enabled:
                kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)  # [B, L-1]
                score_ctx = self._prepare_score_context(ent_raw, kl_pos, s_log_probs, valid_next, input_ids)

            kd_terms = []
            Bsz = t_log_probs.size(0)
            for i in range(Bsz):
                valid_next_i = valid_next[i]
                if valid_next_i.sum() < 3:  # Need at least 3 tokens for bucket selection
                    continue

                if score_enabled:
                    combined = self._build_score_vector(score_ctx, i, valid_next_i)
                    if combined is None:
                        continue
                    vec = combined[valid_next_i].float()
                else:
                    vec = ent_raw[i][valid_next_i].float()

                lower_thr = torch.quantile(vec, self.config.bucket_lower_percent / 100.0)
                upper_thr = torch.quantile(vec, self.config.bucket_upper_percent / 100.0)

                keep = torch.zeros_like(valid_next_i)
                keep_idx = torch.where(valid_next_i)[0]
                sel_mask = (vec >= lower_thr) & (vec <= upper_thr)
                if sel_mask.any():
                    keep[keep_idx[sel_mask]] = True
                    kd_terms.append(
                        self._kl_loss(
                            t_log_probs[i][keep].to(self.student_device),
                            s_log_probs[i][keep]
                        ).mean()
                    )

            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else:  # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient
        elif self.config.distill_type == "random":
            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            score_ctx = None
            kl_pos = None
            if score_enabled:
                ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1] for context only
                kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)  # [B, L-1]
                score_ctx = self._prepare_score_context(ent_raw, kl_pos, s_log_probs, valid_next, input_ids)

            kd_terms = []
            Bsz = t_log_probs.size(0)
            for i in range(Bsz):
                valid_next_i = valid_next[i]
                valid_count = int(valid_next_i.sum().item())
                if valid_count < 2:
                    continue
                k_count = max(1, int(valid_count * self.config.k_percent / 100.0))
                valid_indices = torch.where(valid_next_i)[0]
                if len(valid_indices) < k_count:
                    continue

                if score_enabled:
                    combined = self._build_score_vector(score_ctx, i, valid_next_i)
                    if combined is None:
                        continue
                    score_valid = combined[valid_next_i].float()
                    # turn scores into sampling probs
                    s = score_valid - score_valid.min()
                    s = torch.clamp(s, min=1e-8)
                    probs = s / s.sum()
                    rel = torch.multinomial(probs, num_samples=k_count, replacement=False)
                    selected_indices = valid_indices[rel]
                else:
                    perm = torch.randperm(len(valid_indices), device=self.student_device)
                    selected_indices = valid_indices[perm[:k_count]]

                keep = torch.zeros_like(valid_next_i)
                keep[selected_indices] = True
                if keep.any():
                    kd_terms.append(
                        self._kl_loss(
                            t_log_probs[i][keep].to(self.student_device),
                            s_log_probs[i][keep]
                        ).mean()
                    )

            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else:  # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient
        elif self.config.distill_type == "pos-rs-kd":
            # RS-KD over POSITIONS: sample K% positions by distribution q(i)
            # Default q(i) ∝ H_i^alpha; with scores enabled, q(i) ∝ score_i^alpha
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]

            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            score_ctx = None
            kl_pos = None
            if score_enabled:
                kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)  # [B, L-1]
                score_ctx = self._prepare_score_context(ent_raw, kl_pos, s_log_probs, valid_next, input_ids)

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

                if score_enabled:
                    combined = self._build_score_vector(score_ctx, i, valid_next_i)
                    if combined is None:
                        continue
                    base = combined[valid_next_i].float()
                else:
                    base = ent_raw[i][valid_next_i].float()

                base = torch.clamp(base - base.min(), min=1e-8)  # strictly positive
                if alpha == 0.0:
                    q_un = torch.ones_like(base)
                else:
                    q_un = base.pow(alpha)

                q_un_sum = q_un.sum()
                if q_un_sum <= 0:
                    q = torch.full_like(q_un, 1.0 / q_un.numel())
                else:
                    q = q_un / q_un_sum
                    q = torch.clamp(q, min=q_floor)
                    q = q / q.sum()

                k_count = max(1, int(valid_count * pct))
                sel_rel = torch.multinomial(q, num_samples=k_count, replacement=False)  # [k]

                valid_idx = torch.where(valid_next_i)[0]
                selected_abs = valid_idx[sel_rel]  # [k]

                idx_t = selected_abs.to(t_log_probs.device)
                kl_sel = self._kl_loss(
                    t_log_probs[i, idx_t, :].to(self.student_device),
                    s_log_probs[i, selected_abs, :]
                )  # [k]

                q_sel = q[sel_rel]  # [k]
                w = 1.0 / torch.clamp(q_sel, min=q_floor)
                w = w / w.sum()
                kd_terms.append((kl_sel * w).sum())

            if kd_terms:
                kd_loss = torch.stack(kd_terms).mean()
            else:  # skip the batch if nothing selected
                kd_loss = t_pred.sum() * 0.0  # zero loss with gradient
        elif self.config.distill_type == "linucb":
            if self.bandit_manager is None:
                raise RuntimeError("LinUCB bandit is not initialized.")
            ent_raw = self._entropy_for_selection(input_ids, t_pred)
            t_log_probs_T1 = torch.log_softmax(t_pred, dim=-1)
            s_log_probs_T1 = torch.log_softmax(s_pred, dim=-1)
            kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)

            targets = input_ids[:, 1:].to(self.student_device)
            targets = targets.masked_fill(~valid_next, 0)
            # Teacher/student CE and KL are the contextual features consumed by the bandit.
            teacher_ce = (-t_log_probs_T1.to(self.student_device).gather(-1, targets.unsqueeze(-1)).squeeze(-1)).detach()
            student_ce = (-s_log_probs_T1.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).detach()

            kd_terms, metrics = self.bandit_manager.select_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ent_raw=ent_raw.detach(),
                teacher_ce=teacher_ce,
                student_ce=student_ce,
                kl_pos=kl_pos.detach(),
                valid_next=valid_next,
                temperature=temperature,
            )
            kd_loss = torch.stack(kd_terms).mean() if kd_terms else t_pred.sum() * 0.0
            if metrics:
                extra = metrics

        return kd_loss, extra

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

        # Unified KD temperature (applies to both teacher and student log-softmax)
        T = float(getattr(self.config, "kd_temperature", 1.0))
        T2 = T * T

        # --- student forward (always) ---
        s_logits = self.student(input_ids_s, attention_mask=attn_mask_s).logits
        s_logits = self._sanitize_logits(s_logits, "student")

        # Align to next-token prediction
        s_pred = s_logits[:, :-1, :]  # [B, L-1, V]
        valid_next = attn_mask_s[:, 1:].bool()  # [B, L-1]
        # Compute student log-probs lazily; skip full-vocab softmax if we use cached elimination path
        do_elim_softmax = bool(getattr(self.config, "eliminate_softmax", False))
        s_log_probs = None  # set to log_softmax on-demand when needed (only non-elimination paths)

        # Decide path: cached RS-KD with softmax elimination vs. fallback
        cached_items = self._lookup_cache_batch(input_ids) if bool(getattr(self.config, "offline_cache", False)) else None
        do_elim = bool(getattr(self.config, "eliminate_softmax", False)) and (cached_items is not None)
        # Whether we will use the cached RS-KD vocabulary estimator path at all
        use_vocab_rs_kd = bool(getattr(self.config, "offline_cache", False))

        # Only run teacher forward if we are NOT in elimination mode or cache is missing
        t_pred = t_log_probs = None
        if not do_elim:
            input_ids_t = input_ids.to(self.teacher_device)
            attn_t = attn_mask.to(self.teacher_device)
            with torch.no_grad():
                t_logits = self.teacher(input_ids_t, attention_mask=attn_t, output_hidden_states=False).logits
                t_logits = self._sanitize_logits(t_logits, "teacher")  # [B, L, V]
            t_pred = t_logits[:, :-1, :]
            t_log_probs = torch.log_softmax(t_pred / T, dim=-1)
            if not self._printed_cache_info:
                if getattr(self.config, "offline_cache", False) and cached_items is None:
                    print("[logits-cache] Cache miss or not available → running online teacher forward.")
                else:
                    print("[logits-cache] Running online teacher forward (elimination off).")
                self._printed_cache_info = True
        else:
            if not self._printed_cache_info:
                print("[logits-cache] Softmax elimination active with cache → skipping online teacher forward.")
                self._printed_cache_info = True

        # --- KD loss ---
        # Compute per-position base KD according to mode (positions subset and weights),
        # then replace vocab-sum with RS-KD estimator when enabled.
        extra_metrics: Optional[Dict[str, float]] = None
        if not use_vocab_rs_kd:
            if s_log_probs is None: # compute softmax if cache not rs-kd
                s_log_probs = torch.log_softmax(s_pred / T, dim=-1)
            kd_loss, kd_extra = self._compute_kd_loss(
                t_pred,
                t_log_probs,
                s_pred,
                s_log_probs,
                valid_next,
                input_ids,
                attn_mask,
                T,
            )
            if kd_extra:
                extra_metrics = kd_extra
        else:
            # Replace vocab-sum with RS-KD estimator. Two paths:
            # 1) cached_items available: decode precomputed per-position samples (keep existing per-position loop)
            # 2) no cache: build samples online in a fully batched, vectorized way across all valid positions
            if cached_items is not None:
                # ===== Flatten valid positions across the whole batch =====
                V = int(s_pred.size(-1))
                M_neg = int(getattr(self.config, "sampled_softmax_negatives", 1024))

                valid_mask = valid_next  # [B, L-1] (on student_device)
                batch_idx, pos_idx = torch.nonzero(valid_mask, as_tuple=True)  # [P_total], [P_total]
                P_total = int(batch_idx.numel())

                if P_total == 0:
                    kd_loss = s_pred.sum() * 0.0
                    ce_loss_override = None
                else:
                    # ---- shared negatives for the whole batch ----
                    ids_M_shared = torch.randint(
                        low=0, high=max(1, V), size=(M_neg,), device=self.student_device, dtype=torch.long
                    )
                    log_qM_base = -math.log(max(1, V))

                    # Student rows once for all selected positions: [P_total, V]
                    s_rows = s_pred[batch_idx, pos_idx]  # [P_total, V]
                    # Gather negatives once: [P_total, M_neg]
                    if M_neg > 0:
                        z_M_all = s_rows[:, ids_M_shared] / T
                    else:
                        z_M_all = s_rows.new_zeros((P_total, 0))

                    # ---- Prepare packed cache per batch example, track U per example ----
                    from .offline_cache import decode_ids_probs_from_block
                    B = s_pred.size(0)
                    packed_by_b, U_by_b, sen_by_b = [], [], []
                    for b in range(B):
                        rs = cached_items[b]["rs"]
                        packed_by_b.append(torch.as_tensor(rs["packed"], device=self.student_device, dtype=torch.uint8))
                        U_by_b.append(int(rs["U"]))
                        sen_by_b.append(int(rs["sentinel_id"]))

                    # Pad to U_max to allow a single dense gather
                    U_max = max(U_by_b) if len(U_by_b) > 0 else 0
                    ids_U = torch.zeros((P_total, U_max), dtype=torch.long, device=self.student_device)
                    probs_U = torch.zeros((P_total, U_max), dtype=torch.float32, device=self.student_device)

                    # Tiny decode loop (only copying into dense buffers; all heavy math is vectorized)
                    for r in range(P_total):
                        b = int(batch_idx[r].item())
                        p = int(pos_idx[r].item())
                        U = U_by_b[b]
                        sentinel = sen_by_b[b]
                        packed = packed_by_b[b]
                        if U == 0:
                            continue
                        block = packed[p * U * 3:(p + 1) * U * 3]
                        ids_r, probs_r = decode_ids_probs_from_block(block, U, sentinel)
                        u = ids_r.numel()
                        if u > 0:
                            ids_U[r, :u] = ids_r.to(self.student_device)
                            probs_U[r, :u] = probs_r.to(self.student_device)

                    # --- Proxies for score selection / bandit on elimination path (no full softmax) ---
                    # We will compute per-position:
                    #  - s_logp_on_U via importance-corrected sampled softmax over U ∪ M
                    #  - kd_pos_proxy_rows = cross-entropy over U w.r.t cached q_U (for ranking/loss)
                    #  - ce_pos_proxy_rows = sampled CE using gold y and shared negatives M
                    sum_qU = probs_U.sum(dim=1, keepdim=True)                  # [P,1]
                    denom = (sum_qU + (M_neg / max(1, V))).clamp_min(1e-12)    # [P,1]
                    denom_log = denom.log()

                    log_qU = probs_U.clamp_min(1e-12).log() - denom_log        # [P,U]
                    log_qM = (-math.log(max(1, V)) - denom_log)                # [P,1]

                    z_U = torch.gather(s_rows, 1, ids_U) / T                   # [P,U]
                    zcorr_U = z_U - log_qU                                     # [P,U]
                    zcorr_M = (z_M_all if M_neg > 0 else z_U[:, :0]) - log_qM  # [P,M]

                    logZ = torch.logsumexp(torch.cat([zcorr_U, zcorr_M], dim=1), dim=1, keepdim=True)  # [P,1]
                    s_logp_on_U = zcorr_U - logZ                                # [P,U]

                    kd_pos_proxy_rows = -(probs_U * s_logp_on_U).sum(dim=1)     # [P]

                    y_rows = input_ids_s[batch_idx, pos_idx + 1]                # [P]
                    z_y = s_rows.gather(1, y_rows.view(-1, 1)).squeeze(1) / T   # [P]
                    z_M_ce = (z_M_all - (math.log(max(1, V)))) if M_neg > 0 else z_y.view(-1, 1)[:, :0]
                    z_all = torch.cat([z_y.view(-1, 1), z_M_ce], dim=1)         # [P, 1+M]
                    logZ_ce = torch.logsumexp(z_all, dim=1)                     # [P]
                    ce_pos_proxy_rows = -(z_y - logZ_ce)                        # [P]

                    # Scatter proxies back to [B, L-1]
                    Bsz, Lm1 = valid_next.size()
                    kd_pos_proxy = torch.zeros((Bsz, Lm1), device=self.student_device, dtype=s_rows.dtype)
                    ce_pos_proxy = torch.zeros((Bsz, Lm1), device=self.student_device, dtype=s_rows.dtype)
                    kd_pos_proxy[batch_idx, pos_idx] = kd_pos_proxy_rows
                    ce_pos_proxy[batch_idx, pos_idx] = ce_pos_proxy_rows

                    if not do_elim:
                        # ---------- EKD over cached subset U (vectorized over all positions) ----------
                        if s_log_probs is None:
                            s_log_probs = torch.log_softmax(s_pred / T, dim=-1)
                        s_rows_logp = s_log_probs[batch_idx, pos_idx]          # [P_total, V]
                        s_logp_on_U = torch.gather(s_rows_logp, 1, ids_U)      # [P_total, U_max]
                        kd_loss = -(probs_U * s_logp_on_U).sum(dim=1).mean()
                        ce_loss_override = None
                    else:
                        # ---------- Elimination path with selection-aware proxies ----------
                        distill = getattr(self.config, "distill_type", "vanilla")
                        score_enabled = bool(getattr(self.config, "score_token_selection", False))

                        # Default: keep all valid rows
                        keep_mask = valid_next.clone()

                        # Build score context only if needed and supported types
                        score_ctx = None
                        if score_enabled and distill in {"top-k-tok", "bucket", "random"}:
                            # entropy from cache (t_pred=None so _entropy_for_selection will read cache if available)
                            ent_for_score = self._entropy_for_selection(input_ids, t_pred=None).to(self.student_device)
                            score_ctx = self._prepare_score_context(
                                ent_raw=ent_for_score,
                                kl_pos=kd_pos_proxy,
                                s_log_probs=None,
                                valid_next=valid_next,
                                input_ids=input_ids,
                                student_ce_override=ce_pos_proxy
                            )

                        if distill == "top-k-tok":
                            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
                            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
                            for i in range(valid_next.size(0)):
                                mask_i = valid_next[i]
                                n_valid = int(mask_i.sum().item())
                                if n_valid < 3:
                                    continue
                                k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                                valid_idx_i = torch.where(mask_i)[0]
                                if score_enabled and score_ctx is not None:
                                    combined = self._build_score_vector(score_ctx, i, mask_i)
                                    if combined is None:
                                        continue
                                    scores = combined[mask_i]
                                else:
                                    # fallback: use cached entropy
                                    ent_i = self._entropy_for_selection(input_ids, t_pred=None)[i]
                                    scores = ent_i[mask_i]
                                if scores.numel() == 0:
                                    continue
                                _, top_rel = torch.topk(scores, k=k, largest=True, sorted=False)
                                sel_abs = valid_idx_i[top_rel]
                                keep_mask[i, sel_abs] = True

                        elif distill == "bucket":
                            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
                            ent_for_bucket = None
                            if not score_enabled:
                                ent_for_bucket = self._entropy_for_selection(input_ids, t_pred=None).to(self.student_device)
                            for i in range(valid_next.size(0)):
                                mask_i = valid_next[i]
                                if mask_i.sum() < 3:
                                    continue
                                if score_enabled and score_ctx is not None:
                                    combined = self._build_score_vector(score_ctx, i, mask_i)
                                    if combined is None:
                                        continue
                                    vec = combined[mask_i].float()
                                else:
                                    vec = ent_for_bucket[i][mask_i].float()
                                    
                                low = torch.quantile(vec, self.config.bucket_lower_percent / 100.0)
                                high = torch.quantile(vec, self.config.bucket_upper_percent / 100.0)
                                rel = torch.where(mask_i)[0]
                                sel = (vec >= low) & (vec <= high)
                                if sel.any():
                                    keep_mask[i, rel[sel]] = True

                        elif distill == "random":
                            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
                            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
                            for i in range(valid_next.size(0)):
                                mask_i = valid_next[i]
                                n_valid = int(mask_i.sum().item())
                                if n_valid < 2:
                                    continue
                                k = max(1, int(n_valid * pct))
                                valid_idx_i = torch.where(mask_i)[0]
                                if score_enabled and score_ctx is not None:
                                    combined = self._build_score_vector(score_ctx, i, mask_i)
                                    if combined is None:
                                        continue
                                    scores = combined[mask_i].float()
                                    s = scores - scores.min()
                                    s = torch.clamp(s, min=1e-8)
                                    probs = s / s.sum()
                                    rel = torch.multinomial(probs, num_samples=k, replacement=False)
                                    sel_abs = valid_idx_i[rel]
                                else:
                                    perm = torch.randperm(valid_idx_i.numel(), device=self.student_device)
                                    sel_abs = valid_idx_i[perm[:k]]
                                keep_mask[i, sel_abs] = True

                        elif distill == "linucb":
                            if self.bandit_manager is None:
                                raise RuntimeError("LinUCB bandit is not initialized.")
                            # Build entropy from cache
                            ent_for_bandit = self._entropy_for_selection(input_ids, t_pred=None)
                            # Use proxies for all features to avoid teacher/full softmax
                            kd_terms, metrics = self.bandit_manager.select_tokens(
                                input_ids=input_ids,
                                attention_mask=attn_mask,
                                ent_raw=ent_for_bandit.detach(),
                                teacher_ce=ce_pos_proxy.detach(),
                                student_ce=ce_pos_proxy.detach(),
                                kl_pos=kd_pos_proxy.detach(),
                                valid_next=valid_next,
                                temperature=T,
                            )
                            kd_loss = torch.stack(kd_terms).mean() if kd_terms else s_pred.sum() * 0.0
                            extra_metrics = metrics or None
                            # Sampled CE proxy over all rows (no indices from bandit selection here)
                            ce_loss_override = ce_pos_proxy_rows.mean() if self.config.enable_ce else None
                            # Short-circuit to avoid computing below using keep_mask
                            pass
                        elif distill == "pos-rs-kd":
                            alpha = float(getattr(self.config, "rs_alpha", 1.0))
                            q_floor = float(getattr(self.config, "rs_floor", 1e-6))
                            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
                            kd_terms_local = []

                            # Optional: score-based q(i)
                            use_score = bool(getattr(self.config, "score_token_selection", False))
                            if use_score:
                                ent_for_score = self._entropy_for_selection(input_ids, t_pred=None).to(self.student_device)
                                score_ctx = self._prepare_score_context(
                                    ent_raw=ent_for_score,
                                    kl_pos=kd_pos_proxy,
                                    s_log_probs=None,
                                    valid_next=valid_next,
                                    input_ids=input_ids,
                                    student_ce_override=ce_pos_proxy,
                                )
                            for i in range(valid_next.size(0)):
                                mask_i = valid_next[i]
                                n_valid = int(mask_i.sum().item())
                                if n_valid < 3:
                                    continue

                                if use_score:
                                    combined = self._build_score_vector(score_ctx, i, mask_i)
                                    if combined is None:
                                        continue
                                    base = combined[mask_i].float()
                                else:
                                    ent_i = self._entropy_for_selection(input_ids, t_pred=None)[i]
                                    base = ent_i[mask_i].float()

                                base = torch.clamp(base - base.min(), min=1e-8)
                                q_un = torch.ones_like(base) if alpha == 0.0 else base.pow(alpha)
                                if q_un.sum() <= 0:
                                    q = torch.full_like(q_un, 1.0 / q_un.numel())
                                else:
                                    q = q_un / q_un.sum()
                                    q = torch.clamp(q, min=q_floor)
                                    q = q / q.sum()

                                k_count = max(1, int(n_valid * pct))
                                rel_sel = torch.multinomial(q, num_samples=k_count, replacement=False)  # [k]
                                valid_idx_i = torch.where(mask_i)[0]
                                abs_sel = valid_idx_i[rel_sel]

                                # KD over selected rows with importance correction
                                q_sel = q[rel_sel]                    # [k]
                                w = 1.0 / torch.clamp(q_sel, min=q_floor)
                                w = w / w.sum()

                                kd_terms_local.append((kd_pos_proxy[i, abs_sel] * w).sum())
                            kd_loss = (
                                torch.stack(kd_terms_local).mean()
                                if kd_terms_local else s_pred.sum() * 0.0
                            )

                            # CE on ALL valid rows to mirror top-k-tok (or keep it off if you prefer)
                            ce_loss_override = ce_pos_proxy_rows.mean() if self.config.enable_ce else None

                        # If not the bandit short-circuit above, compute losses using keep_mask
                        if distill != "linucb":
                            # Map selection to flattened rows
                            # If not the bandit short-circuit above, compute losses using keep_mask
                            keep_rows = keep_mask[batch_idx, pos_idx] if P_total > 0 else torch.zeros(
                                0, dtype=torch.bool, device=self.student_device
                            )

                            if distill == "top-k-tok":
                                if keep_rows.any():
                                    # KD: only on selected rows (same as before)
                                    kd_loss = kd_pos_proxy_rows[keep_rows].mean()
                                    # CE: on ALL valid rows to match old top-k behavior
                                    if self.config.enable_ce:
                                        ce_loss_override = ce_pos_proxy_rows.mean()
                                    else:
                                        ce_loss_override = None
                                else:
                                    kd_loss = s_pred.sum() * 0.0
                                    ce_loss_override = None
                            else:
                                # bucket/random: keep previous behavior (KD and CE both on the selected rows)
                                if keep_rows.any():
                                    kd_loss = kd_pos_proxy_rows[keep_rows].mean()
                                    ce_loss_override = ce_pos_proxy_rows[keep_rows].mean() if self.config.enable_ce else None
                                else:
                                    kd_loss = s_pred.sum() * 0.0
                                    ce_loss_override = None
            else:
                # Vectorized online RS-KD over vocabulary: flatten all valid positions across the batch
                assert t_log_probs is not None and t_pred is not None, "Teacher logits required for online RS-KD when cache missing"
                # Move necessary teacher tensors to the student device for gathers with student log-probs
                t_logp_Tkd = t_log_probs.to(self.student_device)           # [B, L-1, V]
                t_logp_T1 = torch.log_softmax(t_pred.to(self.student_device) / 1.0, dim=-1)  # [B, L-1, V]
                p_Tkd = t_logp_Tkd.exp()                                   # [B, L-1, V]

                mask = valid_next  # [B, L-1] bool
                P_total = int(mask.sum().item())
                if P_total == 0:
                    kd_loss = s_pred.sum() * 0.0
                else:
                    # Gather all valid rows once: [P_total, V]
                    p_rows = p_Tkd[mask]
                    if s_log_probs is None:
                        s_log_probs = torch.log_softmax(s_pred / T, dim=-1)
                    s_rows = s_log_probs[mask]
                    t1_rows = t_logp_T1[mask]

                    beta = float(getattr(self.config, "rs_vocab_beta", 1.0))
                    S_vocab = int(getattr(self.config, "rs_vocab_samples", 64))

                    # Build proposal q per-row: q ∝ p_Tkd^beta (with clamp for safety), normalize row-wise
                    q_un = p_rows.clamp_min(1e-12)
                    if beta != 1.0:
                        q_un = q_un.pow(beta)
                    q = q_un / q_un.sum(dim=-1, keepdim=True)  # [P_total, V]

                    # Single batched sampling for all rows
                    S_eff = min(S_vocab, q.size(-1))
                    idx_all = torch.multinomial(q, num_samples=S_eff, replacement=False)  # [P_total, S_eff]

                    # Gather per-row selections for teacher (T=1), proposal q, and student log-probs
                    t_logp1_sel = torch.gather(t1_rows, 1, idx_all)  # [P_total, S_eff]
                    q_sel = torch.gather(q, 1, idx_all)              # [P_total, S_eff]
                    s_logp_sel = torch.gather(s_rows, 1, idx_all)    # [P_total, S_eff]

                    # Vectorized self-normalized importance-weighted KL estimator across rows
                    gamma = 1.0 / max(1e-12, T)
                    p1 = torch.exp(t_logp1_sel)
                    pT_un = p1.pow(gamma)
                    w = pT_un / q_sel.clamp_min(1e-12)
                    w = w / w.sum(dim=1, keepdim=True)
                    logpT_approx = torch.log(pT_un) - torch.log(pT_un.sum(dim=1, keepdim=True).clamp_min(1e-12))
                    diff = logpT_approx - s_logp_sel
                    loss_rows = (w * diff).sum(dim=1)  # [P_total]
                    kd_loss = loss_rows.mean()
        
        # Temperature factor (keep gradients comparable across T, as in standard distillation)
        kd_loss = kd_loss * T2

        # --- CE loss (only valid targets) ---
        if self.config.enable_ce:
            # if sampled CE was computed in the cached path with eliminate_softmax=True, use it
            if 'ce_loss_override' in locals() and ce_loss_override is not None:
                ce_loss = ce_loss_override
            else:
                targets = input_ids_s[:, 1:]  # [B, L-1]
                targets = targets.masked_fill(~valid_next, -100)
                if s_log_probs is None:
                    s_log_probs = torch.log_softmax(s_pred / T, dim=-1)
                V = s_log_probs.size(-1)
                flat_log_probs = s_log_probs.reshape(-1, V)
                flat_targets = targets.reshape(-1)
                ce_loss = F.nll_loss(flat_log_probs, flat_targets, ignore_index=-100, reduction="mean")
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
            return zero + zero, 0.0, 0.0, None

        return total, kd_loss.item(), ce_loss.item(), extra_metrics

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
        
    def train(self, epochs: int = 1, log_every: int = 100):
        """Run distillation training for specified number of epochs."""
        overall_train_start = time.time()
        # make the offline pass once, if requested (no hidden side effects)
        if getattr(self.config, "offline_cache", False):
            if self.cache is None:
                self.cache = init_offline_cache_for_trainer(
                    getattr(self.config, "offline_cache_dir", None),
                    self.compute_cache_signature()
                )
            # Build cache with a single-worker DataLoader to avoid multiprocessing/fork hangs
            # after CUDA initialization on HPC nodes.
            try:
                from torch.utils.data import DataLoader
                dl_for_cache = DataLoader(
                    self.dataloader.dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    collate_fn=self.dataloader.collate_fn,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                )
            except Exception:
                # Fallback: use the existing dataloader
                dl_for_cache = self.dataloader

            self.cache = build_offline_cache_if_needed(
                cache=self.cache,
                teacher=self.teacher, tok=self.tok, dataloader=dl_for_cache,
                config=self.config, teacher_device=self.teacher_device,
                sanitize_logits_fn=self._sanitize_logits,
            )

        if self.bandit_manager is not None:
            # Guard against stale pending batches when resuming training or restarting loops.
            self.bandit_manager.reset()
        # Prepare KD temperature annealing schedule (in units of optimizer updates)
        updates_per_epoch = math.ceil(len(self.dataloader) / max(1, self.config.gradient_accumulation_steps))
        total_updates = updates_per_epoch * max(1, epochs)

        def kd_T_at(update_idx: int) -> float:
            T0 = float(getattr(self.config, "kd_temperature_start", 2.0))
            T1 = float(getattr(self.config, "kd_temperature_end", 1.0))
            hold_frac = float(getattr(self.config, "kd_hold_frac", 0.6))
            hold_u = int(hold_frac * total_updates)
            if update_idx <= hold_u:
                return T0
            tail = max(1, (total_updates - hold_u))
            pos = (update_idx - hold_u) / tail
            return T0 + (T1 - T0) * pos

        for epoch in range(epochs):
            step_start = time.time()
            running = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
            bandit_running: Dict[str, float] = {}
            bandit_steps = 0
            last_reward_metrics: Optional[Dict[str, float]] = None
            self.opt.zero_grad(set_to_none=True)  # Initialize gradients
            
            for step, batch in enumerate(self.dataloader):
                loss, kl_val, ce_val, bandit_metrics = self._forward_batch(batch)
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Only update weights after accumulation steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.global_step += 1
                    # Update KD temperature per schedule if enabled
                    if bool(getattr(self.config, "anneal_kd_temperature", False)):
                        self.config.kd_temperature = kd_T_at(self.global_step)
                        # Log the current KD temperature to visualize the annealing schedule
                        if self.logger:
                            self.logger.log_scalar("train/kd_temperature", float(self.config.kd_temperature), self.global_step)
                    reward_metrics = None
                    if self.bandit_manager is not None:
                        # Consume queued actions collected during forward passes and update the bandit.
                        reward_metrics = self.bandit_manager.process_rewards(self.student, self.teacher)
                    if reward_metrics:
                        last_reward_metrics = reward_metrics
                        if self.logger:
                            self.logger.log(reward_metrics, self.global_step)
                    
                    # Save checkpoint if needed
                    if (self.config.checkpoint_steps > 0 and 
                        self.global_step % self.config.checkpoint_steps == 0):
                        self.save_checkpoint(epoch, step)

                # logging
                running["loss"] += loss.item() * self.config.gradient_accumulation_steps  # Unscale for logging
                running["kl"] += kl_val
                running["ce"] += ce_val
                if bandit_metrics:
                    # These metrics describe the current batch's bandit decisions (e.g., token counts).
                    for key, value in bandit_metrics.items():
                        bandit_running[key] = bandit_running.get(key, 0.0) + value
                    bandit_steps += 1
                
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
                    if bandit_metrics:
                        log_metrics.update(bandit_metrics)
                    self.logger.log(log_metrics, self.global_step)
                    
                if (step + 1) % log_every == 0:
                    n = log_every
                    avg_loss = running['loss'] / n
                    avg_kl = running['kl'] / n
                    avg_ce = running['ce'] / n
                    avg_bandit: Dict[str, float] = {}
                    if bandit_steps > 0:
                        avg_bandit = {k: v / bandit_steps for k, v in bandit_running.items()}
                    
                    elapsed = time.time() - step_start 
                    step_start = time.time()
                    bandit_str = ""
                    if avg_bandit:
                        sel = avg_bandit.get("bandit/selected_tokens", 0.0)
                        overlap = avg_bandit.get("bandit/overlap_selected", 0.0)
                        bandit_str = f" | bandit_sel={sel:.2f} overlap={overlap:.2f}"
                    reward_str = ""
                    if last_reward_metrics:
                        avg_reward = last_reward_metrics.get("bandit/avg_reward", 0.0)
                        reward_str = f" | bandit_reward={avg_reward:.4f}"
                    print(
                        f"ep{epoch + 1} step{step + 1} | "
                        f"loss={avg_loss:.4f} kl={avg_kl:.4f} ce={avg_ce:.4f} "
                        f"| global_step={self.global_step}{bandit_str}{reward_str} | {elapsed:.2f}s total, {elapsed/log_every:.2f}s/step"
                    )
                    
                    # Log averages using new combined logger or legacy loggers
                    avg_metrics = {
                        "train/avg_loss": avg_loss,
                        "train/avg_kl_loss": avg_kl,
                        "train/avg_ce_loss": avg_ce,
                        "train/elapsed_time": elapsed,
                        "train/steps_per_second": log_every / elapsed
                    }
                    if avg_bandit:
                        avg_metrics.update(avg_bandit)
                    if last_reward_metrics:
                        avg_metrics.update(last_reward_metrics)
                    
                    # Log averages
                    if self.logger:
                        self.logger.log(avg_metrics, self.global_step)
                        self.logger.flush()
                        
                    running = {k: 0.0 for k in running}
                    bandit_running = {}
                    bandit_steps = 0
        
        # Final checkpoint and cleanup at the end
        if self.config.checkpoint_steps > 0:
            print("Training completed. Performing final cleanup of old checkpoints...")
            self._cleanup_old_checkpoints()

        overall_train_elapsed = time.time() - overall_train_start
        print(f"[distill] Total training duration: {overall_train_elapsed:.2f}s for {epochs} epoch(s)")
