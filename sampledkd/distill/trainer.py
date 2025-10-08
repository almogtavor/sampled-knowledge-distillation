from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import device
from torch.optim import AdamW

from ..config import TrainingConfig, TrainingMetrics
from ..training.offline_cache import (
    build_offline_cache_if_needed,
    init_offline_cache_for_trainer,
    decode_ids_probs_from_block,
)
from ._mixins.amp_oom import AmpOomMixin
from ._mixins.bandit import BanditMixin
from ._mixins.cache import CacheMixin
from ._mixins.checkpoint import CheckpointMixin
from ._mixins.entropy import EntropyMixin
from ._mixins.gls import GLSMixin
from ._mixins.kd_core import KDCoreMixin
from ._mixins.selection_scoring import SelectionScoringMixin
from ._mixins.rs_vocab import kl_from_vocab_samples
from .ce_estimators import ce_is_estimator


class Distiller(
    AmpOomMixin,
    CheckpointMixin,
    CacheMixin,
    GLSMixin,
    SelectionScoringMixin,
    EntropyMixin,
    KDCoreMixin,
    BanditMixin,
):
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
        
        # ===== OOM Reduction: 8-bit optimizer to save ~2-3x memory =====
        try:
            from bitsandbytes.optim import Adam8bit
            self.opt = Adam8bit(self.student.parameters(), lr=config.lr)
            print("[OOM-opt] Using 8-bit Adam optimizer to reduce memory.")
        except Exception:
            self.opt = AdamW(self.student.parameters(), lr=config.lr)
            print("[OOM-opt] bitsandbytes not available, using standard AdamW.")
        
        self.teacher_device = teacher_device
        self.student_device = student_device
        self._printed_cache_info = False
        
        # ===== OOM Reduction: Enable gradient checkpointing on student =====
        if hasattr(self.student, "gradient_checkpointing_enable"):
            try:
                self.student.gradient_checkpointing_enable()
                print("[OOM-opt] Enabled gradient checkpointing on student model.")
            except Exception as e:
                print(f"[OOM-opt] Could not enable gradient checkpointing: {e}")
        
        # ===== OOM Reduction: Enable memory-efficient attention =====
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            print("[OOM-opt] Enabled memory-efficient SDPA backends.")
        except Exception:
            pass
        
        # Try Flash Attention 2 on student if supported
        if hasattr(self.student, "config") and hasattr(self.student.config, "use_flash_attention_2"):
            try:
                self.student.config.use_flash_attention_2 = True
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
            print(f"[OOM-opt] Using AMP with {self._amp_dtype} and GradScaler.")
        else:
            self._scaler = None
            print(f"[OOM-opt] Using AMP with {self._amp_dtype} (no scaler needed).")
        
        # Logging setup
        self.logger = logger
        self.global_step = 0

        # GLS ring buffer state (inactive unless enabled)
        self._gls_buf: Optional[torch.Tensor] = None
        self._gls_cap: int = int(getattr(self.config, "gls_queue_size", 30000))
        self._gls_count: int = 0
        self._gls_head: int = 0
        
        # Checkpointing
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # offline teacher logits cache: centralized initialization
        self.cache = None

        # Track once-per-run warnings
        self._warned_invalid_targets = False
        self._warned_invalid_logprob = False

        # LinUCB contextual bandit setup
        self._init_bandit_manager()

    def _forward_batch(self, batch):
        # Move inputs
        input_ids = batch["input_ids"] # [B, L]
        attn_mask = batch["attention_mask"]
        input_ids_s = input_ids.to(self.student_device)
        attn_mask_s = attn_mask.to(self.student_device)

        # Unified KD temperature (applies to both teacher and student log-softmax)
        T = float(getattr(self.config, "kd_temperature", 1.0))
        T2 = T * T

        # ===== OOM Reduction: AMP context for mixed precision =====
        from torch.cuda.amp import autocast
        amp_enabled = getattr(self, "_use_amp", False)
        amp_dtype = getattr(self, "_amp_dtype", torch.float32)
        
        # --- student forward (always) ---
        with autocast(enabled=amp_enabled, dtype=amp_dtype):
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
        distill_type = getattr(self.config, "distill_type", "vanilla")
        score_enabled_flag = bool(getattr(self.config, "score_token_selection", False))

        supports_cached_no_elim = (
            use_vocab_rs_kd
            and cached_items is not None
            and not do_elim
            and distill_type in {"vanilla", "top-k-tok", "bucket", "random"}
        )

        # Only run teacher forward if we are NOT in elimination mode or cache is missing/unsupported
        t_pred = t_log_probs = None
        if supports_cached_no_elim and not self._printed_cache_info:
            print("[logits-cache] Using offline cache without elimination → computing KD from cache.", flush=True)
            self._printed_cache_info = True

        if not do_elim and not supports_cached_no_elim:
            input_ids_t = input_ids.to(self.teacher_device)
            attn_t = attn_mask.to(self.teacher_device)
            with torch.no_grad():
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
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
                    ce_pos_proxy_rows = ce_is_estimator(
                        s_rows=s_rows,
                        ids_U=ids_U,
                        probs_U=probs_U,
                        ids_M_shared=ids_M_shared,
                        M_neg=M_neg,
                        y_rows=y_rows,
                    )

                    # Scatter proxies back to [B, L-1]
                    Bsz, Lm1 = valid_next.size()
                    kd_pos_proxy = torch.zeros((Bsz, Lm1), device=self.student_device, dtype=s_rows.dtype)
                    ce_pos_proxy = torch.zeros((Bsz, Lm1), device=self.student_device, dtype=s_rows.dtype)
                    kd_pos_proxy[batch_idx, pos_idx] = kd_pos_proxy_rows
                    ce_pos_proxy[batch_idx, pos_idx] = ce_pos_proxy_rows

                    if not do_elim:
                        if s_log_probs is None:
                            s_log_probs = torch.log_softmax(s_pred / T, dim=-1)
                        s_rows_logp = s_log_probs[batch_idx, pos_idx]          # [P_total, V]
                        s_logp_on_U_exact = torch.gather(s_rows_logp, 1, ids_U)  # [P_total, U_max]
                        kd_rows_exact = -(probs_U * s_logp_on_U_exact).sum(dim=1)
                        ce_loss_override = None

                        if not supports_cached_no_elim:
                            kd_loss = kd_rows_exact.mean() if kd_rows_exact.numel() > 0 else s_pred.sum() * 0.0
                        else:
                            distill = distill_type
                            score_enabled = score_enabled_flag
                            score_ctx = None
                            ent_cached = None
                            if distill in {"top-k-tok", "bucket", "random"}:
                                ent_cached = self._entropy_for_selection(input_ids, t_pred=None).to(self.student_device)
                                if score_enabled:
                                    score_ctx = self._prepare_score_context(
                                        ent_raw=ent_cached,
                                        kl_pos=kd_pos_proxy,
                                        s_log_probs=None,
                                        valid_next=valid_next,
                                        input_ids=input_ids,
                                        student_ce_override=ce_pos_proxy,
                                    )

                            if distill == "vanilla":
                                kd_loss = kd_rows_exact.mean() if kd_rows_exact.numel() > 0 else s_pred.sum() * 0.0
                            elif distill == "top-k-tok":
                                pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
                                if score_enabled and score_ctx is not None:
                                    stat = torch.full_like(ent_cached, float('-inf'))
                                    for i in range(valid_next.size(0)):
                                        mask_i = valid_next[i]
                                        combined = self._build_score_vector(score_ctx, i, mask_i)
                                        if combined is not None:
                                            stat[i] = combined
                                    stat = stat.masked_fill(~valid_next, float('-inf'))
                                else:
                                    stat = ent_cached.masked_fill(~valid_next, float('-inf'))

                                use_gls = bool(getattr(self.config, "gls_enabled", False))
                                keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
                                sel_topk_count = 0
                                sel_gls_count = 0
                                if not use_gls:
                                    for i in range(valid_next.size(0)):
                                        mask_i = valid_next[i]
                                        n_valid = int(mask_i.sum().item())
                                        if n_valid < 3:
                                            continue
                                        k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                                        sel_topk_count += int(k)
                                        valid_idx_i = torch.where(mask_i)[0]
                                        scores = stat[i][mask_i].float()
                                        if scores.numel() == 0:
                                            continue
                                        _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
                                        sel_abs = valid_idx_i[rel]
                                        keep_mask[i, sel_abs] = True
                                else:
                                    self._gls_init_if_needed()
                                    thr = self._gls_threshold(top_percent=self.config.k_percent)
                                    if thr is None:
                                        for i in range(valid_next.size(0)):
                                            mask_i = valid_next[i]
                                            n_valid = int(mask_i.sum().item())
                                            if n_valid < 3:
                                                continue
                                            k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                                            sel_topk_count += int(k)
                                            valid_idx_i = torch.where(mask_i)[0]
                                            scores = stat[i][mask_i].float()
                                            if scores.numel() == 0:
                                                continue
                                            _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
                                            sel_abs = valid_idx_i[rel]
                                            keep_mask[i, sel_abs] = True
                                    else:
                                        keep_mask = (stat >= thr) & valid_next
                                        sel_gls_count = int(keep_mask.sum().item())
                                    vals = stat[valid_next].detach().float().to("cpu")
                                    vals = vals[torch.isfinite(vals)]
                                    self._gls_push(vals)
                                    if getattr(self.config, "gls_log_threshold", False) and ('thr' in locals()) and thr is not None and self.logger:
                                        self.logger.log_scalar("train/gls_threshold", float(thr), self.global_step)

                                if P_total > 0:
                                    selected_rows = keep_mask[batch_idx, pos_idx]
                                else:
                                    selected_rows = torch.zeros(0, dtype=torch.bool, device=self.student_device)
                                if selected_rows.any():
                                    kd_loss = kd_rows_exact[selected_rows].mean()
                                else:
                                    kd_loss = s_pred.sum() * 0.0

                                if self.logger:
                                    self.logger.log({
                                        "train/selected_tokens_topk": float(sel_topk_count),
                                        "train/selected_tokens_gls": float(sel_gls_count),
                                    }, self.global_step)

                            elif distill == "bucket":
                                keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
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
                                        vec = ent_cached[i][mask_i].float()

                                    low = torch.quantile(vec, self.config.bucket_lower_percent / 100.0)
                                    high = torch.quantile(vec, self.config.bucket_upper_percent / 100.0)
                                    rel = torch.where(mask_i)[0]
                                    sel = (vec >= low) & (vec <= high)
                                    if sel.any():
                                        keep_mask[i, rel[sel]] = True

                                if P_total > 0:
                                    selected_rows = keep_mask[batch_idx, pos_idx]
                                else:
                                    selected_rows = torch.zeros(0, dtype=torch.bool, device=self.student_device)
                                if selected_rows.any():
                                    kd_loss = kd_rows_exact[selected_rows].mean()
                                else:
                                    kd_loss = s_pred.sum() * 0.0

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
                                    else:
                                        perm = torch.randperm(valid_idx_i.numel(), device=self.student_device)
                                        rel = perm[:k]
                                    sel_abs = valid_idx_i[rel]
                                    keep_mask[i, sel_abs] = True

                                if P_total > 0:
                                    selected_rows = keep_mask[batch_idx, pos_idx]
                                else:
                                    selected_rows = torch.zeros(0, dtype=torch.bool, device=self.student_device)
                                if selected_rows.any():
                                    kd_loss = kd_rows_exact[selected_rows].mean()
                                else:
                                    kd_loss = s_pred.sum() * 0.0
                            else:
                                kd_loss = kd_rows_exact.mean() if kd_rows_exact.numel() > 0 else s_pred.sum() * 0.0
                    else:
                        # ---------- Elimination path with selection-aware proxies ----------
                        distill = distill_type
                        score_enabled = score_enabled_flag

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
                            # Build ranking stat: default cached entropy; use combined score if enabled
                            ent_cached = self._entropy_for_selection(input_ids, t_pred=None).to(self.student_device)
                            if score_enabled and score_ctx is not None:
                                stat_elim = torch.full_like(ent_cached, float('-inf'))
                                for i in range(valid_next.size(0)):
                                    mask_i = valid_next[i]
                                    combined = self._build_score_vector(score_ctx, i, mask_i)
                                    if combined is not None:
                                        stat_elim[i] = combined
                                stat_elim = stat_elim.masked_fill(~valid_next, float('-inf'))
                            else:
                                stat_elim = ent_cached.masked_fill(~valid_next, float('-inf'))

                            use_gls = bool(getattr(self.config, "gls_enabled", False))
                            sel_topk_count = 0
                            sel_gls_count = 0
                            if not use_gls:
                                # Original per-example top-k
                                pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
                                keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
                                for i in range(valid_next.size(0)):
                                    mask_i = valid_next[i]
                                    n_valid = int(mask_i.sum().item())
                                    if n_valid < 3:
                                        continue
                                    k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                                    sel_topk_count += int(k)
                                    valid_idx_i = torch.where(mask_i)[0]
                                    scores = stat_elim[i][mask_i].float()
                                    if scores.numel() == 0:
                                        continue
                                    _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
                                    sel_abs = valid_idx_i[rel]
                                    keep_mask[i, sel_abs] = True
                            else:
                                # GLS threshold with warm-up fallback
                                self._gls_init_if_needed()
                                thr = self._gls_threshold(top_percent=self.config.k_percent)
                                if thr is None:
                                    pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
                                    keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
                                    for i in range(valid_next.size(0)):
                                        mask_i = valid_next[i]
                                        n_valid = int(mask_i.sum().item())
                                        if n_valid < 3:
                                            continue
                                        k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                                        sel_topk_count += int(k)
                                        valid_idx_i = torch.where(mask_i)[0]
                                        scores = stat_elim[i][mask_i].float()
                                        if scores.numel() == 0:
                                            continue
                                        _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
                                        sel_abs = valid_idx_i[rel]
                                        keep_mask[i, sel_abs] = True
                                else:
                                    keep_mask = (stat_elim >= thr) & valid_next
                                    sel_gls_count = int(keep_mask.sum().item())
                                # After selection, push this batch's stats and optionally log threshold
                                vals = stat_elim[valid_next].detach().float().to("cpu")
                                vals = vals[torch.isfinite(vals)]
                                self._gls_push(vals)
                                if getattr(self.config, "gls_log_threshold", False) and ('thr' in locals()) and thr is not None and self.logger:
                                    self.logger.log_scalar("train/gls_threshold", float(thr), self.global_step)
                            # Log selection counters (per batch, elimination path)
                            if self.logger:
                                self.logger.log({
                                    "train/selected_tokens_topk": float(sel_topk_count),
                                    "train/selected_tokens_gls": float(sel_gls_count),
                                }, self.global_step)

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
                            # NEW: Build importance-weighted mask for KD loss
                            alpha = float(getattr(self.config, "rs_alpha", 1.0))
                            q_floor = float(getattr(self.config, "rs_floor", 1e-6))
                            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))

                            # Create importance-weighted mask [B, L-1]
                            weight_mask = torch.zeros_like(valid_next, dtype=torch.float32)

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
                                base = torch.clamp(base, min=1e-8)
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

                                k_count = max(1, int(n_valid * pct))
                                rel_sel = torch.multinomial(q, num_samples=k_count, replacement=False)  # [k]

                                valid_idx_i = torch.where(mask_i)[0]
                                abs_sel = valid_idx_i[rel_sel]

                                # Compute importance weights (unnormalized) and store in mask
                                q_sel = q[rel_sel]
                                w = 1.0 / torch.clamp(q_sel, min=q_floor)
                                weight_mask[i, abs_sel] = w

                            # Apply weighted mask to KD loss
                            # Weighted per-token batch mean using proxy KD and importance weights
                            kd_loss = (kd_pos_proxy * weight_mask).sum() / weight_mask.sum().clamp_min(1e-12)

                            # CE on ALL valid rows to match top-k-tok behavior
                            ce_loss_override = ce_pos_proxy_rows.mean() if self.config.enable_ce else None

                        # If not the bandit short-circuit above or pos-rs-kd, compute losses using keep_mask
                        if distill not in {"linucb", "pos-rs-kd"}:
                            # NEW: Apply mask to proxy KD loss (all positions computed, only selected contribute)
                            # Scatter keep_mask to full [B, L-1] if needed
                            if distill == "top-k-tok":
                                # KD: masked loss over selected positions
                                kd_loss = (kd_pos_proxy * keep_mask).sum() / keep_mask.sum().clamp_min(1)
                                # CE: on ALL valid positions (standard practice for top-k-tok)
                                ce_loss_override = ce_pos_proxy_rows.mean() if self.config.enable_ce else None
                            else:
                                # bucket/random: KD and CE both on selected positions
                                kd_loss = (kd_pos_proxy * keep_mask).sum() / keep_mask.sum().clamp_min(1)
                                # For CE on selected positions, we need to mask ce_pos_proxy
                                if self.config.enable_ce:
                                    keep_rows = keep_mask[batch_idx, pos_idx] if P_total > 0 else torch.zeros(
                                        0, dtype=torch.bool, device=self.student_device
                                    )
                                    ce_loss_override = ce_pos_proxy_rows[keep_rows].mean() if keep_rows.any() else s_pred.sum() * 0.0
                                else:
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
        if True:
            kd_loss = kd_loss * T2

        # --- CE loss (only valid targets) ---
        if self.config.enable_ce:
            # if sampled CE was computed in the cached path with eliminate_softmax=True, use it
            if 'ce_loss_override' in locals() and ce_loss_override is not None:
                ce_loss = ce_loss_override
            else:
                targets = input_ids_s[:, 1:]  # [B, L-1]
                # In the new mode, CE is on ALL valid tokens (standard masking)
                targets = targets.masked_fill(~valid_next, -100)
                # CE loss should always be at T=1 (standard practice in KD)
                s_log_probs_T1 = torch.log_softmax(s_pred, dim=-1)
                V = s_log_probs_T1.size(-1)
                flat_log_probs = s_log_probs_T1.reshape(-1, V)
                flat_targets = targets.reshape(-1).long().clone()

                ignore_mask = flat_targets == -100
                valid_range_mask = (flat_targets >= 0) & (flat_targets < V)
                invalid_range_mask = ~(ignore_mask | valid_range_mask)
                if invalid_range_mask.any():
                    bad_vals = flat_targets[invalid_range_mask].detach().to("cpu", non_blocking=True)
                    bad_count = int(bad_vals.numel())
                    flat_targets = flat_targets.masked_fill(invalid_range_mask, -100)
                    if bad_count > 0 and not self._warned_invalid_targets:
                        min_bad = int(bad_vals.min().item()) if bad_vals.numel() else 0
                        max_bad = int(bad_vals.max().item()) if bad_vals.numel() else 0
                        sample_vals = bad_vals.unique()
                        if sample_vals.numel() > 5:
                            sample_vals = sample_vals[:5]
                        sample_list = ", ".join(str(int(v.item())) for v in sample_vals)
                        print(
                            "[warn] CE targets out of range (count="
                            f"{bad_count}, min={min_bad}, max={max_bad}, sample=[{sample_list}])"
                            " → masking from loss.",
                            flush=True,
                        )
                        self._warned_invalid_targets = True

                # Mask rows that contain non-finite log-probs to avoid device-side asserts
                finite_row_mask = torch.isfinite(flat_log_probs).all(dim=-1)
                drop_mask = (~finite_row_mask) & (flat_targets != -100)
                if drop_mask.any():
                    drop_count = int(drop_mask.sum().item())
                    flat_targets[drop_mask] = -100
                    if drop_count > 0 and not self._warned_invalid_logprob:
                        print(
                            f"[warn] CE log-probs contained {drop_count} non-finite rows → masking from loss.",
                            flush=True,
                        )
                        self._warned_invalid_logprob = True

                if (flat_targets != -100).any():
                    ce_loss = F.nll_loss(flat_log_probs, flat_targets, ignore_index=-100, reduction="mean")
                else:
                    ce_loss = torch.zeros((), device=self.student_device, dtype=s_pred.dtype)
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
            # top-k% tokens among valid positions; optionally global threshold via GLS
            # Select positions FIRST, then compute KL only on selected rows (efficient)
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))

            # Build ranking statistic with two-stage selection when scoring enabled
            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            
            if score_enabled:
                # === Two-stage selection for efficiency ===
                # Stage 1: Prefilter by entropy (cheap, no softmax needed)
                prefilter_mult = float(getattr(self.config, "score_prefilter_multiplier", 3.0))
                # Stage 2: Compute KL/CE only on prefiltered subset
                
                stat = torch.full_like(ent_raw, float('-inf'))
                for i in range(valid_next.size(0)):
                    mask_i = valid_next[i]
                    n_valid = int(mask_i.sum().item())
                    if n_valid < 3:
                        continue
                    
                    # Stage 1: Select top (k * prefilter_mult) by entropy
                    k_final = max(1, min(n_valid, math.ceil(pct * n_valid)))
                    k_pre = min(n_valid, max(k_final, int(k_final * prefilter_mult)))
                    
                    valid_idx = torch.where(mask_i)[0]
                    ent_valid = ent_raw[i][mask_i]
                    _, pre_rel_idx = torch.topk(ent_valid, k=k_pre, largest=True, sorted=False)
                    prefilter_idx = valid_idx[pre_rel_idx]  # [k_pre] absolute indices
                    
                    # Stage 2: Compute KL/CE only on prefiltered positions (efficient!)
                    # Create mask for prefiltered positions
                    prefilter_mask = torch.zeros_like(mask_i, dtype=torch.bool)
                    prefilter_mask[prefilter_idx] = True
                    
                    # Gather only prefiltered rows for KL computation
                    # Ensure indices are on the same device as teacher log-probs
                    pre_idx_t = prefilter_idx.to(t_log_probs.device)
                    t_rows_pre = t_log_probs[i, pre_idx_t, :].to(self.student_device)  # [k_pre, V]
                    s_rows_pre = s_log_probs[i, prefilter_idx, :]  # [k_pre, V]
                    kl_pre = self._kl_loss(t_rows_pre, s_rows_pre)  # [k_pre]
                    
                    # Build full-size KL tensor with -inf for non-prefiltered
                    kl_pos_partial = torch.full((ent_raw.size(1),), float('-inf'), device=self.student_device)
                    kl_pos_partial[prefilter_idx] = kl_pre
                    
                    # Prepare score context with partial KL
                    score_ctx_i = self._prepare_score_context(
                        ent_raw[i:i+1],
                        kl_pos_partial.unsqueeze(0),
                        s_log_probs[i:i+1] if s_log_probs is not None else None,
                        prefilter_mask.unsqueeze(0),
                        input_ids[i:i+1]
                    )
                    combined = self._build_score_vector(score_ctx_i, 0, prefilter_mask)
                    if combined is not None:
                        stat[i] = combined
                
                stat = stat.masked_fill(~valid_next, float('-inf'))
            else:
                # Entropy-only ranking (no softmax needed, fully efficient)
                stat = ent_raw.masked_fill(~valid_next, float('-inf'))

            use_gls = bool(getattr(self.config, "gls_enabled", False))
            sel_topk_count = 0
            sel_gls_count = 0
            
            # Create boolean mask for positions to include in KD loss
            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)  # [B, L-1]
            
            if not use_gls:
                # Per-example top-k
                for i in range(valid_next.size(0)):
                    mask = valid_next[i]
                    n_valid = int(mask.sum().item())
                    if n_valid < 3:
                        continue
                    k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                    sel_topk_count += int(k)
                    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    scores = stat[i][mask]
                    if scores.numel() == 0:
                        continue
                    _, rel_idx = torch.topk(scores, k=k, largest=True, sorted=False)
                    sel = valid_idx[rel_idx]
                    keep_mask[i, sel] = True
            else:
                # GLS: global threshold over history with warm-up fallback
                self._gls_init_if_needed()
                thr = self._gls_threshold(top_percent=self.config.k_percent)
                if thr is None:
                    # Warm-up: fallback to per-example top-k
                    for i in range(valid_next.size(0)):
                        mask = valid_next[i]
                        n_valid = int(mask.sum().item())
                        if n_valid < 3:
                            continue
                        k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                        sel_topk_count += int(k)
                        valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                        scores = stat[i][mask]
                        if scores.numel() == 0:
                            continue
                        _, rel_idx = torch.topk(scores, k=k, largest=True, sorted=False)
                        sel = valid_idx[rel_idx]
                        keep_mask[i, sel] = True
                else:
                    keep_mask = (stat >= thr) & valid_next
                    sel_gls_count = int(keep_mask.sum().item())
                # Push current batch stats and optionally log threshold
                flat_vals = stat[valid_next].detach().float().to("cpu")
                flat_vals = flat_vals[torch.isfinite(flat_vals)]
                self._gls_push(flat_vals)
                if getattr(self.config, "gls_log_threshold", False) and ('thr' in locals()) and thr is not None and self.logger:
                    self.logger.log_scalar("train/gls_threshold", float(thr), self.global_step)
            
            # Compute KL only on selected positions (efficient)
            rows = keep_mask.nonzero(as_tuple=False)  # [P, 2] -> (b, t)
            if rows.numel() == 0:
                kd_loss = t_pred.sum() * 0.0
            else:
                b_idx, t_idx = rows[:, 0], rows[:, 1]
                # Teacher gather requires indices on the teacher tensor's device
                device_t = t_log_probs.device
                b_idx_t = b_idx.to(device_t)
                t_idx_t = t_idx.to(device_t)
                t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
                # Student tensors live on student device; indices are already there
                s_rows = s_log_probs[b_idx, t_idx, :]
                kd_loss = self._kl_loss(t_rows, s_rows).mean()
            
            # Log selection counters (per batch)
            if self.logger:
                self.logger.log({
                    "train/selected_tokens_topk": float(sel_topk_count),
                    "train/selected_tokens_gls": float(sel_gls_count),
                }, self.global_step)

        elif self.config.distill_type == "bucket":
            # Bucket: distill on tokens with entropy in [lower_bound, upper_bound] percentiles
            # Select positions FIRST, then compute KL only on selected rows (efficient)
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]

            score_enabled = bool(getattr(self.config, "score_token_selection", False))

            # Create boolean mask for positions to include in KD loss
            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)  # [B, L-1]
            
            Bsz = t_log_probs.size(0)
            for i in range(Bsz):
                valid_next_i = valid_next[i]
                if valid_next_i.sum() < 3:  # Need at least 3 tokens for bucket selection
                    continue

                if score_enabled:
                    # === Two-stage selection for efficiency ===
                    # Stage 1: Prefilter by entropy to bucket range
                    ent_valid = ent_raw[i][valid_next_i].float()
                    lower_thr = torch.quantile(ent_valid, self.config.bucket_lower_percent / 100.0)
                    upper_thr = torch.quantile(ent_valid, self.config.bucket_upper_percent / 100.0)
                    
                    valid_idx = torch.where(valid_next_i)[0]
                    prefilter_mask_rel = (ent_valid >= lower_thr) & (ent_valid <= upper_thr)
                    
                    if not prefilter_mask_rel.any():
                        continue
                    
                    prefilter_idx = valid_idx[prefilter_mask_rel]  # Absolute indices in bucket
                    
                    # Stage 2: Compute KL/CE only on bucket positions (efficient!)
                    pre_idx_t = prefilter_idx.to(t_log_probs.device)
                    t_rows_pre = t_log_probs[i, pre_idx_t, :].to(self.student_device)  # [k_bucket, V]
                    s_rows_pre = s_log_probs[i, prefilter_idx, :]  # [k_bucket, V]
                    kl_pre = self._kl_loss(t_rows_pre, s_rows_pre)  # [k_bucket]
                    
                    # Build full-size KL tensor
                    kl_pos_partial = torch.full((ent_raw.size(1),), float('-inf'), device=self.student_device)
                    kl_pos_partial[prefilter_idx] = kl_pre
                    
                    # Create prefilter mask for score context
                    prefilter_mask_full = torch.zeros_like(valid_next_i, dtype=torch.bool)
                    prefilter_mask_full[prefilter_idx] = True
                    
                    # Prepare score context with partial KL
                    score_ctx_i = self._prepare_score_context(
                        ent_raw[i:i+1],
                        kl_pos_partial.unsqueeze(0),
                        s_log_probs[i:i+1] if s_log_probs is not None else None,
                        prefilter_mask_full.unsqueeze(0),
                        input_ids[i:i+1]
                    )
                    combined = self._build_score_vector(score_ctx_i, 0, prefilter_mask_full)
                    if combined is None:
                        continue
                    
                    # Apply bucket thresholds to combined score
                    vec = combined[prefilter_mask_full].float()
                    if vec.numel() < 1:
                        continue
                    score_lower = torch.quantile(vec, max(0.0, (self.config.bucket_lower_percent - self.config.bucket_lower_percent) / 100.0))
                    score_upper = torch.quantile(vec, min(1.0, (self.config.bucket_upper_percent - self.config.bucket_lower_percent) / (100.0 - self.config.bucket_lower_percent)))
                    
                    # Final selection within prefiltered set
                    final_sel = (vec >= score_lower) & (vec <= score_upper)
                    if final_sel.any():
                        keep_mask[i, prefilter_idx[final_sel]] = True
                else:
                    # Entropy-only bucket (no softmax needed)
                    vec = ent_raw[i][valid_next_i].float()
                    lower_thr = torch.quantile(vec, self.config.bucket_lower_percent / 100.0)
                    upper_thr = torch.quantile(vec, self.config.bucket_upper_percent / 100.0)

                    keep_idx = torch.where(valid_next_i)[0]
                    sel_mask = (vec >= lower_thr) & (vec <= upper_thr)
                    if sel_mask.any():
                        keep_mask[i, keep_idx[sel_mask]] = True

            # Compute KL only on selected positions (efficient)
            rows = keep_mask.nonzero(as_tuple=False)  # [P, 2] -> (b, t)
            if rows.numel() == 0:
                kd_loss = t_pred.sum() * 0.0
            else:
                b_idx, t_idx = rows[:, 0], rows[:, 1]
                device_t = t_log_probs.device
                b_idx_t = b_idx.to(device_t)
                t_idx_t = t_idx.to(device_t)
                t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
                s_rows = s_log_probs[b_idx, t_idx, :]
                kd_loss = self._kl_loss(t_rows, s_rows).mean()
        elif self.config.distill_type == "random":
            # Random selection with optional score-weighted sampling
            # Select positions FIRST, then compute KL only on selected rows (efficient)
            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            score_ctx = None
            ent_raw = None
            if score_enabled:
                ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1] for context only
                kl_pos_for_score = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)
                score_ctx = self._prepare_score_context(ent_raw, kl_pos_for_score, s_log_probs, valid_next, input_ids)

            # Create boolean mask for positions to include in KD loss
            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)  # [B, L-1]
            
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

                keep_mask[i, selected_indices] = True

            # Compute KL only on selected positions (efficient)
            rows = keep_mask.nonzero(as_tuple=False)  # [P, 2] -> (b, t)
            if rows.numel() == 0:
                kd_loss = t_pred.sum() * 0.0
            else:
                b_idx, t_idx = rows[:, 0], rows[:, 1]
                device_t = t_log_probs.device
                b_idx_t = b_idx.to(device_t)
                t_idx_t = t_idx.to(device_t)
                t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
                s_rows = s_log_probs[b_idx, t_idx, :]
                kd_loss = self._kl_loss(t_rows, s_rows).mean()
        elif self.config.distill_type == "pos-rs-kd":
            # RS-KD over POSITIONS: sample K% positions by distribution q(i)
            # Select positions FIRST with importance weights, then compute KL only on selected rows
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]

            score_enabled = bool(getattr(self.config, "score_token_selection", False))
            score_ctx = None
            if score_enabled:
                kl_pos_for_score = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)
                score_ctx = self._prepare_score_context(ent_raw, kl_pos_for_score, s_log_probs, valid_next, input_ids)

            # Build per-position importance weights
            Bsz = t_pred.size(0)
            alpha = float(getattr(self.config, "rs_alpha", 1.0))
            q_floor = float(getattr(self.config, "rs_floor", 1e-6))
            pct = max(0.0, min(1.0, self.config.k_percent / 100.0))

            # Collect selected positions and their importance weights
            selected_positions: List[Tuple[int, int, float]] = []  # (b, t, weight)
            
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

                base = torch.clamp(base, min=1e-8)  # H_t ≥ 0 already
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

                # Compute unnormalized importance weights: w = 1/q (global normalization later)
                q_sel = q[sel_rel]  # [k]
                w = 1.0 / torch.clamp(q_sel, min=q_floor)
                
                # Store (batch_idx, time_idx, weight) with UNNORMALIZED weights
                for j, pos in enumerate(selected_abs):
                    selected_positions.append((i, pos.item(), w[j].item()))

            # Compute weighted KL only on selected positions
            if len(selected_positions) == 0:
                kd_loss = t_pred.sum() * 0.0
            else:
                b_indices = torch.tensor([p[0] for p in selected_positions], dtype=torch.long, device=self.student_device)
                t_indices = torch.tensor([p[1] for p in selected_positions], dtype=torch.long, device=self.student_device)
                weights = torch.tensor([p[2] for p in selected_positions], dtype=torch.float32, device=self.student_device)
                # Move indices to teacher device for gather
                b_idx_t = b_indices.to(t_log_probs.device)
                t_idx_t = t_indices.to(t_log_probs.device)
                t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
                s_rows = s_log_probs[b_indices, t_indices, :]
                kl_per_pos = self._kl_loss(t_rows, s_rows)  # [P]
                # Global normalization across ALL selected tokens → weighted per-token batch mean
                w_all = weights  # [P] (unnormalized)
                kd_loss = (kl_per_pos * w_all).sum() / w_all.sum().clamp_min(1e-12)
        elif self.config.distill_type == "linucb":
            if self.bandit_manager is None:
                raise RuntimeError("LinUCB bandit is not initialized.")
            ent_raw = self._entropy_for_selection(input_ids, t_pred)
            # Use the same temperature as KD for CE computation (consistency fix)
            kl_pos = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)

            targets = input_ids[:, 1:].to(self.student_device)
            targets = targets.masked_fill(~valid_next, 0)
            # Teacher/student CE and KL are the contextual features consumed by the bandit.
            # Use log_probs at temperature T (already computed) for consistency
            teacher_ce = (-t_log_probs.to(self.student_device).gather(-1, targets.unsqueeze(-1)).squeeze(-1)).detach()
            student_ce = (-s_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).detach()

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

        # Track OOM failures: consecutive and total
        consecutive_oom_failures = 0
        total_oom_failures = 0
        max_consecutive_ooms = 10
        max_total_oom_percent = 50  # Abort if >50% of batches OOM
        oom_failure_sleep_s = 100
        batches_attempted = 0
        
        for epoch in range(epochs):
            step_start = time.time()
            running = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
            bandit_running: Dict[str, float] = {}
            bandit_steps = 0
            last_reward_metrics: Optional[Dict[str, float]] = None
            self.opt.zero_grad(set_to_none=True)  # Initialize gradients
            
            step = 0  # Manual step counter that only increments on success
            for batch in self.dataloader:
                batches_attempted += 1
                # Per-batch forward/backward with CUDA OOM handling
                try:
                    loss, kl_val, ce_val, bandit_metrics = self._forward_batch(batch)
                    # Scale loss by gradient accumulation steps
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    # ===== OOM Reduction: AMP backward pass =====
                    scaler = getattr(self, "_scaler", None)
                    if scaler is not None:
                        # fp16: use scaler
                        scaler.scale(loss).backward()
                    else:
                        # bfloat16 or no AMP: standard backward
                        loss.backward()
                    
                    # Reset consecutive OOM counter on success
                    consecutive_oom_failures = 0
                    step += 1  # Only increment step on successful batch
                except torch.cuda.OutOfMemoryError:
                    consecutive_oom_failures += 1
                    total_oom_failures += 1
                    # Clear partial grads and GPU caches
                    try:
                        self.opt.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass
                    
                    oom_rate = (total_oom_failures / batches_attempted) * 100
                    print(f"[OOM] CUDA out of memory at epoch {epoch + 1}, step {step + 1}. "
                          f"Consecutive: {consecutive_oom_failures}/{max_consecutive_ooms}, "
                          f"Total: {total_oom_failures}/{batches_attempted} ({oom_rate:.1f}%). "
                          f"Skipping batch...")
                    
                    # Exit if too many consecutive OOMs
                    if consecutive_oom_failures >= max_consecutive_ooms:
                        print(f"[OOM] Reached {max_consecutive_ooms} consecutive OOM failures. "
                              f"Sleeping {oom_failure_sleep_s}s before exiting...")
                        time.sleep(oom_failure_sleep_s)
                        raise RuntimeError(f"Training aborted after {max_consecutive_ooms} consecutive CUDA OOM failures.")
                    
                    # Exit if OOM rate is too high (only check after seeing enough batches)
                    if batches_attempted >= 20 and oom_rate > max_total_oom_percent:
                        print(f"[OOM] OOM rate ({oom_rate:.1f}%) exceeds threshold ({max_total_oom_percent}%). "
                              f"Sleeping {oom_failure_sleep_s}s before exiting...")
                        time.sleep(oom_failure_sleep_s)
                        raise RuntimeError(f"Training aborted: {total_oom_failures}/{batches_attempted} batches failed with OOM ({oom_rate:.1f}%).")
                    
                    # Skip this batch and continue to the next one (step counter doesn't increment)
                    continue
                
                # Only update weights after accumulation steps
                if step % self.config.gradient_accumulation_steps == 0:
                    # ===== OOM Reduction: AMP optimizer step =====
                    scaler = getattr(self, "_scaler", None)
                    if scaler is not None:
                        # fp16: unscale, clip, step, update scaler
                        scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                        scaler.step(self.opt)
                        scaler.update()
                    else:
                        # bfloat16 or no AMP: standard step
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


__all__ = ["Distiller", "kl_from_vocab_samples"]
