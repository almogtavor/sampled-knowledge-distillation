import math
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .linucb import LinUCBBandit


@dataclass
class BanditTokenRecord:
    example_idx: int
    position: int
    context: torch.Tensor
    kl_before: float


@dataclass
class BanditPendingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: List[BanditTokenRecord]
    temperature: float


class LinUCBBanditController:
    """Encapsulates LinUCB token selection and reward updates."""

    def __init__(
        self,
        tokenizer,
        config,
        student_device: torch.device,
        teacher_device: torch.device,
        sanitize_logits_fn,
    ) -> None:
        self.tok = tokenizer
        self.config = config
        self.student_device = torch.device(student_device)
        self.teacher_device = torch.device(teacher_device)
        self._sanitize_logits = sanitize_logits_fn

        feature_dim = 6  # [entropy, teacher CE, student CE, KL, POS code, norm position]
        self.bandit = LinUCBBandit(
            feature_dim=feature_dim,
            alpha=self.config.bandit_alpha,
            lambda_=self.config.bandit_lambda,
            device=self.config.bandit_device,
        )

        self._pos_cache: Dict[int, float] = {}
        self._pending: List[BanditPendingBatch] = []

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _token_pos_code(self, token_id: int) -> float:
        cached = self._pos_cache.get(int(token_id))
        if cached is not None:
            return cached

        token = self.tok.convert_ids_to_tokens(int(token_id))
        token = token.replace("Ġ", "").replace("▁", "").strip()

        if not token:
            category = 3
        elif any(ch.isalpha() for ch in token):
            category = 0
        elif token.isdigit():
            category = 1
        elif all(ch in string.punctuation for ch in token):
            category = 2
        else:
            category = 3

        code = category / 3.0
        self._pos_cache[int(token_id)] = code
        return code

    def _prepare_features(
        self,
        ent_raw: torch.Tensor,
        teacher_ce: torch.Tensor,
        student_ce: torch.Tensor,
        kl_pos: torch.Tensor,
        input_ids: torch.Tensor,
        valid_next: torch.Tensor,
    ) -> List[Optional[Dict[str, torch.Tensor]]]:
        features: List[Optional[Dict[str, torch.Tensor]]] = []
        seq_len = valid_next.size(1)

        for i in range(valid_next.size(0)):
            mask = valid_next[i]
            valid_idx = torch.where(mask)[0]
            if valid_idx.numel() == 0:
                features.append(None)
                continue

            ent_vals = ent_raw[i, valid_idx].float()
            teacher_vals = teacher_ce[i, valid_idx].float()
            student_vals = student_ce[i, valid_idx].float()
            kl_vals = kl_pos[i, valid_idx].float()

            target_ids = input_ids[i, 1:][valid_idx].tolist()
            pos_codes = torch.tensor(
                [self._token_pos_code(tok_id) for tok_id in target_ids],
                device=ent_vals.device,
                dtype=torch.float32,
            )

            denom = max(1, seq_len - 1)
            pos_norm = valid_idx.float() / denom

            contexts = torch.stack(
                (
                    ent_vals,
                    teacher_vals,
                    student_vals,
                    kl_vals,
                    pos_codes,
                    pos_norm.float(),
                ),
                dim=1,
            )

            top25_count = max(1, min(valid_idx.numel(), math.ceil(0.25 * valid_idx.numel())))
            top25_rel = torch.topk(ent_vals, k=top25_count, largest=True, sorted=False).indices
            top25_abs = valid_idx[top25_rel]

            features.append(
                {
                    "contexts": contexts.detach(),
                    "valid_indices": valid_idx,
                    "top25_abs": top25_abs,
                }
            )

        return features

    def select_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ent_raw: torch.Tensor,
        teacher_ce: torch.Tensor,
        student_ce: torch.Tensor,
        kl_pos: torch.Tensor,
        valid_next: torch.Tensor,
        temperature: float,
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        feature_data = self._prepare_features(
            ent_raw=ent_raw,
            teacher_ce=teacher_ce,
            student_ce=student_ce,
            kl_pos=kl_pos,
            input_ids=input_ids,
            valid_next=valid_next,
        )

        kd_terms: List[torch.Tensor] = []
        pending_tokens: List[BanditTokenRecord] = []
        total_examples = 0
        total_selected = 0.0
        total_selected_fraction = 0.0
        total_overlap_selected = 0.0
        total_overlap_top25 = 0.0

        max_actions = self.config.bandit_max_tokens if self.config.bandit_max_tokens is not None else None

        for idx, data in enumerate(feature_data):
            if not data:
                continue
            contexts = data["contexts"]
            if contexts.size(0) == 0:
                continue

            contexts_cpu = contexts.detach().cpu()
            mask, scores = self.bandit.select(
                contexts_cpu,
                threshold=self.config.bandit_threshold,
                max_actions=max_actions,
            )

            n_valid = contexts_cpu.size(0)
            eff_min = max(1, min(self.config.bandit_min_tokens, n_valid))
            if max_actions is not None:
                eff_min = min(eff_min, max_actions)

            if mask.sum().item() < eff_min:
                topk = torch.topk(scores, k=eff_min, largest=True, sorted=False).indices
                new_mask = torch.zeros_like(mask, dtype=torch.bool)
                new_mask[topk] = True
                mask = new_mask

            if mask.sum().item() == 0:
                top_idx = torch.topk(scores, k=1, largest=True).indices
                mask = torch.zeros_like(mask, dtype=torch.bool)
                mask[top_idx] = True

            selected_rel = torch.where(mask)[0]
            valid_idx = data["valid_indices"]
            selected_abs = valid_idx[selected_rel.to(valid_idx.device)]
            selected_abs_device = selected_abs.to(kl_pos.device)

            if selected_abs_device.numel() == 0:
                continue

            kd_terms.append(kl_pos[idx, selected_abs_device].mean())

            for rel_idx, abs_pos in zip(selected_rel.tolist(), selected_abs_device.tolist()):
                pending_tokens.append(
                    BanditTokenRecord(
                        example_idx=idx,
                        position=abs_pos,
                        context=contexts_cpu[rel_idx].clone().float(),
                        kl_before=float(kl_pos[idx, abs_pos].item()),
                    )
                )

            total_examples += 1
            selected_count = float(selected_abs_device.numel())
            total_selected += selected_count
            total_selected_fraction += selected_count / n_valid

            top25_abs = data["top25_abs"].to(selected_abs_device.device)
            intersection = torch.isin(selected_abs_device, top25_abs).float().sum().item()
            total_overlap_selected += intersection / max(1.0, selected_count)
            total_overlap_top25 += intersection / max(1, float(top25_abs.numel()))

        if pending_tokens:
            self._pending.append(
                BanditPendingBatch(
                    input_ids=input_ids.detach().cpu(),
                    attention_mask=attention_mask.detach().cpu(),
                    tokens=pending_tokens,
                    temperature=temperature,
                )
            )

        metrics: Dict[str, float] = {}
        if total_examples > 0:
            metrics = {
                "bandit/selected_tokens": total_selected / total_examples,
                "bandit/selected_fraction": total_selected_fraction / total_examples,
                "bandit/overlap_selected": total_overlap_selected / total_examples,
                "bandit/overlap_top25": total_overlap_top25 / total_examples,
            }

        return kd_terms, metrics

    # ------------------------------------------------------------------
    # Reward processing
    # ------------------------------------------------------------------
    @staticmethod
    def _kl_loss(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        return F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)

    def process_rewards(self, student_model, teacher_model) -> Optional[Dict[str, float]]:
        if not self._pending:
            return None

        contexts: List[torch.Tensor] = []
        rewards: List[float] = []

        with torch.no_grad():
            was_training = student_model.training
            if was_training:
                student_model.eval()

            for record in self._pending:
                if not record.tokens:
                    continue

                input_ids_cpu = record.input_ids
                attn_cpu = record.attention_mask
                input_ids_s = input_ids_cpu.to(self.student_device)
                attn_mask_s = attn_cpu.to(self.student_device)

                s_logits = student_model(input_ids_s, attention_mask=attn_mask_s).logits
                s_logits = self._sanitize_logits(s_logits, "student_eval")
                s_pred = s_logits[:, :-1, :]
                s_log_probs = torch.log_softmax(s_pred / record.temperature, dim=-1)
                valid_next = attn_mask_s[:, 1:].bool()

                input_ids_t = input_ids_cpu.to(self.teacher_device)
                attn_mask_t = attn_cpu.to(self.teacher_device)
                t_logits = teacher_model(input_ids_t, attention_mask=attn_mask_t, output_hidden_states=False).logits
                t_logits = self._sanitize_logits(t_logits, "teacher_eval")
                t_pred = t_logits[:, :-1, :]
                t_log_probs = torch.log_softmax(t_pred / record.temperature, dim=-1)

                kl_pos_after = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)

                for token in record.tokens:
                    if token.example_idx >= kl_pos_after.size(0):
                        continue
                    if token.position >= kl_pos_after.size(1):
                        continue
                    if not valid_next[token.example_idx, token.position]:
                        continue
                    kl_after = kl_pos_after[token.example_idx, token.position].item()
                    reward = token.kl_before - kl_after
                    clip_val = float(getattr(self.config, "bandit_reward_clip", 0.0))
                    if clip_val > 0.0:
                        reward = max(-clip_val, min(clip_val, reward))
                    contexts.append(token.context.clone())
                    rewards.append(reward)

            if was_training:
                student_model.train()

        self._pending.clear()

        if not rewards:
            return None

        contexts_tensor = torch.stack(contexts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        self.bandit.update(contexts_tensor, rewards_tensor)

        avg_reward = float(rewards_tensor.mean().item())
        positive_rate = float((rewards_tensor > 0).float().mean().item())
        return {
            "bandit/avg_reward": avg_reward,
            "bandit/positive_reward_rate": positive_rate,
        }

    def reset(self) -> None:
        self._pending.clear()
        self._pos_cache.clear()