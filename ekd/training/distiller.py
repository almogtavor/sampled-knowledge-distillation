import torch
import torch.nn.functional as F
from torch.optim import AdamW

def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per‑token entropy in nats.
    logits: [seq_len, vocab]
    returns: [seq_len]"""
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(-1)

class Distiller:
    """Main distillation trainer class."""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        dataloader,
        distill_type: str = "vanilla",
        top_k_percent: int = 20,
        lr: float = 5e-5,
        alpha_aux_ce: float = 0.1,
        teacher_device: str = "cpu",
        student_device: str = "cuda",
    ):
        # self.teacher = teacher_model.eval()  # frozen
        self.teacher = teacher_model # todo: is it okay?
        self.student = student_model.train()
        self.tok = tokenizer
        self.dl = dataloader
        self.type = distill_type
        self.top_k = top_k_percent
        self.alpha_ce = alpha_aux_ce
        self.opt = AdamW(self.student.parameters(), lr=lr)
        self.teacher_device = teacher_device
        self.student_device = student_device

    def _kl_loss(self, log_p: torch.Tensor, log_q: torch.Tensor):
        """KL(P||Q) where log_p are teacher log‑probs, log_q are student log‑probs."""
        return F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)

    def _forward_batch(self, batch):
        input_ids_teacher = batch["input_ids"].to(self.teacher_device)
        attention_teacher = batch["attention_mask"].to(self.teacher_device)

        input_ids_student = batch["input_ids"].to(self.student_device)
        attention_student = batch["attention_mask"].to(self.student_device)

        with torch.no_grad():
            t_out = self.teacher(input_ids_teacher, attention_mask=attention_teacher, output_hidden_states=False)
            t_logits = t_out.logits # [B, L, V]
            t_log_probs = torch.log_softmax(t_logits, dim=-1)
            ent = token_entropy(t_logits) # [B, L]

        s_out = self.student(input_ids_student, attention_mask=attention_student)
        s_logits = s_out.logits
        s_log_probs = torch.log_softmax(s_logits, dim=-1)

        if self.type == "vanilla":
            kl = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs) # [B, L]
            kd_loss = kl.mean()
        else: # EKD
            # mask top‑k% entropy tokens per example
            kd_losses = []
            for i in range(ent.size(0)):
                seq_ent = ent[i]
                thresh = torch.quantile(seq_ent, 1 - self.top_k / 100.0)
                mask = seq_ent >= thresh  # high‑entropy positions
                kl_i = self._kl_loss(t_log_probs[i].to(self.student_device), s_log_probs[i]) # [L]
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
            for step, batch in enumerate(self.dl):
                loss, kl_val, ce_val = self._forward_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

                # logging
                running["loss"] += loss.item()
                running["kl"] += kl_val
                running["ce"] += ce_val
                if (step + 1) % log_every == 0:
                    n = log_every
                    print(
                        f"ep{epoch+1} step{step+1} | loss={running['loss']/n:.4f} "
                        f"kl={running['kl']/n:.4f} ce={running['ce']/n:.4f}"
                    )
                    running = {k: 0.0 for k in running} 