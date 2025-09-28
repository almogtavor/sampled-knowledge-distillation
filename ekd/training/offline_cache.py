import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

import torch
from .entropy_utils import truncated_entropy_topk_tail


class TeacherOfflineCache:
    """
    Stores per-example cached data for:
      - truncated entropy approximation H_hat (m = k_approx)
      - RS-KD vocabulary proposal samples and metadata per position
    Data layout on disk (under cache_dir):
      manifest.json:
        { "signature": {...}, "items": { key: "item_<idx>.pt", ... } }
      item_000001.pt (torch.save of dict):
        {
          "key": str,
          "valid_mask": BoolTensor [L-1],
          "topk_m": int,
          "H_hat": FloatTensor [L-1],
          "rs": {
            "S": int,
            # ragged lists per position, stored as list[Tensors]
            "idx": List[LongTensor],  # each [S_i] token ids
            "t_logp": List[FloatTensor],  # each [S_i] teacher log p at those ids
            "q": List[FloatTensor],  # each [S_i] proposal probs
          }
        }
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest: Dict[str, Any] = {"signature": {}, "items": {}}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())

    @staticmethod
    def key_from_ids(input_ids: torch.Tensor) -> str:
        # hash over raw ids so it's stable regardless of batch order
        h = hashlib.sha1(input_ids.cpu().numpy().tobytes()).hexdigest()
        return h

    def save_manifest(self):
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest))
        os.replace(tmp, self.manifest_path)

    def set_signature(self, signature: Dict[str, Any]):
        self.manifest["signature"] = signature
        self.save_manifest()

    def signature_matches(self, signature: Dict[str, Any]) -> bool:
        return self.manifest.get("signature") == signature

    def has(self, key: str) -> bool:
        return key in self.manifest.get("items", {})

    def path_for(self, key: str) -> Path:
        rel = self.manifest["items"][key]
        return self.cache_dir / rel

    def write_item(self, key: str, item: Dict[str, Any]):
        idx = len(self.manifest["items"])
        fname = f"item_{idx:06d}.pt"
        torch.save(item, self.cache_dir / fname)
        self.manifest["items"][key] = fname
        self.save_manifest()

    def read_item(self, key: str) -> Dict[str, Any]:
        return torch.load(self.path_for(key), map_location="cpu")


def _repo_root() -> Path:
    # repo root is two levels up from this file's parent: ekd/training/ -> ekd/ -> project root
    return Path(__file__).resolve().parents[2]


def compute_cache_signature(trainer) -> Dict[str, Any]:
    """Compute a stable signature for the logits cache based on teacher/tokenizer/settings/dataset."""
    return {
        "teacher_name": getattr(getattr(trainer.teacher, "config", None), "_name_or_path", "unknown"),
        "tokenizer_name": getattr(trainer.tok, "name_or_path", "unknown"),
        "max_seq_len": int(trainer.config.max_seq_len),
        "entropy_approx_m": int(getattr(trainer.config, "entropy_approx_m", 20)),
        "rs_vocab_samples": int(getattr(trainer.config, "rs_vocab_samples", 64)),
        "rs_vocab_beta": float(getattr(trainer.config, "rs_vocab_beta", 1.0)),
        "dataset_len": int(len(trainer.dataloader.dataset)) if hasattr(trainer.dataloader, "dataset") else -1,
    }


def _cache_base_dir() -> Path:
    # Single canonical location: <repo_root>/logits_caches
    base = _repo_root() / "logits_caches"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _index_path() -> Path:
    return _cache_base_dir() / "index.json"


def _load_index() -> Dict[str, Any]:
    p = _index_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_index(idx: Dict[str, Any]) -> None:
    p = _index_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, indent=2, sort_keys=True))
    os.replace(tmp, p)


def _signature_hash(sig: Dict[str, Any]) -> str:
    b = json.dumps(sig, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def init_offline_cache_for_trainer(trainer) -> TeacherOfflineCache:
    """Create or reuse a global logits cache dir under <repo_root>/logits_caches/<hash>.
    Also maintain an index.json mapping hash -> signature for clarity.
    Returns a TeacherOfflineCache instance.
    """
    # If user overrode the dir in config, honor it
    cfg_dir = getattr(trainer.config, "offline_cache_dir", None)
    if cfg_dir:
        cache_dir = Path(cfg_dir)
    else:
        sig = compute_cache_signature(trainer)
        h = _signature_hash(sig)
        idx = _load_index()
        if h not in idx:
            idx[h] = sig
            _save_index(idx)
        cache_dir = _cache_base_dir() / h

    cache = TeacherOfflineCache(cache_dir)
    try:
        n_items = len(cache.manifest.get("items", {}))
    except Exception:
        n_items = 0
    print(f"[logits-cache] Enabled. Using cache dir: {cache_dir} (items={n_items})")
    # Optionally, point to the index for discoverability
    print(f"[logits-cache] Index: {_index_path()}")
    return cache


def build_offline_cache_if_needed(self):
    """
    One pass over the dataset with the TEACHER to compute:
      - truncated entropy H_hat with m=k_approx (Sec. 3.6 in EHM paper)
      - RS-KD proposal over vocabulary per position + sampled tokens

    Skips entirely if manifest signature matches.
    """
    # Initialize cache if not already set
    if not getattr(self, "cache", None):
        self.cache = init_offline_cache_for_trainer(self)

    # build signature (changes -> rebuild)
    sig = compute_cache_signature(self)

    if self.cache.signature_matches(sig):
        print(
            f"[logits-cache] Cache found with matching signature – using existing cache at {self.cache.cache_dir}."
        )
        return

    print(
        f"[logits-cache] No cache found or signature changed – building teacher cache (one pass over dataset) at {self.cache.cache_dir}..."
    )
    self.cache.set_signature(sig)

    T = 1.0  # use T=1 for the cached statistics
    k_approx = int(getattr(self.config, "entropy_approx_m", 20))
    S_vocab = int(getattr(self.config, "rs_vocab_samples", 64))
    beta = float(getattr(self.config, "rs_vocab_beta", 1.0))  # q ∝ p^beta

    self.teacher.eval()
    maybe_cache = 0
    with torch.no_grad():
        for batch in self.dataloader:
            input_ids = batch["input_ids"]  # [B, L]
            attn_mask = batch["attention_mask"]  # [B, L]
            input_ids_t = input_ids.to(self.teacher_device)
            attn_t = attn_mask.to(self.teacher_device)

            out = self.teacher(
                input_ids_t, attention_mask=attn_t, output_hidden_states=False
            )
            t_logits = self._sanitize_logits(out.logits, "teacher")  # [B,L,V]

            B, L, V = t_logits.shape
            t_pred = t_logits[:, :-1, :]  # [B, L-1, V]
            valid_next = attn_mask[:, 1:].bool()  # [B, L-1]
            # exact log-probs at T=1 for cache
            t_logp_full = torch.log_softmax(t_pred / T, dim=-1)  # [B, L-1, V]
            p_full = t_logp_full.exp()  # [B, L-1, V]

            # per example
            for i in range(B):
                key = TeacherOfflineCache.key_from_ids(input_ids[i])
                if self.cache.has(key):
                    continue

                valid_i = valid_next[i]  # [L-1]
                pred_i = t_pred[i]  # [L-1, V]
                logp_i = t_logp_full[i]  # [L-1, V]
                p_i = p_full[i]  # [L-1, V]

                # truncated entropy per pos
                H_hat_list = []
                rs_idx_list, rs_logp_list, rs_q_list = [], [], []

                for pos, is_valid in enumerate(valid_i.tolist()):
                    if not is_valid:
                        H_hat_list.append(torch.tensor(0.0))
                        rs_idx_list.append(torch.empty(0, dtype=torch.long))
                        rs_logp_list.append(torch.empty(0))
                        rs_q_list.append(torch.empty(0))
                        continue

                    # H_hat (Top-k + tail)
                    H_hat = truncated_entropy_topk_tail(
                        pred_i[pos], k=k_approx
                    )  # scalar tensor
                    H_hat_list.append(H_hat)

                    # RS-KD over vocabulary: proposal q ∝ p^beta, sample S without replacement
                    p_pos = p_i[pos]  # [V]
                    if beta != 1.0:
                        q_un = p_pos.clamp_min(1e-12).pow(beta)
                    else:
                        q_un = p_pos.clamp_min(1e-12)
                    q = q_un / q_un.sum()

                    S = min(S_vocab, V)
                    # multinomial w/o replacement via Gumbel-top-k trick fallback when needed
                    idx = torch.multinomial(q, num_samples=S, replacement=False)  # [S]
                    t_logp_sel = logp_i[pos, idx]  # [S]
                    q_sel = q[idx]  # [S]

                    rs_idx_list.append(idx.cpu())
                    rs_logp_list.append(t_logp_sel.cpu())
                    rs_q_list.append(q_sel.cpu())

                item = {
                    "key": key,
                    "valid_mask": valid_i.cpu(),
                    "topk_m": k_approx,
                    "H_hat": torch.stack(
                        [h if h.numel() else torch.tensor(0.0) for h in H_hat_list]
                    ).cpu(),  # [L-1]
                    "rs": {
                        "S": S_vocab,
                        "idx": rs_idx_list,
                        "t_logp": rs_logp_list,
                        "q": rs_q_list,
                    },
                }
                self.cache.write_item(key, item)
                maybe_cache += 1
                # periodic progress print every 100 items
                if maybe_cache % 100 == 0:
                    print(f"[logits-cache] Progress: cached {maybe_cache} new items so far...")
    total_items = len(self.cache.manifest.get("items", {}))
    print(f"[logits-cache] Done. Cached {maybe_cache} new items. Total items in cache: {total_items}.")
