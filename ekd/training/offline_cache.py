import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

import torch
from .entropy_utils import truncated_entropy_topk_tail_midpoint


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
        out_path = self.cache_dir / fname
        torch.save(item, out_path)
        self.manifest["items"][key] = fname
        self.save_manifest()
        try:
            return os.path.getsize(out_path)
        except Exception:
            return 0

    def read_item(self, key: str) -> Dict[str, Any]:
        return torch.load(self.path_for(key), map_location="cpu")


def _repo_root() -> Path:
    # repo root is two levels up from this file's parent: ekd/training/ -> ekd/ -> project root
    return Path(__file__).resolve().parents[2]


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


def init_offline_cache_for_trainer(trainer: Any) -> TeacherOfflineCache:
    """Create or reuse a global logits cache dir under <repo_root>/logits_caches/<hash>.
    Also maintain an index.json mapping hash -> signature for clarity.
    Returns a TeacherOfflineCache instance.
    """
    if cfg_dir:
        cache_dir = Path(cfg_dir)
    else:
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


def _build_cache_pass(
    cache: TeacherOfflineCache,
    teacher,
    dataloader,
    teacher_device,
    sanitize_logits_fn,
    T: float,
    k_approx: int,
    S_vocab: int,
    beta: float,
):
    """Internal: run a single offline teacher pass to populate cache. Returns (maybe_cache, V_last)."""
    teacher.eval()
    maybe_cache = 0
    V_last = None
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]  # [B, L]
            attn_mask = batch["attention_mask"]  # [B, L]
            input_ids_t = input_ids.to(teacher_device)
            attn_t = attn_mask.to(teacher_device)

            out = teacher(
                input_ids_t, attention_mask=attn_t, output_hidden_states=False
            )
            t_logits = sanitize_logits_fn(out.logits, "teacher")  # [B,L,V]

            B, L, V = t_logits.shape
            V_last = V
            t_pred = t_logits[:, :-1, :]  # [B, L-1, V]
            valid_next = attn_mask[:, 1:].bool()  # [B, L-1]
            # For cache we persist t_logp_sel strictly at T=1
            t_logp_full_T1 = torch.log_softmax(t_pred / 1.0, dim=-1)  # [B, L-1, V]
            # For RS-KD proposal q we allow temperature T (entropy_approx_temperature)
            t_logp_full_T = torch.log_softmax(t_pred / T, dim=-1)  # [B, L-1, V]
            p_full_T = t_logp_full_T.exp()  # [B, L-1, V]

            # per example
            for i in range(B):
                key = TeacherOfflineCache.key_from_ids(input_ids[i])
                if cache.has(key):
                    continue

                valid_i = valid_next[i]  # [L-1]
                pred_i = t_pred[i]  # [L-1, V]
                logp_i_T1 = t_logp_full_T1[i]  # [L-1, V]
                p_i_T = p_full_T[i]  # [L-1, V]

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
                    H_hat = truncated_entropy_topk_tail_midpoint(
                        pred_i[pos], k=k_approx
                    )  # scalar tensor
                    H_hat_list.append(H_hat)

                    # RS-KD over vocabulary: proposal q ∝ p^beta, sample S without replacement
                    p_pos = p_i_T[pos]  # [V]
                    if beta != 1.0:
                        q_un = p_pos.clamp_min(1e-12).pow(beta)
                    else:
                        q_un = p_pos.clamp_min(1e-12)
                    q = q_un / q_un.sum()

                    S = min(S_vocab, V)
                    # multinomial w/o replacement via Gumbel-top-k trick fallback when needed
                    idx = torch.multinomial(q, num_samples=S, replacement=False)  # [S]
                    t_logp_sel = logp_i_T1[pos, idx]  # [S] (persist at T=1)
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
                cache.write_item(key, item)
                maybe_cache += 1
                # periodic progress print every 100 items
                if maybe_cache % 100 == 0:
                    print(f"[logits-cache] Progress: cached {maybe_cache} new items so far...")
    return maybe_cache, V_last


def _recompute_and_persist_stats(
    cache: TeacherOfflineCache,
    tok,
    k_approx: int,
    V_last: int | None,
):
    """Internal: recompute manifest stats and persist them. Returns (stats, total_items)."""
    idx_map = cache.manifest.get("items", {})
    total_items = len(idx_map)
    # Determine vocabulary size for baseline calculations
    V_base = getattr(tok, "vocab_size", None)
    if V_base is None:
        V_base = V_last if V_last is not None else 0

    stats = {
        "approx_entropy_logits_saved": 0,
        "rs_kd_ids_saved": 0,
        "rs_kd_probs_saved": 0,
        "ce_logits_needed": 0,
        "cache_bytes": 0,
        "baseline_full_logits_bytes": 0,
    }
    total_valid_positions = 0
    for rel in idx_map.values():
        fpath = cache.cache_dir / rel
        try:
            d = torch.load(fpath, map_location="cpu")
        except Exception:
            continue

        vm = d.get("valid_mask", None)
        if vm is None:
            continue
        try:
            seq_valid = int(vm.sum().item())
        except Exception:
            # fallback for non-tensor masks
            seq_valid = int(sum(bool(x) for x in vm))
        total_valid_positions += seq_valid

        topk_m = int(d.get("topk_m", k_approx))
        if V_base:
            stats["approx_entropy_logits_saved"] += int(seq_valid * min(topk_m, V_base))
        else:
            stats["approx_entropy_logits_saved"] += int(seq_valid * topk_m)

        rs = d.get("rs", {}) or {}
        idx_list = rs.get("idx", []) or []
        q_list = rs.get("q", []) or []

        s_total = 0
        for t in idx_list:
            try:
                s_total += int(len(t))
            except Exception:
                pass
        stats["rs_kd_ids_saved"] += s_total

        probs_total = 0
        for t in q_list:
            try:
                probs_total += int(len(t))
            except Exception:
                pass
        if probs_total == 0 and s_total:
            probs_total = s_total
        stats["rs_kd_probs_saved"] += probs_total
        stats["ce_logits_needed"] += s_total

        try:
            stats["cache_bytes"] += int(os.path.getsize(fpath))
        except Exception:
            pass

    if V_base:
        stats["baseline_full_logits_bytes"] = int(total_valid_positions * V_base * 4)

    # Persist stats in manifest
    cache.manifest.setdefault("stats", {})
    cache.manifest["stats"] = stats
    cache.save_manifest()

    return stats, total_items


def build_offline_cache_if_needed(
    cache: TeacherOfflineCache,
    teacher,
    tok,
    dataloader,
    config,
    teacher_device,
    sanitize_logits_fn,
) -> TeacherOfflineCache:
    """
    One pass over the dataset with the TEACHER to compute:
      - truncated entropy H_hat with m=k_approx (Sec. 3.6 in EHM paper)
      - RS-KD proposal over vocabulary per position + sampled tokens

    Skips entirely if manifest signature matches.
    """
    # Expect a pre-initialized cache from caller to avoid hidden side-effects
    if cache is None:
        raise ValueError("cache must be provided (initialize it once via init_offline_cache_for_trainer)")

    # build signature (changes -> rebuild)
    # Build signature from provided components
    sig = {
        "teacher_name": getattr(getattr(getattr(teacher, "config", None), "_name_or_path", None),
                                 "__str__", lambda: getattr(getattr(teacher, "config", None), "_name_or_path", "unknown"))(),
        "tokenizer_name": getattr(tok, "name_or_path", "unknown"),
        "max_seq_len": int(getattr(config, "max_seq_len", 0)),
        "entropy_approx_m": int(getattr(config, "entropy_approx_m", 12)),
        "rs_vocab_samples": int(getattr(config, "rs_vocab_samples", 12)),
        "rs_vocab_beta": float(getattr(config, "rs_vocab_beta", 1.0)),
        "entropy_approx_temperature": float(
            getattr(config, "entropy_approx_temperature", getattr(config, "cache_temperature", 1.0))
        ),
        "dataset_len": int(len(dataloader.dataset)) if hasattr(dataloader, "dataset") else -1,
    }

    if cache.signature_matches(sig):
        print(
            f"[logits-cache] Cache found with matching signature - using existing cache at {cache.cache_dir}."
        )
        return cache

    print(
        f"[logits-cache] No cache found or signature changed - building teacher cache (one pass over dataset) at {cache.cache_dir}..."
    )
    cache.set_signature(sig)

    # Use configured temperature for offline entropy approximation/proposals (distinct from runtime KD temperature)
    T = float(getattr(config, "entropy_approx_temperature", getattr(config, "cache_temperature", 1.0)))
    k_approx = int(getattr(config, "entropy_approx_m", 12))
    S_vocab = int(getattr(config, "rs_vocab_samples", 12))
    beta = float(getattr(config, "rs_vocab_beta", 1.0))  # q ∝ p^beta

    maybe_cache, V_last = _build_cache_pass(
        cache=cache,
        teacher=teacher,
        dataloader=dataloader,
        teacher_device=teacher_device,
        sanitize_logits_fn=sanitize_logits_fn,
        T=T,
        k_approx=k_approx,
        S_vocab=S_vocab,
        beta=beta,
    )
    # Recompute and persist cache-wide stats by scanning manifest items
    stats, total_items = _recompute_and_persist_stats(
        cache=cache, tok=tok, k_approx=k_approx, V_last=V_last
    )

    saved_bytes = max(0, stats.get("baseline_full_logits_bytes", 0) - stats.get("cache_bytes", 0))
    print(f"[logits-cache] Done. Cached {maybe_cache} new items. Total items in cache: {total_items}.")
    print(
        "[logits-cache] Stats: "
        f"approx_entropy_logits_saved={stats['approx_entropy_logits_saved']}, "
        f"rs_ids={stats['rs_kd_ids_saved']}, "
        f"rs_probs={stats['rs_kd_probs_saved']}, "
        f"ce_logits_needed={stats['ce_logits_needed']}, "
        f"bytes: cache={stats['cache_bytes']:,}, "
        f"baseline_full={stats.get('baseline_full_logits_bytes', 0):,}, "
        f"saved={saved_bytes:,}"
    )
    return cache
