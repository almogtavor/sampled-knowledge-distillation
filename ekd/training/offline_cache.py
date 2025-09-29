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
            # Packed CSR-style arrays per sequence (preferred):
            # pos_offsets: Int32 [L-1+1] (start index per position; last is total)
            # idx_flat: Int32 [sum S_i]
            # t_logp_flat: Float16 [sum S_i]
            # q_flat: Float16 [sum S_i]
          }
        }
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest: Dict[str, Any] = {"signature": {}, "items": {}}
        # Batched manifest writes
        self._dirty = False
        self._flush_every = 256  # flush manifest every N items
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())

    @staticmethod
    def key_from_ids(input_ids: torch.Tensor) -> str:
        # hash over raw ids so it's stable regardless of batch order
        h = hashlib.sha1(input_ids.cpu().numpy().tobytes()).hexdigest()
        return h

    def save_manifest(self, force: bool = False):
        if not force and not self._dirty:
            return
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest))
        os.replace(tmp, self.manifest_path)
        self._dirty = False

    def set_signature(self, signature: Dict[str, Any]):
        self.manifest["signature"] = signature
        # Save immediately to anchor cache dir with signature
        self._dirty = True
        self.save_manifest(force=True)

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
        # mark dirty and flush periodically
        self._dirty = True
        if (idx % self._flush_every) == 0:
            self.save_manifest()
        try:
            return os.path.getsize(out_path)
        except Exception:
            return 0

    def read_item(self, key: str) -> Dict[str, Any]:
        return torch.load(self.path_for(key), map_location="cpu")


class ShardedTeacherOfflineCache:
    """Shard-based cache with manifest mapping key -> {shard, index}.

    Shards contain a list of already-packed items, keeping save/load fast and compact.
    """

    def __init__(self, base_dir: Path, shard_size: int = 2048):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base / "manifest.json"
        self.shard_size = int(shard_size)
        self.manifest: Dict[str, Any] = {"signature": {}, "items": {}, "shards": []}
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text())
            except Exception:
                # keep empty manifest if parse fails
                pass
        self._current = []  # buffered items for current shard
        self._dirty = False

    # Compatibility helpers
    def has(self, key: str) -> bool:
        return key in self.manifest.get("items", {})

    @property
    def cache_dir(self) -> Path:
        return self.base

    def _flush_shard(self):
        if not self._current:
            return
        shard_id = len(self.manifest.get("shards", []))
        path = self.base / f"shard_{shard_id:06d}.pt"
        torch.save(self._current, path)
        self.manifest.setdefault("shards", []).append({"path": path.name, "n": len(self._current)})
        self._current = []
        self._dirty = True

    def add_item(self, key: str, item: Dict[str, Any]):
        if key in self.manifest.get("items", {}):
            return 0
        local_index = len(self._current)
        self.manifest.setdefault("items", {})[key] = {"shard": len(self.manifest.get("shards", [])), "index": local_index}
        self._current.append(item)
        self._dirty = True
        if len(self._current) >= self.shard_size:
            self._flush_shard()
        return 0

    # Legacy name used by builder
    def write_item(self, key: str, item: Dict[str, Any]):
        return self.add_item(key, item)

    def read_item(self, key: str) -> Dict[str, Any]:
        ref = self.manifest.get("items", {}).get(key)
        if ref is None:
            raise KeyError(key)
        # Sharded reference
        shard_meta = self.manifest["shards"][ref["shard"]]
        items = torch.load(self.base / shard_meta["path"], map_location="cpu")
        return items[ref["index"]]

    def set_signature(self, signature: Dict[str, Any]):
        self.manifest["signature"] = signature
        self._dirty = True
        self.save_manifest(force=True)

    def signature_matches(self, signature: Dict[str, Any]) -> bool:
        return self.manifest.get("signature") == signature

    def save_manifest(self, force: bool = False):
        if not force and not self._dirty:
            return
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        os.replace(tmp, self.manifest_path)
        self._dirty = False

    def finalize(self):
        self._flush_shard()
        self.save_manifest(force=True)


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


def init_offline_cache_for_trainer(cfg_dir, sig) -> TeacherOfflineCache:
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

    # Use sharded cache for performance; keeps same external methods used by trainer
    cache = ShardedTeacherOfflineCache(cache_dir)
    try:
        n_items = len(cache.manifest.get("items", {}))
    except Exception:
        n_items = 0
    print(f"[logits-cache] Enabled. Using cache dir: {cache_dir} (items={n_items})")
    # Optionally, point to the index for discoverability
    print(f"[logits-cache] Index: {_index_path()}")
    return cache


def pack_ragged(rs_idx_list, rs_logp_list, rs_q_list):
    """Pack ragged RS-KD lists into CSR-style flat arrays.

    Returns: (pos_offsets[Int32], idx_flat[Int32], lp_flat[F16], q_flat[F16])
    """
    lengths = torch.tensor([int(t.numel()) for t in rs_idx_list], dtype=torch.int32)
    pos_offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    if len(lengths) > 0:
        pos_offsets[1:] = torch.cumsum(lengths, dim=0)
    total = int(pos_offsets[-1].item()) if pos_offsets.numel() > 0 else 0
    if total == 0:
        return (
            pos_offsets,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.float16),
            torch.empty(0, dtype=torch.float16),
        )
    idx_flat = torch.empty(total, dtype=torch.int32)
    lp_flat = torch.empty(total, dtype=torch.float16)
    q_flat = torch.empty(total, dtype=torch.float16)
    start = 0
    for t_idx, t_lp, t_q in zip(rs_idx_list, rs_logp_list, rs_q_list):
        n = int(t_idx.numel())
        if n:
            idx_flat[start:start+n] = t_idx.to(torch.int32)
            lp_flat[start:start+n] = t_lp.to(torch.float16)
            q_flat[start:start+n] = t_q.to(torch.float16)
            start += n
    return pos_offsets, idx_flat, lp_flat, q_flat


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
                # collect on device, transfer once per sequence
                rs_idx_seq, rs_logp_seq, rs_q_seq = [], [], []

                for pos, is_valid in enumerate(valid_i.tolist()):
                    if not is_valid:
                        H_hat_list.append(torch.tensor(0.0, device=pred_i.device))
                        rs_idx_seq.append(torch.empty(0, dtype=torch.long, device=pred_i.device))
                        rs_logp_seq.append(torch.empty(0, device=pred_i.device))
                        rs_q_seq.append(torch.empty(0, device=pred_i.device))
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

                    rs_idx_seq.append(idx)
                    rs_logp_seq.append(t_logp_sel)
                    rs_q_seq.append(q_sel)

                # Move once to CPU and cast to compact dtypes, then pack into CSR-style arrays
                rs_idx_list = [t.cpu().to(torch.int32) for t in rs_idx_seq]
                rs_logp_list = [t.cpu().to(torch.float16) for t in rs_logp_seq]
                rs_q_list = [t.cpu().to(torch.float16) for t in rs_q_seq]
                pos_offsets, idx_flat, lp_flat, q_flat = pack_ragged(rs_idx_list, rs_logp_list, rs_q_list)

                item = {
                    "key": key,
                    "valid_mask": valid_i.cpu(),
                    "topk_m": k_approx,
                    "H_hat": torch.stack(
                        [h if h.numel() else torch.tensor(0.0, device=pred_i.device) for h in H_hat_list]
                    ).to(torch.float16).cpu(),  # [L-1]
                    "rs": {
                        "S": S_vocab,
                        "pos_offsets": pos_offsets,
                        "idx_flat": idx_flat,
                        "t_logp_flat": lp_flat,
                        "q_flat": q_flat,
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
    for key in idx_map.keys():
        try:
            d = cache.read_item(key)
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
        # Packed representation 
        idx_flat = rs["idx_flat"]
        q_flat = rs["q_flat"]
        s_total = int(len(idx_flat))
        probs_total = int(len(q_flat))
        stats["rs_kd_ids_saved"] += s_total
        stats["rs_kd_probs_saved"] += probs_total
        stats["ce_logits_needed"] += s_total

    # Compute cache on-disk size: prefer shard sizes when present
    shards = cache.manifest.get("shards")
    if isinstance(shards, list) and shards:
        for sh in shards:
            try:
                stats["cache_bytes"] += int(os.path.getsize(cache.cache_dir / sh["path"]))
            except Exception:
                pass
    else:
        # No shards means empty cache; keep bytes at 0
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
    # Flush any pending shard so that subsequent reads during stats pass are valid
    finalize = getattr(cache, "finalize", None)
    if callable(finalize):
        cache.finalize()
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
    # Ensure data is flushed: finalize sharded cache (flush shard + manifest) or save manifest
    if callable(finalize):
        cache.finalize()
    else:
        cache.save_manifest(force=True)
    return cache
