import os
import json
import hashlib
import time
import math
from pathlib import Path
from typing import Dict, Any

import torch
from .entropy_utils import truncated_entropy_topk_tail_midpoint

# ---- Gumbel-based RS-KD packing (fixed-U entries per position) ----
ID_BITS = 17
PROB_BITS = 7
PROB_QMAX = (1 << PROB_BITS) - 1  # 127
S_SAMPLES_DEFAULT = 50  # draws per position


def gumbel_like(x: torch.Tensor) -> torch.Tensor:
    # Kept for reference; not used by the optimized sampler below
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u.clamp_min(1e-12)))


def sample_with_replacement_from_logits(logits: torch.Tensor, N: int, tau: float, g_buf: torch.Tensor | None = None) -> torch.Tensor:
    """Sample N i.i.d. draws per row from softmax(logits/tau) via Gumbel-Max without normalizing.

    Uses an in-place exponential buffer to generate Gumbel noise: if E~Exp(1), then -log(E) ~ Gumbel(0,1).

    logits: [P, V]  -> returns indices Tensor [P, N]
    """
    z = logits / float(tau)
    P, _ = z.shape
    out = torch.empty(P, N, device=z.device, dtype=torch.long)
    if g_buf is None or g_buf.shape != z.shape or g_buf.device != z.device:
        g_buf = torch.empty_like(z)
    for n in range(N):
        g_buf.exponential_()      # E ~ Exp(1)
        g = g_buf.log_().neg_()   # -log(E) ~ Gumbel(0,1)
        out[:, n] = (z + g).argmax(dim=-1)
    return out


def topU_unique_counts_per_row(samples: torch.Tensor, U: int):
    """For each row in `samples` [P,N], compute unique ids and counts, keep top-U by count."""
    P, _ = samples.shape
    ids_list, cnts_list = [], []
    for r in range(P):
        ids_r, cnts_r = samples[r].unique(return_counts=True)
        if ids_r.numel() > U:
            top = torch.topk(cnts_r, k=U, largest=True, sorted=False).indices
            ids_r, cnts_r = ids_r[top], cnts_r[top]
        ids_list.append(ids_r)
        cnts_list.append(cnts_r)
    return ids_list, cnts_list


def counts_to_q7(cnts: torch.Tensor, N: int) -> torch.Tensor:
    if cnts.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    x = cnts.float() / float(N)
    q = torch.round(x * PROB_QMAX).to(torch.int32)
    diff = int(PROB_QMAX - q.sum().item())
    if diff != 0:
        residual = (x * PROB_QMAX - q.float()).abs()
        order = torch.argsort(residual, descending=True)
        k = min(len(order), abs(diff))
        sign = 1 if diff > 0 else -1
        q[order[:k]] = (q[order[:k]] + sign).clamp(0, PROB_QMAX)
    return q.to(torch.uint8)


def pack_id_q7(ids17: torch.Tensor, q7: torch.Tensor) -> torch.Tensor:
    x = (ids17.to(torch.int64) & ((1 << ID_BITS) - 1)) | (q7.to(torch.int64) << ID_BITS)
    x = x & ((1 << 24) - 1)
    b0 = (x & 0xFF).to(torch.uint8)
    b1 = ((x >> 8) & 0xFF).to(torch.uint8)
    b2 = ((x >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(-1).contiguous()


def unpack_id_q7(packed_flat: torch.Tensor):
    b = packed_flat.view(-1, 3).to(torch.int64)
    x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    ids17 = x & ((1 << ID_BITS) - 1)
    q7 = (x >> ID_BITS) & ((1 << PROB_BITS) - 1)
    return ids17.to(torch.int32), q7.to(torch.uint8)


def build_fixedU_packed_rows(ids_list, cnts_list, U: int, N: int, V: int) -> torch.Tensor:
    P = len(ids_list)
    out = torch.empty(P * U * 3, dtype=torch.uint8)
    for r in range(P):
        ids_r, cnts_r = ids_list[r], cnts_list[r]
        q7 = counts_to_q7(cnts_r, N)
        e = ids_r.numel()
        if e < U:
            pad = U - e
            ids_r = torch.cat([ids_r, torch.full((pad,), V, dtype=torch.int32, device=ids_r.device)])
            q7 = torch.cat([q7, torch.zeros(pad, dtype=torch.uint8, device=q7.device)])
        packed = pack_id_q7(ids_r[:U].to(torch.int32), q7[:U])
        out[r * U * 3:(r + 1) * U * 3] = packed.cpu()
    return out


def decode_ids_probs_from_block(block: torch.Tensor, U: int, sentinel_id: int):
    ids, q7 = unpack_id_q7(block)
    keep = (ids != sentinel_id) & (q7 > 0)
    ids = ids[keep].long()
    probs = (q7[keep].float() / PROB_QMAX).clamp_min(0.0)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return ids, probs


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
    k_approx: int,
    S_vocab: int,
    beta: float,
):
    """Internal: run a single offline teacher pass to populate cache. Returns (maybe_cache, V_last).

    Note: T/beta/S_vocab are not needed for RS sampling that uses Gumbel with
    kd_temperature (tau), U_max, and N_samples configured via build_offline_cache_if_needed.
    """
    teacher.eval()
    build_start_time = time.time()
    last_log_time = build_start_time
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

            # per example
            for i in range(B):
                key = TeacherOfflineCache.key_from_ids(input_ids[i])
                if cache.has(key):
                    continue

                valid_i = valid_next[i]  # [L-1] (bool on CPU)
                pred_i = t_pred[i]       # [L-1, V] (on teacher_device)

                # Build index of valid positions (CPU long)
                pos_idx = torch.nonzero(valid_i, as_tuple=False).squeeze(-1)

                # Decide format from signature; always store Ĥ
                sig = cache.manifest.get('signature', {})
                use_u8 = bool(sig.get('H_hat_u8', True))

                # Preallocate and compute per-position truncated entropy only for valid positions
                H_arr = torch.zeros(valid_i.numel(), device=pred_i.device, dtype=torch.float32)
                for pos_t in pos_idx:
                    pos = int(pos_t.item())
                    H_val = truncated_entropy_topk_tail_midpoint(pred_i[pos], k=k_approx)
                    H_arr[pos] = H_val
                if use_u8:
                    # Quantize to uint8 using cap = ln(V) (natural units), monotonic for selection tasks
                    H_cap = max(1e-6, math.log(max(2, V)))
                    H_norm = (H_arr.clamp(min=0.0, max=H_cap) / H_cap) * 255.0
                    H_stored = torch.round(H_norm).to(torch.uint8).cpu()  # [L-1]
                else:
                    H_stored = H_arr.to(torch.float16).cpu()
                # Build fixed-U packed rows with Gumbel sampling (batched over valid rows)
                tau_target = float(sig.get('kd_temperature', 1.0))
                U_max = int(sig.get('rs_vocab_samples', 12))  # number of unique tokens to store per position for the subsequent sampling
                N_samples = int(sig.get('rs_samples', S_SAMPLES_DEFAULT))

                if pos_idx.numel() > 0:
                    # rows_logits: [P, V] gathered in one shot
                    rows_logits = pred_i.index_select(0, pos_idx.to(pred_i.device))
                    samples = sample_with_replacement_from_logits(rows_logits, N=N_samples, tau=tau_target)
                    ids_list, cnts_list = topU_unique_counts_per_row(samples, U=U_max)
                    packed_rows = build_fixedU_packed_rows(ids_list, cnts_list, U=U_max, N=N_samples, V=V)  # [P*U*3]
                else:
                    packed_rows = torch.empty(0, dtype=torch.uint8)

                rs_packed = torch.full((valid_i.numel() * U_max * 3,), 0, dtype=torch.uint8)
                # Scatter packed rows back to absolute positions; small loop over valid positions only
                for ridx, pos_t in enumerate(pos_idx):
                    pos = int(pos_t.item())
                    start = ridx * U_max * 3
                    rs_packed[pos * U_max * 3:(pos + 1) * U_max * 3] = packed_rows[start:start + U_max * 3]

                item = {
                    "key": key,
                    "valid_mask": valid_i.cpu(),
                    "topk_m": k_approx,
                    ("H_hat_u8" if use_u8 else "H_hat"): H_stored,
                    "rs": {
                        "U": int(U_max),
                        "N": int(N_samples),
                        "id_bits": int(ID_BITS),
                        "prob_bits": int(PROB_BITS),
                        "sentinel_id": int(V),
                        "packed": rs_packed,
                    },
                }
                cache.write_item(key, item)
                maybe_cache += 1
                # periodic progress print every 100 items
                if maybe_cache % 100 == 0:
                    now = time.time()
                    total_elapsed = now - build_start_time
                    delta_elapsed = now - last_log_time
                    print(
                        f"[logits-cache] Progress: cached {maybe_cache} new items so far... "
                        f"total={total_elapsed:.2f}s, since_prev={delta_elapsed:.2f}s"
                    )
                    last_log_time = now
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
    shards = cache.manifest.get("shards")
    if isinstance(shards, list) and shards:
        # Efficient path: load each shard once and iterate in-memory
        for sh in shards:
            try:
                items = torch.load(cache.cache_dir / sh.get("path", ""), map_location="cpu")
            except Exception:
                items = []
            for d in items:
                vm = d.get("valid_mask")
                if vm is None:
                    continue
                seq_valid = int(torch.as_tensor(vm).sum().item())
                total_valid_positions += seq_valid

                topk_m = int(d.get("topk_m", k_approx))
                if V_base:
                    stats["approx_entropy_logits_saved"] += int(seq_valid * min(topk_m, V_base))
                else:
                    stats["approx_entropy_logits_saved"] += int(seq_valid * topk_m)

                rs = d.get("rs", {}) or {}
                if "packed" in rs:
                    U = int(rs.get("U", 0))
                    s_total = int(seq_valid * U)
                    stats["rs_kd_ids_saved"] += s_total
                    stats["rs_kd_probs_saved"] += s_total
                    stats["ce_logits_needed"] += s_total

        # Compute cache on-disk size from shard files directly
        for sh in shards:
            try:
                stats["cache_bytes"] += int(os.path.getsize(cache.cache_dir / sh["path"]))
            except Exception:
                pass
    else:
        # No shards present: assume empty or non-sharded cache and skip stats to avoid heavy I/O.
        # Stats remain zeros; cache_bytes also remains 0 in this branch.
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
        # RS packing/signature knobs
        "kd_temperature": float(getattr(config, "kd_temperature", 1.0)),
        "rs_vocab_samples": int(getattr(config, "rs_vocab_samples", 12)),
        "rs_samples": int(getattr(config, "rs_samples", S_SAMPLES_DEFAULT)),
        "id_bits": int(ID_BITS),
        "prob_bits": int(PROB_BITS),
        "dataset_len": int(len(dataloader.dataset)) if hasattr(dataloader, "dataset") else -1,
        "H_hat_u8": bool(getattr(config, "H_hat_u8", True)),
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

    # S_vocab, beta are no-ops for Gumbel RS-KD
    k_approx = int(getattr(config, "entropy_approx_m", 12))
    S_vocab = 0
    beta = 1.0

    build_wall_start = time.time()
    maybe_cache, V_last = _build_cache_pass(
        cache=cache,
        teacher=teacher,
        dataloader=dataloader,
        teacher_device=teacher_device,
        sanitize_logits_fn=sanitize_logits_fn,
        k_approx=k_approx,
        S_vocab=S_vocab,
        beta=beta,
    )
    # Flush any pending shard so that subsequent reads during stats pass are valid
    finalize = getattr(cache, "finalize", None)
    if callable(finalize):
        cache.finalize()
    # Recompute and persist cache-wide stats by scanning manifest items
    try:
        stats, total_items = _recompute_and_persist_stats(
            cache=cache, tok=tok, k_approx=k_approx, V_last=V_last
        )
    except Exception as e:
        # Don't fail the run if stats collection hits a schema mismatch
        stats = {
            "approx_entropy_logits_saved": 0,
            "rs_kd_ids_saved": 0,
            "rs_kd_probs_saved": 0,
            "ce_logits_needed": 0,
            "cache_bytes": 0,
            "baseline_full_logits_bytes": 0,
        }
        total_items = len(cache.manifest.get("items", {}))
        print(f"[logits-cache][warn] Stats recompute failed: {e}. Continuing without stats.")
    build_wall_elapsed = time.time() - build_wall_start

    saved_bytes = max(0, stats.get("baseline_full_logits_bytes", 0) - stats.get("cache_bytes", 0))
    print(f"[logits-cache] Done. Cached {maybe_cache} new items. Total items in cache: {total_items}.")
    print(f"[logits-cache] Cache build duration: {build_wall_elapsed:.2f}s")
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
