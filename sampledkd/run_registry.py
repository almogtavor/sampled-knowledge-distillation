import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Keys that do NOT affect training semantics and should be excluded from the hash
EXCLUDED_KEYS = {
    # pure logging/output knobs
    "output_dir",
    "tensorboard_dir",
    "wandb_project",
    "wandb_entity",
    "wandb_enabled",
    # checkpoint/save cadence
    "checkpoint_steps",
    "keep_checkpoints",
    # cache location (but keep offline_cache flag itself)
    "offline_cache_dir",
    "runs_registry",
    "override",
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable dict of parameters filtered/sorted for hashing.

    - Drops EXCLUDED_KEYS
    - Ensures nested structures are basic Python types
    - Does NOT include timestamps, job IDs, or other runtime-only metadata
    """
    def _to_basic(x: Any) -> Any:
        if isinstance(x, (str, int, float, type(None), bool)):
            return x
        if isinstance(x, (list, tuple)):
            return [_to_basic(v) for v in x]
        if isinstance(x, dict):
            # sort keys for stability
            return {k: _to_basic(x[k]) for k in sorted(x.keys())}
        # Fallback to string representation (stable for enums/literals)
        return str(x)

    filtered = {k: v for k, v in params.items() if k not in EXCLUDED_KEYS}
    return _to_basic(filtered)


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Compute a stable SHA256 hash from the normalized parameters."""
    norm = normalize_params(params)
    blob = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # If file corrupted or wrong format, back it up and start fresh
    try:
        bak = path.with_suffix(path.suffix + ".bak")
        path.replace(bak)
    except Exception:
        pass
    return []


def _save_registry(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def find_entry(items: List[Dict[str, Any]], params_hash: str) -> Optional[int]:
    for i, it in enumerate(items):
        if it.get("id") == params_hash:
            return i
    return None


def get_entry(registry_path: Path, params_hash: str) -> Optional[Dict[str, Any]]:
    """Return the registry entry for the given hash, if present."""
    items = _load_registry(registry_path)
    idx = find_entry(items, params_hash)
    if idx is None:
        return None
    return items[idx]


def upsert_run_start(registry_path: Path, params: Dict[str, Any], *,
                     experiment_name: Optional[str] = None,
                     job_id: Optional[str] = None,
                     model_output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Upsert a run entry at training start. Returns the entry dict.

    Schema:
    {
      id: <hash>,
      params: <filtered params>,
      status: "started"|"trained"|"evaluated"|"skipped",
      runs: { train: {...} },
      evals: { light: {...}, heavy: {...} }
    }
    """
    items = _load_registry(registry_path)
    norm_params = normalize_params(params)
    h = compute_params_hash(params)
    idx = find_entry(items, h)
    base = {
        "id": h,
        "params": norm_params,
        "status": "started",
        # Completion flags to disambiguate partial runs from finished ones
        "completed_train": False,
        "completed_eval": False,
        "runs": {
            "train": {
                "experiment": experiment_name,
                "job_id": job_id,
                "output_dir": model_output_dir,
            }
        },
        "evals": {},
    }
    if idx is None:
        items.append(base)
        entry = base
    else:
        # Update minimal fields but keep existing evals/results
        items[idx]["params"] = norm_params
        # Ensure flags exist for backward compatibility
        items[idx].setdefault("completed_train", False)
        items[idx].setdefault("completed_eval", False)
        items[idx].setdefault("runs", {}).setdefault("train", {}).update({
            "experiment": experiment_name,
            "job_id": job_id,
            "output_dir": model_output_dir,
        })
        items[idx].setdefault("evals", {})
        # Do not override status here; caller can change later
        entry = items[idx]
    _save_registry(registry_path, items)
    return entry


def mark_trained(registry_path: Path, params_hash: str, *, model_output_dir: Optional[str] = None) -> None:
    items = _load_registry(registry_path)
    idx = find_entry(items, params_hash)
    if idx is None:
        return
    items[idx]["status"] = "trained"
    items[idx]["completed_train"] = True
    if model_output_dir:
        items[idx].setdefault("runs", {}).setdefault("train", {}).update({"output_dir": model_output_dir})
    _save_registry(registry_path, items)


def upsert_eval_results(
    registry_path: Path,
    params_hash: str,
    suite: str,
    results: Dict[str, Dict[str, float]],
    averages: Dict[str, float],
    task_status: Dict[str, str],
    model_path: Optional[str] = None,
) -> None:
    items = _load_registry(registry_path)
    idx = find_entry(items, params_hash)
    if idx is None:
        # Create a minimal stub if training didn't register
        eval_entry = {
            "results": results,
            "averages": averages,
            "task_status": task_status,
        }
        if model_path:
            eval_entry["model_evaluated"] = model_path
        items.append({
            "id": params_hash,
            "params": {},
            "status": "evaluated",
            "completed_train": False,
            "completed_eval": True,
            "runs": {},
            "evals": {suite: eval_entry},
        })
    else:
        eval_entry = {
            "results": results,
            "averages": averages,
            "task_status": task_status,
        }
        if model_path:
            eval_entry["model_evaluated"] = model_path
        items[idx].setdefault("evals", {})[suite] = eval_entry
        items[idx]["status"] = "evaluated"
        items[idx].setdefault("completed_train", False)
        items[idx]["completed_eval"] = True
    _save_registry(registry_path, items)


def exists(registry_path: Path, params_hash: str) -> bool:
    # Only treat as existing (to block a rerun) if the training for that params hash
    # has completed successfully. Incomplete runs should not block a new run.
    items = _load_registry(registry_path)
    idx = find_entry(items, params_hash)
    if idx is None:
        return False
    entry = items[idx]
    return bool(entry.get("completed_train", False))
