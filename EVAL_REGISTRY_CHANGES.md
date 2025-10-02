# Evaluation Registry Changes

## Summary

Modified the evaluation system to:
1. **Record which model was evaluated** in `runs.json`
2. **Remove timestamps** from the registry to keep it cleaner and more focused on results

## Changes Made

### 1. `ekd/run_registry.py`

#### Updated `upsert_eval_results()`
- **Added** `model_path` parameter to record which model was evaluated
- **Removed** all timestamp fields (`created_at`, `last_update`, `updated_at`)
- Now stores `model_evaluated` field in each eval entry when model_path is provided

**Before:**
```python
def upsert_eval_results(
    registry_path: Path,
    params_hash: str,
    suite: str,
    results: Dict[str, Dict[str, float]],
    averages: Dict[str, float],
    task_status: Dict[str, str],
) -> None:
    # ... stored with "updated_at": now
```

**After:**
```python
def upsert_eval_results(
    registry_path: Path,
    params_hash: str,
    suite: str,
    results: Dict[str, Dict[str, float]],
    averages: Dict[str, float],
    task_status: Dict[str, str],
    model_path: Optional[str] = None,  # NEW
) -> None:
    # ... stores "model_evaluated": model_path (no timestamps)
```

#### Updated `upsert_run_start()`
- **Removed** `created_at`, `last_update`, and `started_at` timestamp fields
- Schema now focused on run metadata without temporal information

#### Updated `mark_trained()`
- **Removed** `last_update` timestamp field
- Only updates status and completion flags

### 2. `ekd/evaluations/eval.py`

#### Updated evaluation result recording
- **Added** logic to determine model path (handles both local dirs and HF hub IDs)
- **Passes** `model_path` to all `upsert_eval_results()` calls
- Model path captured from:
  - Local HF directory path, OR
  - HuggingFace hub ID (when using `--from_hf`)

**Implementation:**
```python
# Prepare model_path for registry (use string representation)
model_path_str = str(model_dir) if not args.from_hf else base_model_dir

# Pass to registry
upsert_eval_results(
    Path(args.runs_registry), 
    params_hash, 
    args.suite, 
    merged, 
    averages, 
    task_status, 
    model_path=model_path_str  # NEW
)
```

## New runs.json Schema

### Before (with timestamps):
```json
{
  "id": "abc123...",
  "params": {...},
  "created_at": "2025-10-02T06:08:27.871784Z",
  "last_update": "2025-10-02T16:23:39.287581Z",
  "status": "evaluated",
  "runs": {
    "train": {
      "experiment": "distill-vanilla-...",
      "job_id": "15480",
      "output_dir": "/path/to/model",
      "started_at": "2025-10-02T06:08:27.871784Z"
    }
  },
  "evals": {
    "light": {
      "results": {...},
      "averages": {...},
      "task_status": {...},
      "updated_at": "2025-10-02T16:23:39.287581Z"
    }
  }
}
```

### After (cleaner, with model info):
```json
{
  "id": "abc123...",
  "params": {...},
  "status": "evaluated",
  "runs": {
    "train": {
      "experiment": "distill-vanilla-...",
      "job_id": "15480",
      "output_dir": "/path/to/model"
    }
  },
  "evals": {
    "light": {
      "results": {...},
      "averages": {...},
      "task_status": {...},
      "model_evaluated": "/path/to/evaluated/model"
    }
  },
  "completed_train": false,
  "completed_eval": true
}
```

## Key Improvements

### 1. Model Traceability
- **Before**: No record of which specific model directory/checkpoint was evaluated
- **After**: `model_evaluated` field captures exact model path or HF hub ID

**Use cases**:
- Track evaluations of different checkpoints from same training run
- Distinguish between evaluations of exported .pt vs original HF models
- Know which HF hub model was benchmarked

### 2. Cleaner JSON
- **Before**: Multiple redundant timestamp fields (`created_at`, `last_update`, `updated_at`, `started_at`)
- **After**: No timestamps, focus on metadata and results

**Benefits**:
- Easier to read and compare entries
- Smaller file size
- No temporal coupling (results are timeless facts)
- Simpler diffs when comparing runs

### 3. Backward Compatibility
- Existing code that reads `runs.json` will still work
- New fields are optional (`model_evaluated` only added when available)
- Completion flags (`completed_train`, `completed_eval`) unchanged

## Example Usage

### Evaluate a local HF model:
```bash
python -m ekd.evaluations.eval /path/to/model --suite light
```

Result in `runs.json`:
```json
{
  "evals": {
    "light": {
      "model_evaluated": "/path/to/model",
      "results": {...}
    }
  }
}
```

### Evaluate a HuggingFace hub model:
```bash
python -m ekd.evaluations.eval Qwen/Qwen3-8B --from_hf --suite light
```

Result in `runs.json`:
```json
{
  "evals": {
    "light": {
      "model_evaluated": "Qwen/Qwen3-8B",
      "results": {...}
    }
  }
}
```

### Evaluate a .pt checkpoint:
```bash
python -m ekd.evaluations.eval checkpoint_epoch1_step1000.pt --suite heavy
```

Result in `runs.json`:
```json
{
  "evals": {
    "heavy": {
      "model_evaluated": "eval_runs/exports/export_checkpoint_epoch1_step1000",
      "results": {...}
    }
  }
}
```

## Migration Notes

### Existing runs.json files
- Old entries with timestamps will continue to work
- New evaluations will not add timestamps
- You can manually clean up old timestamps if desired, or leave them (they're ignored by the code)

### Timestamp removal rationale
1. **Not actionable**: Timestamps don't change interpretation of results
2. **Version control**: Git already tracks when files changed
3. **Noise**: Made diffs harder to read when comparing experiments
4. **Focus**: Registry is about _what_ was run, not _when_

## Testing

To verify the changes work correctly:

```bash
# Run a quick evaluation
python -m ekd.evaluations.eval <your_model_path> --suite light

# Check runs.json
cat results/runs.json | jq '.[] | {id, model_evaluated: .evals.light.model_evaluated}'

# Should show:
# {
#   "id": "abc123...",
#   "model_evaluated": "<your_model_path>"
# }
```

## Files Modified

1. `ekd/run_registry.py` - Registry management functions
2. `ekd/evaluations/eval.py` - Evaluation orchestrator
3. This document - `EVAL_REGISTRY_CHANGES.md`

---

**Summary**: Evaluation results now include which model was evaluated, and all timestamps have been removed for a cleaner, more focused registry.
