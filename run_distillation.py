import argparse
import os
import random
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from sampledkd.config import TrainingConfig
from sampledkd.run_registry import (
    compute_params_hash,
    upsert_run_start,
    mark_trained,
    exists as registry_exists,
    normalize_params,
    get_entry,
)
from sampledkd.data.dataset import AIMEJsonl, DistillCollator, PackedTokenDataset, PromptDataset
from sampledkd.models.loader import load_model
from sampledkd.distill import Distiller
from sampledkd.distill._mixins.amp_oom import AmpOomMixin
from sampledkd.training.entrypoint_utils import load_teacher_with_fallback, load_fineweb_subset
from sampledkd.training.distributed import (
    create_distributed_sampler,
    distributed_barrier,
    distributed_broadcast_object_list,
    destroy_distributed,
    is_rank0,
    setup_distributed_context,
)
from sampledkd.training.offline_cache import (
    execute_cache_plan,
    plan_offline_cache,
)

# Import logging utils with fallback
try:
    from sampledkd.logging.wandb_utils import create_training_combined_logger
except ImportError:
    def create_training_combined_logger(*args, **kwargs):
        return None
def parse_args_to_config() -> TrainingConfig:
    """Parse command line arguments and create TrainingConfig."""
    parser = argparse.ArgumentParser(description="Entropy-guided KD for LLMs")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--student_quant_bits", type=int, choices=[4, 8], default=None,
                        help="Optionally quantize student for memory (not typical during training)")
    parser.add_argument("--distill_type", choices=[
        "vanilla",
        "top-k-tok",
        "random",
        "bucket",
        "pos-rs-kd",
        "linucb",
    ], default="vanilla")
    parser.add_argument("--k_percent", type=int, default=20, help="for top-k-tok and random")
    parser.add_argument(
        "--normalize_topk_by_length",
        action="store_true",
        default=False,
        help="When set, top-k token quota is based on the batch-average valid length instead of per-example length",
    )
    parser.add_argument("--kd_temperature", type=float, default=2.0, help="Unified KD temperature for teacher/student log-softmax and T^2 scaling")
    parser.add_argument("--entropy_approx_temperature", type=float, default=2.0, help="Temperature for offline entropy approximation (and RS-KD proposal)")
    # KD temperature annealing controls
    parser.add_argument("--anneal_kd_temperature", action="store_true", default=False,
                        help="Enable annealing of kd_temperature during training")
    parser.add_argument("--kd_temperature_start", type=float, default=2.0,
                        help="Starting KD temperature when annealing is enabled")
    parser.add_argument("--kd_temperature_end", type=float, default=1.0,
                        help="Final KD temperature when annealing is enabled")
    parser.add_argument("--kd_hold_frac", type=float, default=0.6,
                        help="Fraction of total updates to hold at start temperature before linear decay")
    # RS-KD (position-sampling) hyperparams
    parser.add_argument("--rs_alpha", type=float, default=1.0,
                        help="Exponent on entropy for sampling dist: q(i) ∝ H_i^alpha (alpha∈[0,∞))")
    parser.add_argument("--rs_floor", type=float, default=1e-6,
                        help="Minimum probability floor to avoid huge weights / degeneracy")
    parser.add_argument(
        "--bucket_lower_percent",
        type=int,
        default=int(os.environ.get("BUCKET_LOWER_PERCENT", 70)),
        help="For bucket mode: lower bound percentile (skip bottom X%)",
    )
    parser.add_argument(
        "--bucket_upper_percent",
        type=int,
        default=int(os.environ.get("BUCKET_UPPER_PERCENT", 80)),
        help="For bucket mode: upper bound percentile (skip top Y%)",
    )
    parser.add_argument("--score_token_selection", action="store_true", default=False,
                        help="Rank tokens by composite score (entropy + student CE + KL) when selecting top-k/bucket tokens")
    parser.add_argument("--score_normalize", choices=["none", "z", "minmax"], default="z",
                        help="Normalization applied per example to score components before weighting")
    parser.add_argument("--score_entropy_weight", type=float, default=1.0,
                        help="Weight for teacher entropy component in score-based KD")
    parser.add_argument("--score_ce_weight", type=float, default=1.0,
                        help="Weight for student cross-entropy component in score-based KD")
    parser.add_argument("--score_kl_weight", type=float, default=1.0,
                        help="Weight for teacher-student KL component in score-based KD")
    parser.add_argument("--bandit_alpha", type=float, default=1.0,
                        help="Exploration coefficient for LinUCB contextual bandit")
    parser.add_argument("--bandit_lambda", type=float, default=1.0,
                        help="L2 regularization strength for LinUCB covariance matrix")
    parser.add_argument("--bandit_threshold", type=float, default=0.0,
                        help="Minimum UCB score required for LinUCB to keep a token")
    parser.add_argument("--bandit_min_tokens", type=int, default=1,
                        help="Force LinUCB to keep at least this many tokens per example")
    parser.add_argument("--bandit_max_tokens", type=int, default=None,
                        help="Optional cap on the number of tokens LinUCB can keep per example")
    parser.add_argument("--bandit_device", type=str, default="cpu",
                        help="Device to maintain LinUCB statistics (cpu or cuda)")
    parser.add_argument("--bandit_reward_clip", type=float, default=5.0,
                        help="Absolute clip value applied to KL improvement rewards before LinUCB updates")
    parser.add_argument("--enable_ce", action="store_true", default=True, 
                        help="Enable cross-entropy loss in addition to KD loss")
    parser.add_argument("--alpha_ce", type=float, default=0.1,
                        help="Weight for cross-entropy loss (vs KD loss). Total loss = (1-alpha_ce)*L_KD + alpha_ce*L_CE")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--prompt_col", type=str, default=None,
                        help="name of text prompt column for HF datasets")
    parser.add_argument("--answer_col", type=str, default=None,
                        help="name of answer column for HF datasets")
    parser.add_argument("--fineweb_tokens", type=int, default=50_000_000,
                        help="Token budget when streaming FineWeb-Edu (used when datasets[0] == 'fineweb')")
    parser.add_argument(
        "--disable_packing",
        dest="enable_packing",
        action="store_false",
        help="Disable sequence packing into fixed-length windows",
    )
    parser.add_argument(
        "--enable_packing",
        dest="enable_packing",
        action="store_true",
        help="Enable sequence packing into fixed-length windows",
    )
    parser.set_defaults(enable_packing=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--max_seq_len", type=int, default=512)  # to save memory
    parser.add_argument("--lr", type=float, default=1e-5)  # Reduced from 5e-5
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensorboard_dir", type=str, default="tb", 
                        help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_steps", type=int, default=500,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--keep_checkpoints", type=int, default=3,
                        help="Number of recent checkpoints to keep")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="(Optional) HF dataset config, e.g. for gsm8k use '--dataset_config main' or 'socratic'"
    )
    parser.add_argument("--offline_cache", action="store_true", default=True,
                        help="Enable offline caching mode: automatically create/use teacher cache for entropy approximation and vocab RS-KD.")
    parser.add_argument("--no_offline_cache", dest="offline_cache", action="store_false",
                        help="Disable offline caching mode (use online teacher forward pass).")
    parser.add_argument("--offline_cache_dir", type=str, default=None,
                        help="Where to store/read the offline teacher cache (defaults under output_dir).")
    parser.add_argument("--entropy_approx_m", type=int, default=12,
                        help="Top-k for truncated-entropy approximation, m=12 by default.")
    parser.add_argument("--rs_vocab_samples", type=int, default=12,
                        help="How many vocab tokens to sample per position for RS-KD. 36 bytes per position")
    parser.add_argument("--rs_vocab_beta", type=float, default=1.0,
                        help="Proposal exponent: q ∝ p^beta (beta=1 is proportional to p).")
    # Sampled softmax elimination (only in cached RS-KD path)
    parser.add_argument("--eliminate_softmax", action="store_true", default=False,
                        help="Eliminate full-vocab softmax in cached RS-KD path using sampled softmax and importance correction")
    parser.add_argument("--no_eliminate_softmax", dest="eliminate_softmax", action="store_false",
                        help="Disable softmax elimination (force full-vocab softmax).")

    parser.add_argument("--sampled_softmax_negatives", type=int, default=1500,
                        help="Number of uniform negative samples per position when --eliminate_softmax is set")
    parser.add_argument("--ddp_offline", action="store_true", default=False,
                        help="Enable distributed (torchrun) offline-mode training across multiple GPUs")
    parser.add_argument("--no_ddp_offline", dest="ddp_offline", action="store_false",
                        help="Disable distributed offline mode (single-process run)")
    # Global-Level Selection (GLS) over tokens — only impacts top-k-tok when enabled
    parser.add_argument("--gls_enabled", action="store_true", default=False,
                        help="Enable global-level selection FIFO queue (only impacts top-k-tok)")
    parser.add_argument("--gls_queue_size", type=int, default=30000,
                        help="Capacity of GLS FIFO queue for computing global threshold")
    parser.add_argument("--gls_log_threshold", action="store_true", default=False,
                        help="Log the GLS threshold each time it's computed")
    # Unified upsert registry controls
    parser.add_argument("--runs_registry", type=str, default="results/runs.json",
                        help="Path to the unified runs JSON registry (a JSON list).")
    parser.add_argument("--override", action="store_true", default=False,
                        help="If set, run even if an identical-params hash already exists in the registry.")
    # Reproducibility
    default_seed = int(os.environ.get("SEED", "1337"))
    default_det = bool(int(os.environ.get("DETERMINISTIC", "0")))
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", default=default_det,
                        help="Enable deterministic algorithms (may slow down, sets cudnn.deterministic and use_deterministic_algorithms)")
    args = parser.parse_args()
    
    # Convert argparse Namespace to TrainingConfig
    return TrainingConfig(**vars(args))


def main():
    """Main training function using Pydantic configuration."""
    config = parse_args_to_config()

    ddp_ctx = setup_distributed_context(config)
    ddp_world_size = ddp_ctx.world_size
    ddp_rank = ddp_ctx.rank
    ddp_local_rank = ddp_ctx.local_rank
    is_main_rank = ddp_ctx.is_main_rank
    
    # Print all CLI parameters at startup
    print("[launch] python", " ".join(sys.argv), flush=True)
    if is_main_rank:
        print("=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        try:
            params_dict = config.model_dump()
        except Exception:
            params_dict = vars(config)
        
        for key, value in sorted(params_dict.items()):
            print(f"  {key:30s} = {value}")
        print("=" * 80)
        print()
    else:
        try:
            params_dict = config.model_dump()
        except Exception:
            params_dict = vars(config)

    # global seeding
    seed_offset = config.seed + ddp_rank
    random.seed(seed_offset)
    np.random.seed(seed_offset)
    torch.manual_seed(seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_offset)

    # Optional deterministic mode
    if getattr(config, "deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    # Prefer node-local tmp to avoid NFS .nfs tombstones on cleanup
    # If SLURM provides a local scratch (e.g., /dev/shm), use it; fall back to repo tmp
    node_tmp = os.environ.get("TMPDIR")
    if not node_tmp:
        shm_candidate = f"/dev/shm/{os.environ.get('USER', 'user')}.sampledkd.{os.environ.get('SLURM_JOB_ID', 'local')}"
        try:
            os.makedirs(shm_candidate, exist_ok=True)
            node_tmp = shm_candidate
        except Exception:
            node_tmp = f"/home/joberant/NLP_2425b/{os.environ.get('USER', 'user')}/sampledkd/tmp"
        os.environ["TMPDIR"] = node_tmp
    # Point HF caches to tmp (can still hit shared cache via symlink if needed)
    os.environ.setdefault("HF_HOME", os.path.join(node_tmp, "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Set CUDA memory management settings for better memory efficiency
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear any cached memory
        
    # Speed optimizations (safe) or deterministic fallback
    if getattr(config, "deterministic", False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ----------------- runs registry preflight -----------------
    registry_path = Path(getattr(config, "runs_registry", "results/runs.json"))
    try:
        params_dict = config.model_dump()
    except Exception:
        params_dict = vars(config)
    params_hash = compute_params_hash(params_dict)

    if is_main_rank:
        if registry_exists(registry_path, params_hash) and not getattr(config, "override", False):
            entry = get_entry(registry_path, params_hash)
            completed_eval = bool(entry and entry.get("completed_eval"))
            runs_info = entry.get("runs", {}) if entry else {}
            existing_output_dir = (runs_info.get("train") or {}).get("output_dir") if runs_info else None
            print(f"[registry] Run with identical parameters already exists (id={params_hash}). Use --override to force rerun. Exiting gracefully.")
            needs_eval = not completed_eval
            meta_output_dir = existing_output_dir or ""
            print(f"[registry] duplicate params_hash={params_hash} needs_eval={int(needs_eval)} output_dir={meta_output_dir}")
            sys.exit(11 if needs_eval else 10)

        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "local")
        experiment_name = f"distill-{config.distill_type}-{current_date}_{job_id}"
        if config.distill_type in {"top-k-tok", "random"}:
            experiment_name += f"_k={config.k_percent}"
        elif config.distill_type == "bucket":
            experiment_name += f"_bucket={config.bucket_lower_percent}-{config.bucket_upper_percent}"

        cli_args = " ".join(shlex.quote(arg) for arg in sys.argv)
        upsert_run_start(
            registry_path,
            params_dict,
            experiment_name=experiment_name,
            job_id=job_id,
            model_output_dir=config.output_dir,
            launch_args=cli_args,
        )
    else:
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "local")
        experiment_name = f"distill-{config.distill_type}-{current_date}_{job_id}"
        if config.distill_type in {"top-k-tok", "random"}:
            experiment_name += f"_k={config.k_percent}"
        elif config.distill_type == "bucket":
            experiment_name += f"_bucket={config.bucket_lower_percent}-{config.bucket_upper_percent}"

    # Persist normalized params and hash alongside model outputs for downstream eval
    if is_main_rank:
        try:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            params_out = Path(config.output_dir) / "run_params.json"
            with open(params_out, "w", encoding="utf-8") as f:
                import json as _json
                _json.dump({
                    "id": params_hash,
                    "params": normalize_params(params_dict),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[registry] Warning: failed to write run_params.json: {e}")

    # ----------------- teacher / student device planning -----------------
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count == 0:
        raise RuntimeError("No CUDA devices available.")

    total_vram_gb = 0
    for i in range(device_count):
        vram_bytes = torch.cuda.get_device_properties(i).total_memory
        total_vram_gb += vram_bytes / (1024**3)
    if is_main_rank:
        print(
            "Success: Detected "
            + str(device_count)
            + " GPUs with "
            + str(round(total_vram_gb, 1))
            + " GB total VRAM available."
        )

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        physical = [int(x) for x in cvd.split(",") if x != ""]
        local_avail = list(range(len(physical)))
    else:
        physical = list(range(device_count))
        local_avail = list(range(device_count))

    if config.ddp_offline:
        if ddp_local_rank >= len(local_avail):
            raise RuntimeError(f"LOCAL_RANK={ddp_local_rank} exceeds available GPUs {local_avail}")
        student_local = ddp_local_rank
        student_device = torch.device(f"cuda:{student_local}")
        teacher_locals = [student_local]
        teacher_student_exclusion = None
    else:
        student_local = local_avail[1] if len(local_avail) >= 2 else local_avail[0]
        student_device = torch.device(f"cuda:{student_local}")
        teacher_locals = [g for g in local_avail if g != student_local]
        if teacher_locals:
            teacher_student_exclusion = student_local
        else:
            teacher_locals = [student_local]
            teacher_student_exclusion = None
    
    if is_main_rank:
        print(f"CUDA_VISIBLE_DEVICES: {cvd}")
        print(f"Available GPUs (local indices): {local_avail}")
    print(f"[rank {ddp_rank}] Student GPU (local): {student_local}")
    print(f"[rank {ddp_rank}] Teacher GPUs (local): {teacher_locals}")

    # ---- Tokenizer and dataset preparation (before teacher load) ----
    tok = AutoTokenizer.from_pretrained(
        config.teacher_model,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=False,
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if all(p.endswith(".jsonl") for p in config.datasets):
        aime = AIMEJsonl([Path(p) for p in config.datasets])
        raw_texts = [aime[i]["prompt"] for i in range(len(aime))]
    else:
        if config.datasets[0].lower() == "fineweb":
            budget = int(getattr(config, "fineweb_tokens", 50_000_000))
            print(f"Loading FineWeb-Edu subset with {budget:,} tokens, seed {config.seed}")
            cached_examples = load_fineweb_subset(
                tok,
                max_tokens=budget,
                seed=config.seed,
                max_seq_len=config.max_seq_len,
                packing_enabled=bool(getattr(config, "enable_packing", True)),
            )
            raw_texts = [ex["prompt"] for ex in cached_examples]
        else:
            print(f"Loading Hugging Face dataset: {config.datasets[0]}")
            hf_dataset = load_dataset(config.datasets[0], config.dataset_config)["train"] if config.dataset_config \
                else load_dataset(config.datasets[0])["train"]
            prompt_col = config.prompt_col or "prompt"
            answer_col = config.answer_col
            print(f"Using columns - prompt: '{prompt_col}', answer: '{answer_col}'")
            raw_texts = []
            for ex in hf_dataset:
                prompt_text = ex[prompt_col]
                if answer_col is not None and answer_col in ex and ex[answer_col] is not None:
                    raw_texts.append(f"{prompt_text}\n{ex[answer_col]}")
                else:
                    raw_texts.append(prompt_text)

    if getattr(config, "enable_packing", True):
        dataset = PackedTokenDataset(raw_texts, tok, config.max_seq_len)
        if len(dataset) == 0:
            raise RuntimeError("Packed dataset is empty. Reduce max_seq_len or provide longer input texts.")
    else:
        dataset = PromptDataset(raw_texts)
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty. Provide non-empty input texts.")

    packed_dataset = dataset

    collate = DistillCollator(tok, config.max_seq_len)
    gen = torch.Generator()
    gen.manual_seed(config.seed)

    def _seed_worker(worker_id: int):
        base = int(config.seed)
        np.random.seed(base + worker_id)
        random.seed(base + worker_id)

    sampler = create_distributed_sampler(
        dataset,
        config=config,
        seed=seed_offset,
        shuffle=True,
        drop_last=False,
    )

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
        generator=gen,
        worker_init_fn=_seed_worker,
    )

    dataset_size = len(dataset)

    teacherless_modes = {"vanilla", "top-k-tok", "bucket", "random", "linucb"}
    cache_plan = plan_offline_cache(
        config,
        tok,
        dataset_size,
        is_main_rank=is_main_rank,
        teacherless_modes=teacherless_modes,
    )
    cache_ready = cache_plan.cache_ready
    teacher_required = cache_plan.teacher_required
    teacher_rank0_only = cache_plan.teacher_rank0_only

    teacher = None
    teacher_inputs_device = torch.device("cpu")
    if teacher_required:
        load_here = not teacher_rank0_only or is_rank0()
        if load_here:
            if is_main_rank or not config.ddp_offline:
                print("Loading teacher with GPU-first fallback...", flush=True)
            teacher, _, teacher_inputs_device = load_teacher_with_fallback(
                model_name=config.teacher_model,
                prefer_gpus=teacher_locals,
                student_gpu=teacher_student_exclusion,
            )
        elif is_main_rank:
            print("Teacher will be hosted on rank 0; skipping local load on this rank.")

    # Optionally pre-build offline cache before loading students to free teacher VRAM
    cache_result = execute_cache_plan(
        cache_plan,
        config=config,
        tok=tok,
        packed_dataset=packed_dataset,
        collate_fn=collate,
        teacher=teacher,
        teacher_inputs_device=teacher_inputs_device,
        seed_offset=seed_offset,
        sanitize_logits_fn=AmpOomMixin._sanitize_logits,
        is_main_rank=is_main_rank,
        teacherless_modes=teacherless_modes,
    )

    teacher = cache_result.teacher
    teacher_inputs_device = cache_result.teacher_inputs_device
    cache_ready = cache_result.cache_ready
    teacher_required = cache_result.teacher_required
    teacher_rank0_only = cache_result.teacher_rank0_only


    teacher_device_str = str(teacher_inputs_device)

    if getattr(config, "ddp_offline", False):
        obj = [teacher_device_str] if is_rank0() else [None]
        obj = distributed_broadcast_object_list(obj, src=0)
        teacher_device_str = obj[0] or "cpu"
        teacher_inputs_device = torch.device(teacher_device_str)
        distributed_barrier()
    else:
        teacher_inputs_device = torch.device(teacher_device_str)

    setattr(config, "_teacher_device_str", teacher_device_str)
    setattr(config, "_teacher_rank0_owner", teacher_rank0_only)
    setattr(config, "_teacher_required", teacher_required)

    print("Loading student on its own GPU...", flush=True)
    student = load_model(
        config.student_model,
        device_map=student_local,
        quant_bits=config.student_quant_bits,
    )

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.config.use_cache = False

    student.train()
    student.config.use_cache = False

    if teacher is not None or teacher_required:
        print(f"Teacher device: {teacher_inputs_device}")
    else:
        if is_main_rank:
            print("Teacher load skipped; using offline cache exclusively for teacher signals.")
    print(f"Student device: {student_device}")

    if student_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(student_device.index)} (device {student_device.index})")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(student_device) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(student_device) / 1024**3:.2f} GB")
    else:
        print(f"Using device: {student_device}")

    # Initialize logging with experiment name (built above)
    
    # Initialize combined logger (W&B + TensorBoard)
    combined_logger = None
    if is_main_rank:
        combined_logger = create_training_combined_logger(
            config, experiment_name, tensorboard_dir=config.tensorboard_dir
        )

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        config=config,  # Pass the entire config instead of individual args
        teacher_device=teacher_inputs_device,
        student_device=student_device,
        logger=combined_logger,  # Use new combined logger
    )

    distiller.train(epochs=config.epochs)

    if getattr(config, "ddp_offline", False):
        distributed_barrier()

    if is_main_rank:
        if combined_logger:
            try:
                combined_logger.log_artifact(config.output_dir, f"student_model_{experiment_name}", "model")
                combined_logger.finish()
            except Exception:
                pass

        model_to_save = getattr(distiller, "student", student)
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        print("Saving student to", config.output_dir)
        model_to_save.save_pretrained(config.output_dir)
        tok.save_pretrained(config.output_dir)

        try:
            mark_trained(registry_path, params_hash, model_output_dir=config.output_dir)
        except Exception as e:
            print(f"[registry] Failed to mark trained: {e}")

    if getattr(config, "ddp_offline", False):
        distributed_barrier()
        destroy_distributed()


if __name__ == "__main__":
    main()
