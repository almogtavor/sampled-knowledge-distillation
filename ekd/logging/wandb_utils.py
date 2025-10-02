#!/usr/bin/env python3
"""
Logging Utilities for EKD Project

This module provides utilities for logging training runs and evaluations to both
Weights & Biases and TensorBoard.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
import re

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

    if TYPE_CHECKING:
        # For type checkers/linting only; avoids runtime import cycles
        from ekd.config import TrainingConfig


class WandBLogger:
    """W&B logging utility for EKD project."""
    
    def __init__(
        self,
        project: str = "selective-entropy-knowledge-distillation",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        notes: Optional[str] = None,
        resume: str = "allow",
        run_id: Optional[str] = None,
    ):
        # Allow environment to override
        self.project = os.getenv("WANDB_PROJECT", project)
        self.entity  = os.getenv("WANDB_ENTITY", entity or "") or None
        self.group   = os.getenv("WANDB_GROUP", group)
        self.job_type= os.getenv("WANDB_JOB_TYPE", job_type)
        self.notes   = os.getenv("WANDB_NOTES", notes)
        resume_env   = os.getenv("WANDB_RESUME", resume)
        run_id_env   = os.getenv("WANDB_RUN_ID", run_id)
        self.run = None

        def _is_rank0():
            return os.getenv("RANK") in (None, "0") and os.getenv("LOCAL_RANK") in (None, "0") and os.getenv("SLURM_PROCID") in (None, "0")

        offline = os.getenv("WANDB_MODE", "online") == "offline" or os.getenv("WANDB_DISABLED", "").lower() in ("true", "1")
        self.enabled = WANDB_AVAILABLE and _is_rank0() and not offline

        if self.enabled:
            try:
                # Login if key present; else rely on ~/.netrc
                if os.getenv("WANDB_API_KEY"):
                    wandb.login(key=os.getenv("WANDB_API_KEY"))
                settings = wandb.Settings(start_method=os.getenv("WANDB_START_METHOD", "thread"))
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    name=name,
                    config=config or {},
                    tags=tags or [],
                    group=self.group,
                    job_type=self.job_type,
                    notes=self.notes,
                    resume=resume_env,
                    id=run_id_env,
                    settings=settings,
                    reinit=True,
                )
                print(f"W&B logging initialized: {self.run.get_url()}")
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")
                self.enabled = False
        else:
            if not WANDB_AVAILABLE:
                print("W&B not available: pip install wandb")
            else:
                print("W&B disabled (non-rank0 or WANDB_MODE=offline/WANDB_DISABLED=true)")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if self.enabled and self.run:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                print(f"Error logging to W&B: {e}")

    def log_artifact(self, artifact_path: str, name: str, type: str, description: str = "") -> None:
        """Log an artifact to W&B."""
        if self.enabled and self.run:
            try:
                # W&B artifact name may only contain [A-Za-z0-9_.-]
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
                artifact = wandb.Artifact(name=safe_name, type=type, description=description)
                if os.path.isdir(artifact_path):
                    artifact.add_dir(artifact_path)
                else:
                    artifact.add_file(artifact_path)
                self.run.log_artifact(artifact)
                print(f"Artifact '{safe_name}' logged to W&B")
            except Exception as e:
                print(f"Error logging artifact to W&B: {e}")

    def log_table(self, table_name: str, columns: list, data: list) -> None:
        """Log a table to W&B."""
        if self.enabled and self.run and WANDB_AVAILABLE:
            try:
                table = wandb.Table(columns=columns, data=data)
                self.run.log({table_name: table})
            except Exception as e:
                print(f"Error logging table to W&B: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self.run:
            try:
                self.run.finish()
                print("W&B run finished successfully")
            except Exception as e:
                print(f"Error finishing W&B run: {e}")


class TensorBoardLogger:
    """TensorBoard logging utility for EKD project."""
    
    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.enabled:
            try:
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                print(f"TensorBoard logging enabled at {self.log_dir}")
            except Exception as e:
                print(f"Failed to initialize TensorBoard: {e}")
                self.enabled = False
        else:
            print("TensorBoard not available")
    
    def log_scalar(self, name: str, value: float, step: int = 0) -> None:
        """Log a scalar value to TensorBoard."""
        if self.enabled and self.writer:
            try:
                self.writer.add_scalar(name, value, step)
            except Exception as e:
                print(f"Error logging scalar to TensorBoard: {e}")
    
    def log_scalars(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log multiple scalar values to TensorBoard."""
        if self.enabled and self.writer:
            try:
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, step)
            except Exception as e:
                print(f"Error logging scalars to TensorBoard: {e}")
    
    def flush(self) -> None:
        """Flush TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.flush()
            except Exception as e:
                print(f"Error flushing TensorBoard: {e}")
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.close()
                print("TensorBoard logger closed successfully")
            except Exception as e:
                print(f"Error closing TensorBoard: {e}")


class CombinedLogger:
    """Combined logger that handles both W&B and TensorBoard logging."""
    
    def __init__(
        self,
        wandb_logger: Optional[WandBLogger] = None,
        tensorboard_logger: Optional[TensorBoardLogger] = None,
    ):
        """Initialize combined logger.
        
        Args:
            wandb_logger: W&B logger instance
            tensorboard_logger: TensorBoard logger instance
        """
        self.wandb_logger = wandb_logger
        self.tensorboard_logger = tensorboard_logger
        self.global_step = 0
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to both W&B and TensorBoard."""
        if step is None:
            step = self.global_step
            
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step)
        
        if self.tensorboard_logger:
            self.tensorboard_logger.log_scalars(metrics, step)
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar to both W&B and TensorBoard."""
        if step is None:
            step = self.global_step
            
        if self.wandb_logger:
            self.wandb_logger.log({name: value}, step)
        
        if self.tensorboard_logger:
            self.tensorboard_logger.log_scalar(name, value, step)
    
    def log_artifact(self, artifact_path: str, name: str, artifact_type: str = "model") -> None:
        """Log artifact to W&B."""
        if self.wandb_logger:
            self.wandb_logger.log_artifact(artifact_path, name, artifact_type)
    
    def log_table(self, table_name: str, columns: list, data: list) -> None:
        """Log table to W&B."""
        if self.wandb_logger:
            self.wandb_logger.log_table(table_name, columns, data)
    
    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1
    
    def flush(self) -> None:
        """Flush both loggers."""
        if self.tensorboard_logger:
            self.tensorboard_logger.flush()
    
    def finish(self) -> None:
        """Finish and close both loggers."""
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        if self.tensorboard_logger:
            self.tensorboard_logger.close()


def create_training_logger(config, experiment_name: Optional[str] = None) -> WandBLogger:
    """Create a W&B logger for training runs."""
    if experiment_name is None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "local")
        experiment_name = f"distill-{config.distill_type}-{current_date}_{job_id}"
        if config.distill_type == "top-k-tok" or config.distill_type == "random":
            experiment_name += f"_k={config.k_percent}"
        elif config.distill_type == "bucket":
            experiment_name += f"_bucket={config.bucket_lower_percent}-{config.bucket_upper_percent}"
    
    wandb_config = {
        # Training config
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "distill_type": config.distill_type,
        "k_percent": config.k_percent,
        "bucket_lower_percent": getattr(config, 'bucket_lower_percent', None),
        "bucket_upper_percent": getattr(config, 'bucket_upper_percent', None),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_seq_len": config.max_seq_len,
        "lr": config.lr,
        "datasets": config.datasets,
        # System info
        "job_id": os.getenv("SLURM_JOB_ID", "local"),
        "experiment_name": experiment_name,
    }

    # If training uses FineWeb, log the token budget when available
    try:
        if isinstance(getattr(config, 'datasets', None), list) and 'fineweb' in config.datasets:
            fw_tokens = getattr(config, 'fineweb_tokens', None)
            if fw_tokens is not None:
                wandb_config["fineweb_tokens"] = int(fw_tokens)
    except Exception:
        pass
    
    tags = [
        config.distill_type,
        f"k={config.k_percent}" if config.distill_type != "vanilla" else "vanilla",
        "training"
    ]
    
    # Add mode-specific tags
    if config.distill_type == "top-k-tok":
        tags.append(f"k={config.k_percent}")
    elif config.distill_type == "bucket":
        tags.append(f"bucket={config.bucket_lower_percent}-{config.bucket_upper_percent}")
    elif config.distill_type == "vanilla":
        tags.append("all-tokens")
    # Optional tag to surface FineWeb token budget in the UI
    try:
        if isinstance(getattr(config, 'datasets', None), list) and 'fineweb' in config.datasets:
            fw_tokens = getattr(config, 'fineweb_tokens', None)
            if fw_tokens is not None:
                tags.append(f"fineweb_tokens={int(fw_tokens)}")
    except Exception:
        pass
    
    return WandBLogger(
        project=getattr(config, 'wandb_project', 'selective-entropy-knowledge-distillation'),
        entity=getattr(config, 'wandb_entity', None),
        name=experiment_name,
        config=wandb_config,
        tags=tags,
        group=os.getenv("WANDB_GROUP"),
        job_type="train",
        notes=os.getenv("WANDB_NOTES"),
        resume=os.getenv("WANDB_RESUME", "allow"),
        run_id=os.getenv("WANDB_RUN_ID"),
    )


def create_evaluation_logger(base_model: str, models_evaluated: list) -> WandBLogger:
    """Create a W&B logger for evaluation runs."""
    experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    wandb_config = {
        "base_model": base_model,
        "evaluation_date": datetime.now().isoformat(),
        "models_evaluated": models_evaluated,
    }
    
    tags = ["evaluation", "benchmarks"]
    
    return WandBLogger(
        name=experiment_name,
        config=wandb_config,
        tags=tags
    )


def log_evaluation_results(logger: WandBLogger, model_tag: str, results: Dict[str, Dict[str, float]]) -> None:
    """Log evaluation results for a specific model."""
    if not logger.enabled:
        return
        
    try:
        # Flatten metrics for W&B logging
        wandb_metrics = {}
        for suite_name, suite_metrics in results.items():
            for metric_name, value in suite_metrics.items():
                wandb_metrics[f"{model_tag}/{suite_name}/{metric_name}"] = value
        
        # Also log summary metrics
        wandb_metrics[f"{model_tag}/total_metrics_count"] = sum(len(suite) for suite in results.values())
        
        logger.log(wandb_metrics)
        
        # Create a results table for this model
        results_data = [
            [suite_name, metric_name, value]
            for suite_name, suite_metrics in results.items()
            for metric_name, value in suite_metrics.items()
        ]
        
        logger.log_table(
            f"{model_tag}_results_table",
            ["Suite", "Metric", "Value"],
            results_data
        )
        
        print(f"Logged {len(wandb_metrics)} metrics for {model_tag} to W&B")
        
    except Exception as e:
        print(f"Error logging {model_tag} metrics to W&B: {e}")


# Standalone evaluation logging functions (for backward compatibility)
def log_evaluation_to_wandb(tag: str, merged_metrics: Dict[str, Dict[str, float]], project: str) -> None:
    """Log evaluation metrics to W&B (backward-compatible helper).

    This variant respects common env vars, uses a safe start method, and
    flattens only numeric metrics. If W&B is unavailable or disabled via env,
    it prints a diagnostic and returns.
    """
    if not WANDB_AVAILABLE:
        print("W&B not available, skipping wandb logging")
        return
    # Honor env-based disable/offline modes
    offline = os.getenv("WANDB_MODE", "online") == "offline" or os.getenv("WANDB_DISABLED", "").lower() in ("true", "1")
    if offline:
        print("W&B disabled/offline in environment, skipping wandb logging")
        return
    try:
        settings = wandb.Settings(start_method=os.getenv("WANDB_START_METHOD", "thread"))
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", project),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=f"eval-{tag}",
            group=os.getenv("WANDB_GROUP"),
            notes=os.getenv("WANDB_NOTES"),
            resume=os.getenv("WANDB_RESUME", "allow"),
            id=os.getenv("WANDB_RUN_ID"),
            settings=settings,
            reinit=True,
        )
        flat: Dict[str, float] = {}
        for task, metrics in merged_metrics.items():
            if not isinstance(metrics, dict):
                continue
            for metric, val in metrics.items():
                if isinstance(val, (int, float)):
                    try:
                        flat[f"{task}/{metric}"] = float(val)
                    except Exception:
                        continue
        if flat:
            run.log(flat)
        run.finish()
        print(f"✓ Logged {len(flat)} metrics to W&B project '{run.project}'")
    except Exception as e:
        print(f"Failed to log to W&B: {e}")


def log_evaluation_to_tensorboard(
    tag: str, 
    merged_metrics: Dict[str, Dict[str, float]], 
    log_dir: str = "tb_logs"
) -> None:
    """Log evaluation metrics to TensorBoard."""
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available, skipping tensorboard logging")
        return
    try:
        tb_path = Path(log_dir) / tag
        tb_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_path))
        metric_count = 0
        for task, metrics in merged_metrics.items():
            for metric, val in metrics.items():
                writer.add_scalar(f"{task}/{metric}", val)
                metric_count += 1
        writer.close()
        print(f"✓ Logged {metric_count} metrics to TensorBoard at {tb_path}")
    except Exception as e:
        print(f"Failed to log to TensorBoard: {e}")


def create_training_combined_logger(
    config: "TrainingConfig",
    experiment_name: str,
    tensorboard_dir: Optional[str] = None
) -> CombinedLogger:
    """Create a combined logger for training with both W&B and TensorBoard.
    
    Args:
        config: Training configuration
        experiment_name: Name of the experiment
        tensorboard_dir: Directory for TensorBoard logs (optional)
        
    Returns:
        CombinedLogger instance
    """
    # Create W&B logger
    wandb_logger = create_training_logger(config, experiment_name)
    
    # Create TensorBoard logger
    tensorboard_logger = None
    if tensorboard_dir:
        tb_path = Path(tensorboard_dir) / experiment_name
        tensorboard_logger = TensorBoardLogger(str(tb_path))
    
    return CombinedLogger(wandb_logger, tensorboard_logger)
