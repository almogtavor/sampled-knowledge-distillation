#!/usr/bin/env python3
"""
W&B Logging Utilities for EKD Project

This module provides utilities for logging training runs and evaluations to Weights & Biases.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """W&B logging utility for EKD project."""
    
    def __init__(
        self,
        project: str = "selective-entropy-knowledge-distillation",
        entity: str = "selective-entropy-knowledge-distillation",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.project = project
        self.entity = entity
        self.run = None
        self.enabled = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY") is not None
        
        if self.enabled:
            try:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=config or {},
                    tags=tags or []
                )
                print(f"W&B logging initialized: {self.run.get_url()}")
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")
                self.enabled = False
        else:
            print("W&B not available (missing wandb package or API key)")

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
                artifact = wandb.Artifact(name=name, type=type, description=description)
                if os.path.isdir(artifact_path):
                    artifact.add_dir(artifact_path)
                else:
                    artifact.add_file(artifact_path)
                self.run.log_artifact(artifact)
                print(f"Artifact '{name}' logged to W&B")
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


def create_training_logger(config, experiment_name: Optional[str] = None) -> WandBLogger:
    """Create a W&B logger for training runs."""
    if experiment_name is None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "local")
        experiment_name = (
            f"distill-{config.distill_type}-{current_date}_{job_id}"
            + (f"_k={config.top_k_percent}" if config.distill_type == "vanilla" else "")
        )
    
    wandb_config = {
        # Training config
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "distill_type": config.distill_type,
        "top_k_percent": config.top_k_percent,
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
    
    tags = [
        config.distill_type,
        f"k={config.top_k_percent}" if config.distill_type == "vanilla" else "ekd",
        "training"
    ]
    
    return WandBLogger(
        project=getattr(config, 'wandb_project', 'selective-entropy-knowledge-distillation'),
        entity=getattr(config, 'wandb_entity', 'selective-entropy-knowledge-distillation'),
        name=experiment_name,
        config=wandb_config,
        tags=tags
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
