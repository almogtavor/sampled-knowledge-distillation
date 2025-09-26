from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from pathlib import Path


class TrainingConfig(BaseModel):
    """Configuration for entropy knowledge distillation training."""
    
    # Model settings
    teacher_model: str
    student_model: str
    teacher_quant_bits: Optional[int] = None  # 4 or 8 to enable bitsandbytes quant for teacher
    student_quant_bits: Optional[int] = None  # optional quant for student (usually None for training)
    distill_type: Literal["vanilla", "top-k-tok", "random", "bucket"] = "vanilla"
    k_percent: int = Field(default=20, description="for top-k-tok and random")
    enable_ce: bool = Field(default=True, description="Enable cross-entropy loss in addition to KD loss")
    
    # Bucket mode parameters (for distill_type="bucket")
    bucket_lower_percent: int = Field(default=70, description="Lower bound for bucket mode (e.g., 70% means skip bottom 70%)")
    bucket_upper_percent: int = Field(default=80, description="Upper bound for bucket mode (e.g., 80% means skip top 20%)")
    
    # Dataset settings
    datasets: List[str]
    prompt_col: Optional[str] = None
    answer_col: Optional[str] = None
    dataset_config: Optional[str] = None
    
    # Training hyperparameters
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = Field(default=4, description="Number of steps to accumulate gradients")
    max_seq_len: int = Field(default=512, description="Maximum sequence length to save memory")
    lr: float = Field(default=1e-5, description="Learning rate")
    
    # Output and logging
    output_dir: str
    tensorboard_dir: str = "tb"
    wandb_project: str = "selective-entropy-knowledge-distillation"
    wandb_entity: str = "selective-entropy-knowledge-distillation"
    wandb_enabled: bool = True
    
    # Checkpointing
    checkpoint_steps: int = Field(default=500, description="Save checkpoint every N steps (0 to disable)")
    keep_checkpoints: int = Field(default=3, description="Number of recent checkpoints to keep")


class CheckpointData(BaseModel):
    """Structure for training checkpoint data."""
    
    epoch: int
    step: int
    global_step: int
    distill_type: str
    k_percent: int
    model_state_dict: dict
    optimizer_state_dict: dict
    
    class Config:
        # Allow arbitrary types for PyTorch state dicts
        arbitrary_types_allowed = True


class TrainingMetrics(BaseModel):
    """Structure for training metrics and logging."""
    
    loss: float
    kl_loss: float
    ce_loss: float
    epoch: int
    step: int
    global_step: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "train/loss": self.loss,
            "train/kl_loss": self.kl_loss,
            "train/ce_loss": self.ce_loss,
            "train/epoch": self.epoch
        }
    
    def to_wandb_dict(self) -> dict:
        """Convert to W&B-specific dictionary with additional context."""
        return {
            "train/loss": self.loss,
            "train/kl_loss": self.kl_loss,
            "train/ce_loss": self.ce_loss,
            "train/epoch": self.epoch,
            "train/step": self.step,
            "train/global_step": self.global_step,
        }
    
    def to_running_dict(self) -> dict:
        """Convert to running averages dictionary."""
        return {
            "loss": self.loss,
            "kl": self.kl_loss,
            "ce": self.ce_loss
        }
