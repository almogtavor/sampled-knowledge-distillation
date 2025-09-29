from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class TrainingConfig(BaseModel):
    """Configuration for entropy knowledge distillation training."""
    
    # Model settings
    teacher_model: str
    student_model: str
    teacher_quant_bits: Optional[int] = None  # 4 or 8 to enable bitsandbytes quant for teacher
    student_quant_bits: Optional[int] = None  # optional quant for student (usually None for training)
    distill_type: Literal["vanilla", "top-k-tok", "random", "bucket", "pos-rs-kd"] = "vanilla"
    k_percent: int = Field(default=20, description="for top-k-tok and random")
    enable_ce: bool = Field(default=True, description="Enable cross-entropy loss in addition to KD loss")
    alpha_ce: float = Field(default=0.1, description="Weight for cross-entropy loss (vs KD loss). Total loss = (1-alpha_ce)*L_KD + alpha_ce*L_CE")    
    # RS-KD parameters (for distill_type="pos-rs-kd")
    rs_alpha: float = Field(default=1.0, description="Exponent on entropy for sampling dist: q(i) ∝ H_i^alpha (alpha∈[0,∞))")
    rs_epsilon: float = Field(default=0.02, description="Mixture with uniform for tail coverage: q ← (1-ε)q + ε·uniform")
    rs_floor: float = Field(default=1e-6, description="Minimum probability floor to avoid huge weights / degeneracy")
    kd_temperature: float = Field(default=1.0, description="Unified KD temperature used for teacher/student log-softmax and loss scaling")
    entropy_approx_temperature: float = Field(default=1.0, description="Temperature used during offline pass for entropy approximation (and RS-KD proposal if applicable)")
    # KD temperature annealing (optional)
    anneal_kd_temperature: bool = Field(default=False, description="Enable annealing schedule for kd_temperature during training")
    kd_temperature_start: float = Field(default=2.0, description="Starting KD temperature when annealing")
    kd_temperature_end: float = Field(default=1.0, description="Final KD temperature when annealing")
    kd_hold_frac: float = Field(default=0.6, description="Fraction of total updates to hold at start temperature before linear decay")
    # RS-KD parameters (for distill_type="pos-rs-kd")
    rs_alpha: float = Field(default=1.0, description="Exponent on entropy for sampling dist: q(i) ∝ H_i^alpha (alpha∈[0,∞))")
    rs_epsilon: float = Field(default=0.02, description="Mixture with uniform for tail coverage: q ← (1-ε)q + ε·uniform")
    rs_floor: float = Field(default=1e-6, description="Minimum probability floor to avoid huge weights / degeneracy")
    
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
    # Reproducibility
    seed: int = Field(default=1337, description="Random seed for reproducibility")
    deterministic: bool = Field(default=False, description="Enable deterministic algorithms (may slow down)")
    
    # Output and logging
    output_dir: str
    tensorboard_dir: str = "tb"
    wandb_project: str = "selective-entropy-knowledge-distillation"
    wandb_entity: str = "selective-entropy-knowledge-distillation"
    wandb_enabled: bool = True
    
    # Offline cache (teacher precomputation for entropy approx + RS-KD over vocab)
    offline_cache: bool = True
    offline_cache_dir: Optional[str] = None  # if None, defaults to f"{output_dir}/teacher_offline_cache"
    # Params used by the offline cache builder
    entropy_approx_m: int = Field(default=20, description="Top-m used in truncated entropy approximation")
    rs_vocab_samples: int = Field(default=64, description="Number of vocab samples per position for RS-KD")
    rs_vocab_beta: float = Field(default=1.0, description="Proposal exponent for RS-KD over vocab: q ∝ p^beta")
    
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
