from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for entropy knowledge distillation training."""
    
    # Model settings
    teacher_model: str
    student_model: str
    teacher_quant_bits: Optional[int] = None  # 4 or 8 to enable bitsandbytes quant for teacher
    student_quant_bits: Optional[int] = None  # optional quant for student (usually None for training)
    distill_type: Literal["vanilla", "top-k-tok", "random", "bucket", "pos-rs-kd", "linucb"] = "vanilla"
    k_percent: int = Field(default=20, description="for top-k-tok and random")
    enable_ce: bool = Field(default=True, description="Enable cross-entropy loss in addition to KD loss")
    alpha_ce: float = Field(default=0.1, description="Weight for cross-entropy loss (vs KD loss). Total loss = (1-alpha_ce)*L_KD + alpha_ce*L_CE")    
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

    # Score-KD parameters
    score_token_selection: bool = Field(default=False, description="Use composite score (entropy + student CE + KL) to rank tokens instead of pure entropy")
    score_normalize: Literal["none", "z", "minmax"] = "z"
    score_entropy_weight: float = Field(default=1.0, description="Weight for teacher entropy component in score-based KD")
    score_ce_weight: float = Field(default=1.0, description="Weight for student cross-entropy component in score-based KD")
    score_kl_weight: float = Field(default=1.0, description="Weight for teacher-student KL component in score-based KD")

    # LinUCB contextual bandit parameters
    bandit_alpha: float = Field(default=1.0, description="Exploration coefficient for LinUCB (higher = more exploratory)")
    bandit_lambda: float = Field(default=1.0, description="L2 regularization for LinUCB covariance matrix")
    bandit_threshold: float = Field(default=0.0, description="Minimum UCB score for a token to be selected")
    bandit_min_tokens: int = Field(default=1, description="Minimum number of tokens to distill per example when using LinUCB")
    bandit_max_tokens: Optional[int] = Field(default=None, description="Optional cap on tokens distilled per example in LinUCB mode")
    bandit_device: str = Field(default="cpu", description="Device to maintain the LinUCB statistics on (cpu or cuda)")
    bandit_reward_clip: float = Field(default=5.0, description="Absolute clip value applied to KL improvement rewards before LinUCB update")
    
    # Dataset settings
    datasets: List[str]
    prompt_col: Optional[str] = None
    answer_col: Optional[str] = None
    dataset_config: Optional[str] = None
    # FineWeb streaming token budget (used when datasets[0] == "fineweb")
    fineweb_tokens: int = Field(default=50_000_000, description="Token budget when streaming FineWeb-Edu")
    
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
    # Unified runs registry
    runs_registry: str = Field(default="results/runs.json", description="Path to the unified runs JSON registry")
    override: bool = Field(default=False, description="If true, run even if an identical-params hash exists in the registry")
    
    # Offline cache (teacher precomputation for entropy approx + RS-KD over vocab)
    offline_cache: bool = True
    offline_cache_dir: Optional[str] = None  # if None, defaults to f"{output_dir}/teacher_offline_cache"
    # Params used by the offline cache builder
    entropy_approx_m: int = Field(default=20, description="Top-m used in truncated entropy approximation")
    rs_vocab_samples: int = Field(default=64, description="Number of vocab samples per position for RS-KD")
    rs_vocab_beta: float = Field(default=1.0, description="Proposal exponent for RS-KD over vocab: q ∝ p^beta")
    # Entropy cache policy (always stored): True => uint8, False => fp16
    H_hat_u8: bool = Field(default=True, description="Store Ĥ as uint8 (True) or fp16 (False)")

    # Sampled softmax elimination (only active when using offline cache within cached path)
    eliminate_softmax: bool = Field(default=False, description="Eliminate full-vocab softmax in cached RS-KD path using sampled softmax and importance correction")
    sampled_softmax_negatives: int = Field(default=1024, description="Number of uniform negative samples per position when eliminate_softmax=True")
    
    # Global-Level Selection (GLS) over tokens — only affects top-k-tok when enabled
    gls_enabled: bool = Field(default=False, description="Enable global-level selection FIFO queue (only impacts top-k-tok)")
    gls_queue_size: int = Field(default=30000, description="Capacity of GLS FIFO queue for computing global threshold")
    gls_log_threshold: bool = Field(default=False, description="Log the GLS threshold each time it's computed")
    
    # Checkpointing
    checkpoint_steps: int = Field(default=500, description="Save checkpoint every N steps (0 to disable)")
    keep_checkpoints: int = Field(default=3, description="Number of recent checkpoints to keep")

    # Distributed training (offline DDP) context
    ddp_offline: bool = Field(default=False, description="Enable offline-mode DDP across multiple GPUs")
    ddp_world_size: int = Field(default=1, description="World size for DDP runs")
    ddp_rank: int = Field(default=0, description="Global rank for DDP runs")
    ddp_local_rank: int = Field(default=0, description="Local rank for DDP runs")


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
