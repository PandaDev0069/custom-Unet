"""Serializable central experiment configuration."""
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    data_root: str = ""
    output_dir: str = "filament_outputs"
    image_size: int = 2048  # Only synthetic APIs may change this.
    seed: int = 10
    val_fraction: float = 0.2
    multi_annotator_train_only: bool = True
    model: dict = field(default_factory=lambda: dict(name="modified", in_channels=1,
        out_channels=2, init_features=16, normalization="group", activation="relu", checkpoint_blocks=True))
    loss: dict = field(default_factory=lambda: dict(bce_weight=.35, dice_weight=.40,
        boundary_weight=.25, cldice_weight=0., boundary_pos_weight=5., skeleton_iterations=5))
    separation_radius: int = 1
    centerline: bool = False
    normalization: str = "unit"
    augment: dict = field(default_factory=lambda: dict(flips=True, rotate90=True,
        brightness=.05, contrast=.10, gamma=.10, noise=0.))
    epochs: int = 50
    patience: int = 8
    pq_every: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "plateau"
    accumulation_steps: int = 4
    microbatch: int | str = "auto"
    batch_candidates: list = field(default_factory=lambda: [1, 2, 4])
    width_fallbacks: list = field(default_factory=lambda: [8])
    memory_fraction: float = .85
    grad_clip: float = 1.
    amp: bool = True
    channels_last: bool = True
    workers: int = 4  # Total CPU worker budget, split across GPU ranks.
    prefetch_factor: int = 2
    log_every: int = 20
    resume: str = ""
    deterministic: bool = True
    smoke: bool = False
    smoke_records: int = 2
    postprocess: dict = field(default_factory=lambda: dict(method="boundary", foreground_threshold=.5,
        boundary_threshold=.5, min_area=20, seed_min_area=3, seed_distance=5,
        boundary_weight=2., closing_radius=0, reconnect_distance=0))

    def validate(self):
        if self.image_size != 2048: raise ValueError("Production requires native 2048x2048")
        if self.model.get("out_channels") != 2: raise ValueError("Production trains exactly two output channels")
        if self.normalization not in ("unit", "robust"): raise ValueError("Unknown normalization")
        if min(self.epochs, self.patience, self.pq_every, self.accumulation_steps, self.prefetch_factor, self.log_every) < 1:
            raise ValueError("Epoch, evaluation, accumulation and prefetch counts must be positive")
        if self.workers < 0 or not 0 < self.memory_fraction < 1 or not 0 < self.val_fraction < 1:
            raise ValueError("Invalid workers, VRAM fraction or validation fraction")
        if self.microbatch != "auto" and (not isinstance(self.microbatch, int) or self.microbatch < 1):
            raise ValueError("microbatch must be a positive integer or auto")
        if not self.batch_candidates or min(self.batch_candidates) < 1 or self.separation_radius < 1:
            raise ValueError("Probe batches and separation radius must be positive")
        if self.scheduler not in ("plateau", "cosine", "none"): raise ValueError("Unknown scheduler")
        return self

    def to_dict(self): return asdict(self)
