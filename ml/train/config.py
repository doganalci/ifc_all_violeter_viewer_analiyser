from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    # Data
    dataset_root: str = "../codex1"
    cache_root: str = "./data/cache"
    include_baselines: bool = False
    use_rule_oracle: bool = False         # Kural-tabanlı etiket augmentation
    mask_numeric_features: bool = False   # Feature leak'i engelle (OverallWidth vb. sıfırla)
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 0

    # Model
    model: str = "gat"        # 'gat' | 'hetero_gat'
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.3
    edge_emb_dim: int = 8

    # Optim
    epochs: int = 50
    lr: float = 5e-3
    weight_decay: float = 5e-4
    pos_weight: float | None = None   # None → derive from train class balance
    patience: int = 10                # early-stopping on val F1

    # Eval
    threshold: float = 0.5

    # Runtime
    device: str = "auto"      # 'auto' | 'cpu' | 'cuda'
    seed: int = 42
    run_dir: str = "./runs"
    run_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # local import to avoid hard dep at config time
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def ensure_dirs(self) -> Path:
        d = Path(self.run_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        Path(self.cache_root).expanduser().mkdir(parents=True, exist_ok=True)
        return d
