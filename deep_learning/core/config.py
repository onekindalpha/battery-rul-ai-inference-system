# deep_learning/core/config.py

from dataclasses import dataclass
from typing import Dict, Any

try:
    from ray import tune
except ImportError:
    tune = None

# inner_lr 상한 (meta_utils랑 의미 맞추기용)
INNER_LR_MAX = 0.005


@dataclass
class Config:
    # -----------------------------
    # Data source / preprocessing
    # -----------------------------
    # "nasa" / "cacle" / "both"
    dataset_source: str = "nasa"
    db_type: str = "causal"        # 그대로 causal 고정 사용
    feature_set: str = "full"      # "full" / "cleaned"
    use_ceemdan: bool = True

    # -----------------------------
    # Repro / paths
    # -----------------------------
    RANDOM_SEED: int = 42
    smoke_test: bool = False
    output_dir: str = "deep_learning/core_model_output"
    checkpoint_dir: str = "./core_checkpoints_nasa"

    # -----------------------------
    # Data / features
    # -----------------------------
    seq_len: int = 20
    horizon: int = 7
    eol_threshold: float = 0.8

    # scaler를 fit할 때 한 배터리 앞부분 몇 %만 사용할지
    scaler_fit_ratio: float = 0.3

    # 시퀀스 / summary feature clip
    seq_clip: float = 10.0
    sum_clip: float = 10.0

    # -----------------------------
    # RUL scaling
    # -----------------------------
    rul_mode: str = "minmax"       # "minmax" / "log1p" 등
    clip_min: float = -1e6
    clip_max: float = 1e6
    max_rul_clip: float = 500.0    # 필요시 max_rul_train을 이 값으로 클립할 때 사용 가능
    # [수정 후] 3000.0 (CACLE 최대 수명까지 커버)
    #max_rul_clip: float = 3000.0
    # -----------------------------
    # Model (Transformer backbone)
    # -----------------------------
    use_resnet: bool = True
    d_model: int = 16
    nhead: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    stochastic_depth_prob: float = 0.0
    token_clip: float = 10.0

    # loss weights (필요 시 meta_utils에서 사용)
    dual_weight: float = 0.0
    physics_weight: float = 0.0
    cal_weight: float = 0.1
    mu_weight: float = 1.0

    # -----------------------------
    # BMAML-SVGD hyper-params
    # -----------------------------
    inner_lr: float = 1e-3
    inner_lr_max: float = INNER_LR_MAX
    adaptation_steps: int = 15
    leader_steps: int = 1
    stop_tol: float = 5e-5

    num_particles: int = 3
    particle_init_std: float = 0.05
    noise_std: float = 1e-3

    # -----------------------------
    # Meta batching
    # -----------------------------
    meta_batch_size: int = 2
    k_shot: int = 16
    q_query: int = 16

    # prefix sampling (MetaBatteryDataset)
    prefix_min_ratio: float = 0.4
    prefix_max_ratio: float = 0.9
    val_prefix_ratio: float = 0.5

    # support set sampling strategy: "recent" / "random" / "mixed"
    support_strategy: str = "mixed"

    # -----------------------------
    # RUL=0 tail 처리 옵션 (학습에서)
    # -----------------------------
    # True면 한 배터리에서 RUL=0인 윈도우가 tail에 여러 개 있을 때
    #   → 첫 윈도우 하나(또는 한두 개)만 남기고 나머지 0-RUL 윈도우는 버림
    drop_zero_tail: bool = False
    # -----------------------------
    # Logging / W&B
    # -----------------------------
    use_wandb: bool = False
    wandb_project: str = "battery_rul_bmaml_svgd_core"
    wandb_group: str = "bmaml_svgd_both_core"
    wandb_mode: str = "online"
    wandb_run_name: str = "both_meta_core"

    # -----------------------------
    # Outer loop (backbone / meta)
    # -----------------------------
    epochs: int = 40
    # learning_rate: float = 1e-4
    # weight_decay: float = 1e-4
    learning_rate: float = 3e-4  # 혹은 5e-5
    weight_decay: float = 1e-5   # 조금 줄이기

    outer_clip: float = 1.0

    report_every: int = 1
    max_patience: int = 40
    max_patience_backbone: int = 20
    warmup_epochs: int = 5

    # supervised backbone 전용
    supervised_batch_size: int = 64


# -----------------------------------------
# Ray Tune search space (단일 Config로 고정)
# -----------------------------------------
def get_search_space() -> Dict[str, Any]:
    if tune is None:
        return {}

    # Config 기본값을 쓰고, 튜닝은 사실상 비활성화
    return {
        "d_model": tune.choice([16]),
        "num_particles": tune.choice([3]),
        "meta_batch_size": tune.choice([2]),
        "inner_lr": tune.choice([1e-3]),
        "learning_rate": tune.choice([1e-4]),
    }


search_space = get_search_space()
