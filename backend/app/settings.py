from pydantic import BaseModel
import os
from pathlib import Path

class Settings(BaseModel):
    # Base directory of this backend app
    BASE_DIR: str = str(Path(__file__).parent)
    # Data root mounted into container
    DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
    # Either a parquet directory (partitioned) or a csv file for demo
    # Try multiple locations for compatibility (Docker vs local dev)
    DATA_SOURCE: str = os.getenv("DATA_SOURCE", str(
        next(
            (p for p in [
                Path("/app/data/nasa_features_rul.csv"),
                Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/nasa_features_rul.csv"),
            ] if p.exists()),
            Path("/app/data/nasa_features_rul.csv")
        )
    ))
    META_CREF_PATH: str = os.getenv("META_CREF_PATH", "/app/data/battery_meta_c_ref.json")
    # Model weights path (optional, keep private on server)
    MODEL_PATH: str | None = os.getenv("MODEL_PATH")
    PRECOMP_DIR: str | None = os.getenv("PRECOMP_DIR", str(
        next(
            (p for p in [
                Path("/app/data/precomputed_from_export_v2"),
                Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/precomputed_from_export_v2"),
            ] if p.exists()),
            Path("/app/data/precomputed_from_export_v2")
        )
    ))
    # Default downsample stride to keep payload small
    DEFAULT_STRIDE: int = int(os.getenv("DEFAULT_STRIDE", "1"))

settings = Settings()
