import json
from pathlib import Path


class PrecomputedRULLoader:
    def __init__(self, export_root: Path):
        self.root = export_root

    def get_path(self, battery: str, r_ratio: float) -> Path:
        fname = f"{battery}_r{r_ratio:.2f}.json"
        return self.root / fname

    def has_precomputed(self, battery: str, r_ratio: float) -> bool:
        return self.get_path(battery, r_ratio).exists()

    def load(self, battery: str, r_ratio: float):
        p = self.get_path(battery, r_ratio)
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    def save(self, battery: str, r_ratio: float, record: dict):
        p = self.get_path(battery, r_ratio)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(record, f, indent=2)

