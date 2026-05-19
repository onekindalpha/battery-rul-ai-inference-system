from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "scripts" / "prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py"

if not TARGET.exists():
    raise SystemExit(f"Runner target not found: {TARGET}")

runpy.run_path(str(TARGET), run_name="__main__")
