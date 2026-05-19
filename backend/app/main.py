import re
import json
import time
import threading
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add deep_learning module to path
_base_path = Path("/app") if Path("/app").exists() else Path("/Users/velocitygoal/battery-rul-v11")
sys.path.insert(0, str(_base_path / "deep_learning" / "core"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .settings import settings
# --- HOTFIX V3: local NASA features CSV path fallback ---
# Reason: local dev runs outside Docker, but some settings/data_access code may still point to
# /app/data/nasa_features_rul.csv. Inject this before importing fetch_cycles so the backend
# resolves the real local CSV path.
try:
    import os as _hotfix_os
    _default_csv = "/app/data/nasa_features_rul.csv" if Path("/app/data/nasa_features_rul.csv").exists() else "/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/nasa_features_rul.csv"
    _hotfix_csv = Path(_hotfix_os.environ.get("NASA_FEATURES_RUL_CSV", _default_csv))
    if _hotfix_csv.exists():
        _hotfix_csv_s = str(_hotfix_csv)
        for _k in (
            "NASA_FEATURES_RUL_CSV",
            "NASA_FEATURES_PATH",
            "NASA_FEATURES_CSV",
            "FEATURES_CSV_PATH",
            "CSV_PATH",
            "DATA_PATH",
        ):
            _hotfix_os.environ.setdefault(_k, _hotfix_csv_s)
        for _attr in (
            "NASA_FEATURES_RUL_CSV",
            "NASA_FEATURES_PATH",
            "NASA_FEATURES_CSV",
            "FEATURES_CSV_PATH",
            "CSV_PATH",
            "DATA_PATH",
        ):
            try:
                if hasattr(settings, _attr):
                    object.__setattr__(settings, _attr, _hotfix_csv_s)
            except Exception:
                pass
    else:
        print(f"[HOTFIX V3] WARNING: NASA features CSV not found at {_hotfix_csv}")
except Exception as _hotfix_e:
    print(f"[HOTFIX V3] WARNING: failed to apply CSV path fallback: {_hotfix_e}")
# --- END HOTFIX V3 ---
from .data_access import fetch_cycles
from .diagnostics import add_eol_columns, coverage_stats, first_consecutive

# macOS Docker Desktop bind mounts can intermittently raise EDEADLK (Errno 35) on reads.
_FILE_LOCK = threading.Lock()

# Your intended r-ratio menu
ALLOWED_R = [0.10, 0.20, 0.30, 0.40]
_ALLOWED_SET = {round(x, 2) for x in ALLOWED_R}

# Precomputed cache loaded once at startup to avoid runtime file I/O deadlocks.
_PRECOMP_PAYLOAD: dict[tuple[str, float], Any] = {}
_PRECOMP_FILE: dict[tuple[str, float], str] = {}
_PRECOMP_R_BY_BAT: dict[str, list[float]] = {}

# Docker-aware path resolver
def _resolve_path(*candidates: str) -> Path:
    """Return first existing path from candidates, or last candidate."""
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p
    return Path(candidates[-1]) if candidates else Path("/app")

# Initialize core paths
_APP_ROOT = Path("/app") if Path("/app").exists() else Path("/Users/velocitygoal/battery-rul-v11")
_CKPT_PATH = _resolve_path(
    "/app/core_checkpoints/nasa_bmaml_best_re.pt",
    "/Users/velocitygoal/battery-rul-v11/core_checkpoints/nasa_bmaml_best_re.pt"
)
_SCRIPTS_PATH = _resolve_path(
    "/app/scripts/prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py",
    "/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/scripts/prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py"
)
_SHAP_CANDIDATES = [
    "/app/data/shap_outputs/bmaml_shap_seq_feature_importance.json",
    "/app/deep_learning/core/shap_outputs/bmaml_shap_seq_feature_importance.json",
    "/app/shap_outputs/bmaml_shap_seq_feature_importance.json",
    "/Users/velocitygoal/battery_project/v11/deep_learning/core/shap_outputs/bmaml_shap_seq_feature_importance.json",
    "/Users/velocitygoal/battery-rul-v11/shap_outputs/bmaml_shap_seq_feature_importance.json",
]

app = FastAPI(title="Battery RUL API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- HOTFIX_V28_HARD_REINFER_AND_SHAP_OVERRIDE ---
@app.middleware("http")
async def _v28_hard_reinfer_and_shap_override(request, call_next):
    import json as _json
    import math as _math
    import os as _os
    import re as _re
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path
    try:
        from fastapi.responses import JSONResponse as _JSONResponse
    except Exception:
        _JSONResponse = JSONResponse

    path = request.url.path
    method = request.method.upper()

    CKPT = _CKPT_PATH
    RUNNER = _SCRIPTS_PATH
    LIVE_DIR = _resolve_path(
        "/app/data/live_reinfer_results",
        "/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/live_reinfer_results"
    )
    PYROOTS = [str(_APP_ROOT), str(_APP_ROOT / "backend"), str(_APP_ROOT / "deep_learning")]
    SHAP_CANDIDATES = _SHAP_CANDIDATES

    def resp(code, payload):
        return _JSONResponse(status_code=code, content=payload)

    def safe_float(v, default=None):
        try:
            x = float(v)
            return x if _math.isfinite(x) else default
        except Exception:
            return default

    def battery_from_path():
        patterns = [
            r"/api/battery/([^/]+)/reinfer",
            r"/api/battery/([^/]+)/re-inference",
            r"/api/live-reinfer-v\d+/([^/]+)",
            r"/api/live-reinfer/([^/]+)",
        ]
        for pat in patterns:
            m = _re.fullmatch(pat, path)
            if m:
                return m.group(1)
        if path in {"/api/reinfer", "/api/re-inference", "/api/live-reinfer", "/api/live-reinfer-v28"}:
            return request.query_params.get("battery") or request.query_params.get("battery_id") or request.query_params.get("bid")
        return None

    # Hard override any old reinfer endpoint BEFORE old middleware/routes can return the stale
    # "No live model runner script found..." error.
    bid = battery_from_path()
    if bid and method in {"GET", "POST"}:
        if not CKPT.exists():
            return resp(500, {"ok": False, "error": "checkpoint not found", "path": str(CKPT)})
        if not RUNNER.exists():
            return resp(500, {"ok": False, "error": "runner not found", "path": str(RUNNER)})

        r = safe_float(request.query_params.get("r_ratio", request.query_params.get("rRatio", "0.1")), 0.1)
        r = round(float(r), 2)
        tag = "r" + f"{r:.2f}".replace(".", "p")
        run_id = _time.strftime("%Y%m%d_%H%M%S")
        out_dir = LIVE_DIR / f"{bid}_{tag}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        expected_json = out_dir / f"{bid}_viz_meta_{tag}.json"
        timeout = int(safe_float(request.query_params.get("timeout", "900"), 900))

        env = dict(_os.environ)
        env["PYTHONPATH"] = ":".join(PYROOTS + [env.get("PYTHONPATH", "")])
        env["CHECKPOINT_PATH"] = str(CKPT)
        env["BMAML_CHECKPOINT"] = str(CKPT)
        env["PRECOMP_DIR"] = str(out_dir)

        args = [
            str(RUNNER),
            "--ckpt", str(CKPT),
            "--eval_dataset", "from_ckpt",
            "--r_ratio", str(r),
            "--bids", str(bid),
            "--out_dir", str(out_dir),
            "--save_json", "1",
            "--save_batch_json", "1",
            "--min_support", "0",
            "--cap_before_eol", "1",
            "--ratio_base", "pos",
        ]

        commands = []
        conda = _shutil.which("conda")
        if conda:
            commands.append(("conda run -n battery-maml", [conda, "run", "-n", "battery-maml", "python"] + args))
        for p in [
            "/opt/anaconda3/envs/battery-maml/bin/python",
            "/Users/velocitygoal/anaconda3/envs/battery-maml/bin/python",
            "/Users/velocitygoal/miniforge3/envs/battery-maml/bin/python",
            "/Users/velocitygoal/mambaforge/envs/battery-maml/bin/python",
        ]:
            if _Path(p).exists():
                commands.append((p, [p] + args))
        if _sys.executable:
            commands.append((_sys.executable, [_sys.executable] + args))
        which_py = _shutil.which("python")
        if which_py:
            commands.append((which_py, [which_py] + args))

        seen = set()
        deduped = []
        for label, cmd in commands:
            key = tuple(cmd)
            if key not in seen:
                deduped.append((label, cmd))
                seen.add(key)

        attempts = []
        started = _time.time()
        for label, cmd in deduped:
            try:
                proc = _subprocess.run(
                    cmd,
                    cwd=PYROOTS[0],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                attempts.append({
                    "executor": label,
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-2200:],
                    "stderr_tail": proc.stderr[-2200:],
                })
                if proc.returncode == 0 and expected_json.exists():
                    item = _json.loads(expected_json.read_text(encoding="utf-8"))
                    return resp(200, {
                        "ok": True,
                        "success": True,
                        "mode": "live_bmaml_reinfer_session_only",
                        "battery": bid,
                        "r_ratio": r,
                        "tag": tag,
                        "executor": label,
                        "checkpoint": str(CKPT),
                        "runner": str(RUNNER),
                        "out_dir": str(out_dir),
                        "json_path": str(expected_json),
                        "baseline_overwritten": False,
                        "elapsed_sec": round(_time.time() - started, 2),
                        "item": item,
                        "stdout_tail": proc.stdout[-2200:],
                        "stderr_tail": proc.stderr[-2200:],
                    })
            except Exception as e:
                attempts.append({"executor": label, "exception": str(e)})

        return resp(500, {
            "ok": False,
            "success": False,
            "mode": "live_bmaml_reinfer_session_only",
            "battery": bid,
            "r_ratio": r,
            "expected_json": str(expected_json),
            "baseline_overwritten": False,
            "error": "all runner attempts failed or expected JSON was not produced",
            "attempts": attempts,
        })

    # Hard SHAP endpoint override.
    if path in {"/api/fixed4/shap-current", "/api/fixed4/shap-v28", "/api/fixed4/shap-v27", "/api/fixed4/shap-v26", "/api/fixed4/shap-v25"}:
        p = next((_Path(x) for x in SHAP_CANDIDATES if _Path(x).exists()), None)
        if p is None:
            return resp(404, {"ok": False, "error": "SHAP global importance JSON not found", "searched": SHAP_CANDIDATES})

        try:
            obj = _json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            return resp(500, {"ok": False, "error": f"failed to read SHAP JSON: {e}", "path": str(p)})

        def is_num(v):
            try:
                return _math.isfinite(float(v))
            except Exception:
                return False

        def list_items(xs):
            out = []
            for x in xs or []:
                if not isinstance(x, dict):
                    continue
                name = x.get("feature") or x.get("name") or x.get("feature_name") or x.get("column") or x.get("col")
                val = x.get("importance", x.get("value", x.get("mean_abs_shap", x.get("abs_mean", x.get("shap", None)))))
                if name is not None and is_num(val):
                    out.append({"feature": str(name), "importance": float(val)})
            return out

        def parse(o):
            if isinstance(o, list):
                got = list_items(o)
                if got:
                    return got
            if not isinstance(o, dict):
                return []
            if o and all(is_num(v) for v in o.values()):
                return [{"feature": str(k), "importance": float(v)} for k, v in o.items()]
            for key in ["items", "data", "global_importance", "feature_importance", "shap_importance"]:
                got = list_items(o.get(key))
                if got:
                    return got
            for nk in ["feature_names", "features", "names", "columns"]:
                names = o.get(nk)
                if not isinstance(names, list):
                    continue
                for vk in ["importance", "importances", "values", "mean_abs_shap", "shap_values", "global_importance"]:
                    vals = o.get(vk)
                    if isinstance(vals, list) and len(vals) == len(names):
                        out = []
                        for a, b in zip(names, vals):
                            if is_num(b):
                                out.append({"feature": str(a), "importance": float(b)})
                        if out:
                            return out
            for v in o.values():
                got = parse(v)
                if got:
                    return got
            return []

        items = parse(obj)
        items = [x for x in items if x.get("feature") and is_num(x.get("importance"))]
        items.sort(key=lambda x: float(x["importance"]), reverse=True)
        return resp(200, {
            "ok": True,
            "source": str(p),
            "kind": "global_model_feature_importance",
            "note": "Global BMAML/sequence-model feature importance; not a cycle-local anomaly driver.",
            "items": items,
        })

    return await call_next(request)
# --- END HOTFIX_V28_HARD_REINFER_AND_SHAP_OVERRIDE ---


@app.middleware("http")
async def _v24_reinfer_alias_middleware(request, call_next):
    import re as _re

    path = request.url.path
    m = (
        _re.fullmatch(r"/api/live-reinfer-v\d+/([^/]+)", path)
        or _re.fullmatch(r"/api/live-reinfer/([^/]+)", path)
        or _re.fullmatch(r"/api/battery/([^/]+)/re-inference", path)
    )
    if m:
        # Re-route old frontend calls into the canonical no-overwrite re-inference route.
        request.scope["path"] = f"/api/battery/{m.group(1)}/reinfer"
    return await call_next(request)


# --- HOTFIX_V21_REAL_REINFER_SHAP_EXPLAIN_CLEANUP ---
@app.middleware("http")
async def _hotfix_v21_real_reinfer_shap_explain_cleanup(request, call_next):
    import os as _os
    import re as _re
    import sys as _sys
    import json as _json
    import time as _time
    import math as _math
    import subprocess as _subprocess
    from pathlib import Path as _Path

    try:
        from fastapi.responses import JSONResponse as _JSONResponse
    except Exception:
        _JSONResponse = JSONResponse

    path = request.url.path

    CKPT = _Path("/Users/velocitygoal/battery-rul-v11/core_checkpoints/nasa_bmaml_best_re.pt")
    RUNNER = _Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/scripts/prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py")
    PRECOMP_DIR = _Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/precomputed_from_export_v2")
    SHAP_JSON = _Path("/Users/velocitygoal/battery_project/v11/deep_learning/core/shap_outputs/bmaml_shap_seq_feature_importance.json")
    PYROOTS = ["/Users/velocitygoal/battery_project/v11/server/backend", "/Users/velocitygoal/battery_project/v11", "/Users/velocitygoal/battery-rul-v11"]

    def _resp(code, payload):
        return _JSONResponse(status_code=code, content=payload)

    def _clean(x):
        try:
            if isinstance(x, dict):
                return {str(k): _clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_clean(v) for v in x]
            if hasattr(x, "item"):
                return _clean(x.item())
            if isinstance(x, float) and not _math.isfinite(x):
                return None
            return x
        except Exception:
            return x

    def _battery_from_path():
        patterns = [
            r"/api/live-reinfer-v21/([^/]+)",
            r"/api/live-reinfer-v20/([^/]+)",
            r"/api/live-reinfer/([^/]+)",
            r"/api/battery/([^/]+)/reinfer",
            r"/api/battery/([^/]+)/re-inference",
        ]
        for pat in patterns:
            m = _re.fullmatch(pat, path)
            if m:
                return m.group(1)
        if path in {"/api/reinfer", "/api/re-inference", "/api/live-reinfer-v21"}:
            return request.query_params.get("battery") or request.query_params.get("battery_id") or request.query_params.get("bid")
        return None

    def _try_reload_precomputed():
        for name in [
            "_load_precomputed_all",
            "load_precomputed_all",
            "_reload_precomputed",
            "reload_precomputed",
            "load_all_precomputed",
            "_load_precomputed",
        ]:
            fn = globals().get(name)
            if callable(fn):
                try:
                    out = fn()
                    return {"ok": True, "method": name, "return": str(out)[:300]}
                except Exception as e:
                    return {"ok": False, "method": name, "error": str(e)}
        return {"ok": False, "method": None, "error": "precomputed reload function not found"}

    def _is_num(v):
        try:
            return _math.isfinite(float(v))
        except Exception:
            return False

    def _normalize_shap_items(obj):
        def normalize_list(xs):
            out = []
            for x in xs or []:
                if not isinstance(x, dict):
                    continue
                name = x.get("feature") or x.get("name") or x.get("feature_name") or x.get("column") or x.get("col")
                val = x.get("importance", x.get("value", x.get("mean_abs_shap", x.get("abs_mean", x.get("shap", None)))))
                if name is not None and _is_num(val):
                    out.append({"feature": str(name), "importance": float(val)})
            return out

        if isinstance(obj, list):
            got = normalize_list(obj)
            if got:
                return got

        if not isinstance(obj, dict):
            return []

        if obj and all(_is_num(v) for v in obj.values()):
            return [{"feature": str(k), "importance": float(v)} for k, v in obj.items()]

        for key in ["items", "data", "global_importance", "feature_importance", "shap_importance"]:
            got = normalize_list(obj.get(key))
            if got:
                return got

        for nk in ["feature_names", "features", "names", "columns"]:
            names = obj.get(nk)
            if not isinstance(names, list):
                continue
            for vk in ["importance", "importances", "values", "mean_abs_shap", "shap_values", "global_importance"]:
                vals = obj.get(vk)
                if isinstance(vals, list) and len(vals) == len(names):
                    out = []
                    for a, b in zip(names, vals):
                        if _is_num(b):
                            out.append({"feature": str(a), "importance": float(b)})
                    if out:
                        return out

        for v in obj.values():
            got = _normalize_shap_items(v)
            if got:
                return got

        return []

    if path in {"/api/fixed4/shap-v21", "/api/fixed4/shap-v20"}:
        try:
            if not SHAP_JSON.exists():
                return _resp(404, {
                    "ok": False,
                    "error": "SHAP global importance JSON not found",
                    "path": str(SHAP_JSON),
                })

            obj = _json.loads(SHAP_JSON.read_text(encoding="utf-8", errors="ignore"))
            items = _normalize_shap_items(obj)
            items = [x for x in items if x.get("feature") and _is_num(x.get("importance"))]
            items.sort(key=lambda x: float(x["importance"]), reverse=True)

            return _resp(200, _clean({
                "ok": True,
                "source": str(SHAP_JSON),
                "kind": "global_model_feature_importance",
                "note": "Global BMAML/sequence-model feature importance; not a cycle-local anomaly driver.",
                "items": items,
            }))
        except Exception as e:
            return _resp(500, {"ok": False, "error": "shap-v21 failed: " + str(e), "path": str(SHAP_JSON)})

    battery_id = _battery_from_path()
    if battery_id:
        r_ratio = str(request.query_params.get("r_ratio", request.query_params.get("rRatio", "0.1")))
        timeout = int(float(request.query_params.get("timeout", "900")))

        if not CKPT.exists():
            return _resp(500, {"ok": False, "error": "checkpoint not found", "checkpoint": str(CKPT)})
        if not RUNNER.exists():
            return _resp(500, {"ok": False, "error": "runner not found", "runner": str(RUNNER)})

        PRECOMP_DIR.mkdir(parents=True, exist_ok=True)

        env = dict(_os.environ)
        env["PYTHONPATH"] = ":".join(PYROOTS + [env.get("PYTHONPATH", "")])
        env["CHECKPOINT_PATH"] = str(CKPT)
        env["BMAML_CHECKPOINT"] = str(CKPT)
        env["PRECOMP_DIR"] = str(PRECOMP_DIR)

        py = _sys.executable or "python3"
        cmd = [
            py,
            str(RUNNER),
            "--ckpt", str(CKPT),
            "--eval_dataset", "from_ckpt",
            "--r_ratio", str(r_ratio),
            "--bids", str(battery_id),
            "--out_dir", str(PRECOMP_DIR),
            "--save_json", "1",
            "--save_batch_json", "1",
            "--min_support", "0",
            "--cap_before_eol", "1",
            "--ratio_base", "pos",
        ]

        started = _time.time()
        try:
            proc = _subprocess.run(
                cmd,
                cwd="/Users/velocitygoal/battery_project/v11/server/backend",
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except Exception as e:
            return _resp(500, {
                "ok": False,
                "mode": "live_bmaml_reinfer",
                "battery": battery_id,
                "r_ratio": r_ratio,
                "checkpoint": str(CKPT),
                "runner": str(RUNNER),
                "out_dir": str(PRECOMP_DIR),
                "error": str(e),
                "cmd": cmd,
            })

        elapsed = round(_time.time() - started, 2)

        tag = "default"
        try:
            rr = float(r_ratio)
            tag = "r" + f"{rr:.2f}".replace(".", "p")
        except Exception:
            pass

        expected_json = PRECOMP_DIR / f"{battery_id}_viz_meta_{tag}.json"
        reload_info = _try_reload_precomputed() if proc.returncode == 0 else {"ok": False, "skipped": True}

        payload = {
            "ok": proc.returncode == 0,
            "mode": "live_bmaml_reinfer",
            "battery": battery_id,
            "r_ratio": r_ratio,
            "checkpoint": str(CKPT),
            "runner": str(RUNNER),
            "out_dir": str(PRECOMP_DIR),
            "expected_json": str(expected_json),
            "expected_json_exists": expected_json.exists(),
            "elapsed_sec": elapsed,
            "returncode": proc.returncode,
            "cmd": cmd,
            "cache_reload": reload_info,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }

        return _resp(200 if proc.returncode == 0 else 500, _clean(payload))

    return await call_next(request)
# --- END HOTFIX_V21_REAL_REINFER_SHAP_EXPLAIN_CLEANUP ---


# --- HOTFIX_V20_FIXED4_COMPARE_EXPLAIN_SAFE ---
@app.middleware("http")
async def _hotfix_v20_fixed4_compare_explain_safe(request, call_next):
    import re as _re, os as _os, sys as _sys, time as _time, subprocess as _subprocess, math as _math, glob as _glob, json as _jsonmod
    from pathlib import Path as _Path
    try:
        from fastapi.responses import JSONResponse as _JSONResponse
    except Exception:
        _JSONResponse = JSONResponse
    path = request.url.path
    def resp(code,payload): return _JSONResponse(status_code=code, content=payload)
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, list): return [clean(v) for v in x]
        if hasattr(x,'item'): return clean(x.item())
        if isinstance(x,float) and not _math.isfinite(x): return None
        return x
    here=_Path(__file__).resolve(); project=here.parents[2]; repo=here.parents[3]
    def csv_path():
        for p in [project/'data/nasa_features_rul.csv', project/'analysis/nasa_features_rul.csv', repo/'analysis/nasa_features_rul.csv', repo/'data/nasa_features_rul.csv', _Path('/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/nasa_features_rul.csv'), _Path('/Users/velocitygoal/battery-rul-v11/analysis/nasa_features_rul.csv')]:
            if p.exists(): return p
        return None
    if path == '/api/fixed4/compare-v20':
        try:
            import pandas as pd, numpy as np
            p=csv_path()
            if p is None: return resp(500, {'ok':False,'error':'nasa_features_rul.csv not found'})
            df=pd.read_csv(p)
            if 'battery_id' not in df.columns and 'battery' in df.columns: df['battery_id']=df['battery'].astype(str)
            if 'cycle' not in df.columns and 'cycle_num' in df.columns: df['cycle']=df['cycle_num']
            df['battery_id']=df['battery_id'].astype(str); df['cycle']=df['cycle'].astype(int)
            metric=str(request.query_params.get('metric','soh')); cohort=str(request.query_params.get('cohort','all'))
            bids=[b.strip() for b in str(request.query_params.get('batteries','B0018,B0043')).split(',') if b.strip()]
            opts={'soh':('soh','SoH (%)','pct'),'capacity_pct':('capacity_mean','Capacity (% of initial)','cap_pct'),'capacity_mean':('capacity_mean','Capacity (% of initial)','cap_pct'),'impedance_sum':('impedance_sum','Impedance sum (Ω)','raw'),'dcr':('dcr','DCR (Ω)','raw'),'thermal_stress':('thermal_stress','Thermal stress','raw'),'temperature_mean':('temperature_mean','Temp mean (°C)','raw'),'lli':('lli','LLI','raw'),'lam':('lam','LAM','raw')}
            key,label,mode=opts.get(metric,opts['soh'])
            def filt(d):
                if cohort=='all' or 'ambient_temp_c' not in d.columns: return d
                t=d['ambient_temp_c'].astype(float)
                if cohort=='temp_le_10': return d[t<=10.0]
                if cohort=='temp_10_25': return d[(t>10.0)&(t<25.0)]
                if cohort=='temp_ge_25': return d[t>=25.0]
                return d
            def band(d):
                if mode=='cap_pct':
                    if 'capacity_mean' not in d.columns: return {'x':[],'median':[],'q25':[],'q75':[]}
                    tmp=d[['battery_id','cycle','capacity_mean']].dropna().copy()
                    if tmp.empty: return {'x':[],'median':[],'q25':[],'q75':[]}
                    first=tmp.sort_values(['battery_id','cycle']).groupby('battery_id')['capacity_mean'].first().rename('cap0')
                    tmp=tmp.join(first,on='battery_id'); tmp=tmp[tmp['cap0'].astype(float)>0]
                    tmp['value']=tmp['capacity_mean'].astype(float)/tmp['cap0'].astype(float)*100.0
                else:
                    if key not in d.columns: return {'x':[],'median':[],'q25':[],'q75':[]}
                    tmp=d[['cycle',key]].dropna().copy()
                    if tmp.empty: return {'x':[],'median':[],'q25':[],'q75':[]}
                    tmp['value']=tmp[key].astype(float)
                    if mode=='pct': tmp['value']=tmp['value']*100.0
                g=tmp.groupby('cycle')['value']; med=g.median(); q25=g.quantile(.25); q75=g.quantile(.75)
                return {'x':med.index.astype(float).tolist(),'median':med.astype(float).tolist(),'q25':q25.astype(float).tolist(),'q75':q75.astype(float).tolist()}
            series=[]
            for bid in bids:
                one=df[df['battery_id']==str(bid)].sort_values('cycle').copy()
                if one.empty or key not in one.columns:
                    series.append({'battery':bid,'x':[],'y':[],'missing':True}); continue
                x=one['cycle'].astype(float).values; y=one[key].astype(float).values
                if key=='soh': y=y*100.0
                elif mode=='cap_pct' and key=='capacity_mean':
                    try:
                        cap0=float(one.iloc[0]['capacity_mean'])
                        if cap0>0: y=one['capacity_mean'].astype(float).values/cap0*100.0
                    except Exception: pass
                series.append({'battery':bid,'x':x.tolist(),'y':[None if not np.isfinite(v) else float(v) for v in y]})
            return resp(200, clean({'ok':True,'source':str(p),'metric':metric,'metric_key':key,'metric_mode':mode,'metric_label':label,'cohort':cohort,'series':series,'band':band(filt(df.copy()))}))
        except Exception as e: return resp(500, {'ok':False,'error':f'fixed4 compare-v20 failed: {e}'})
    if path == '/api/fixed4/shap-v20':
        try:
            files=[]
            for r in [project/'shap_outputs',project/'data/shap_outputs',repo/'shap_outputs',repo/'data/shap_outputs']:
                files += [_Path(x) for x in _glob.glob(str(r/'*.json'))]
            files=[x for x in files if x.exists()]
            if not files: return resp(404, {'ok':False,'error':'SHAP files not found'})
            p=sorted(files,key=lambda x:x.stat().st_mtime,reverse=True)[0]
            d=_jsonmod.loads(p.read_text(encoding='utf-8',errors='ignore'))
            names=d.get('feature_names') or d.get('features') or d.get('names'); vals=d.get('importance') or d.get('importances') or d.get('values') or d.get('shap_values')
            if isinstance(vals,dict): items=[{'feature':str(k),'importance':float(v)} for k,v in vals.items()]
            elif isinstance(names,list) and isinstance(vals,list): items=[{'feature':str(a),'importance':float(b)} for a,b in zip(names,vals)]
            else: items=[]
            items=[x for x in items if x['feature'] and _math.isfinite(x['importance'])]; items.sort(key=lambda x:x['importance'], reverse=True)
            return resp(200, {'ok':True,'source':str(p),'items':items})
        except Exception as e: return resp(500, {'ok':False,'error':f'shap-v20 failed: {e}'})
    m=_re.fullmatch(r'/api/live-reinfer-v20/([^/]+)', path)
    if m:
        bid=m.group(1); rr=str(request.query_params.get('r_ratio',request.query_params.get('rRatio','0.1'))); timeout=int(float(request.query_params.get('timeout','360')))
        candidates=[repo/'export_rul_dashboard_data_meta_fixed.py',repo/'export_rul_dashboard_data_meta_fixed_v4.py',repo/'run_bmaml_reinfer.py',repo/'run_live_reinfer.py',repo/'bmaml_svgd_reinfer.py',project/'export_rul_dashboard_data_meta_fixed.py',project/'backend/export_rul_dashboard_data_meta_fixed.py']
        script=next((p for p in candidates if p.exists()),None)
        if script is None: return resp(500, {'ok':False,'mode':'live_model_reinfer','error':'No live model runner script found','searched':[str(p) for p in candidates]})
        env=dict(_os.environ); env.update({'DEFAULT_R_RATIO':rr,'R_RATIO':rr,'BATTERY_ID':bid,'TARGET_BATTERY':bid,'LIVE_REINFER':'1','PYTHONPATH':f'{repo}:{project}:{env.get("PYTHONPATH","")}'})
        py=_sys.executable or 'python3'; cmds=[[py,str(script),'--battery',bid,'--r_ratio',rr],[py,str(script),'--battery_id',bid,'--r_ratio',rr],[py,str(script),'--target_battery',bid,'--r_ratio',rr],[py,str(script),'--r_ratio',rr],[py,str(script)]]
        attempts=[]; start=_time.time()
        for cmd in cmds:
            try:
                pr=_subprocess.run(cmd,cwd=str(repo),env=env,text=True,capture_output=True,timeout=timeout)
                attempts.append({'cmd':cmd,'returncode':pr.returncode,'stdout_tail':pr.stdout[-2500:],'stderr_tail':pr.stderr[-2500:]})
                if pr.returncode==0: return resp(200, {'ok':True,'mode':'live_model_reinfer','battery':bid,'r_ratio':rr,'script':str(script),'cmd':cmd,'elapsed_sec':round(_time.time()-start,2),'stdout_tail':pr.stdout[-2500:],'stderr_tail':pr.stderr[-2500:]})
            except Exception as e: attempts.append({'cmd':cmd,'exception':str(e)})
        return resp(500, {'ok':False,'mode':'live_model_reinfer','script':str(script),'error':'runner found but all invocations failed','attempts':attempts})
    return await call_next(request)
# --- END HOTFIX_V20_FIXED4_COMPARE_EXPLAIN_SAFE ---


# --- HOTFIX_V19_FIXED4_COMPARE_AND_LIVE_REINFER ---
@app.middleware("http")
async def _hotfix_v19_fixed4_compare_and_live_reinfer(request, call_next):
    import re as _re
    from pathlib import Path as _Path
    import os as _os
    import sys as _sys
    import time as _time
    import subprocess as _subprocess
    import math as _math

    try:
        from fastapi.responses import JSONResponse as _JSONResponse
    except Exception:
        _JSONResponse = JSONResponse

    path = request.url.path

    def _json(status, payload):
        return _JSONResponse(status_code=status, content=payload)

    def _find_csv():
        here = _Path(__file__).resolve()
        candidates = [
            here.parents[2] / "data" / "nasa_features_rul.csv",
            here.parents[2] / "analysis" / "nasa_features_rul.csv",
            here.parents[3] / "analysis" / "nasa_features_rul.csv",
            here.parents[3] / "data" / "nasa_features_rul.csv",
            _Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/nasa_features_rul.csv"),
            _Path("/Users/velocitygoal/battery-rul-v11/analysis/nasa_features_rul.csv"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _clean(x):
        try:
            import numpy as _np
            if isinstance(x, dict):
                return {str(k): _clean(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_clean(v) for v in x]
            if hasattr(x, "item"):
                return _clean(x.item())
            if isinstance(x, float) and (not _math.isfinite(x)):
                return None
            return x
        except Exception:
            return x

    if path == "/api/fixed4/compare":
        try:
            import pandas as _pd
            import numpy as _np

            csv_path = _find_csv()
            if csv_path is None:
                return _json(500, {"ok": False, "error": "nasa_features_rul.csv not found"})

            df = _pd.read_csv(csv_path)
            if "battery_id" not in df.columns and "battery" in df.columns:
                df["battery_id"] = df["battery"].astype(str)
            if "cycle" not in df.columns and "cycle_num" in df.columns:
                df["cycle"] = df["cycle_num"]
            if "cycle_num" not in df.columns and "cycle" in df.columns:
                df["cycle_num"] = df["cycle"]

            df["battery_id"] = df["battery_id"].astype(str)
            df["cycle"] = df["cycle"].astype(int)

            metric = str(request.query_params.get("metric", "capacity_pct"))
            cohort = str(request.query_params.get("cohort", "all"))
            bids_raw = str(request.query_params.get("batteries", "B0018,B0043"))
            bids = [b.strip() for b in bids_raw.split(",") if b.strip()]

            metric_map = {
                "soh": ("soh", "SoH (%)", "soh_pct"),
                "capacity_pct": ("capacity_mean", "Capacity (% of initial)", "cap_pct"),
                "capacity_mean": ("capacity_mean", "Capacity mean", "raw"),
                "impedance_sum": ("impedance_sum", "Impedance sum (Ω)", "raw"),
                "dcr": ("dcr", "DCR (Ω)", "raw"),
                "thermal_stress": ("thermal_stress", "Thermal stress", "raw"),
                "temperature_mean": ("temperature_mean", "Temp mean (°C)", "raw"),
                "lli": ("lli", "LLI", "raw"),
                "lam": ("lam", "LAM", "raw"),
            }
            metric_key, metric_label, mode = metric_map.get(metric, metric_map["capacity_pct"])

            def apply_cohort_filter(d):
                if cohort == "all" or "ambient_temp_c" not in d.columns:
                    return d
                t = d["ambient_temp_c"].astype(float)
                if cohort == "temp_le_10":
                    return d[t <= 10.0]
                if cohort == "temp_10_25":
                    return d[(t > 10.0) & (t < 25.0)]
                if cohort == "temp_ge_25":
                    return d[t >= 25.0]
                return d

            df_band = apply_cohort_filter(df.copy())

            band = {"x": [], "median": [], "q25": [], "q75": []}
            if metric_key in df_band.columns:
                if mode == "cap_pct":
                    tmp = df_band[["battery_id", "cycle", "capacity_mean"]].dropna().copy()
                    if not tmp.empty:
                        first = tmp.sort_values(["battery_id", "cycle"]).groupby("battery_id")["capacity_mean"].first().rename("cap0")
                        tmp = tmp.join(first, on="battery_id")
                        tmp = tmp[tmp["cap0"].astype(float) > 0]
                        tmp["value"] = tmp["capacity_mean"].astype(float) / tmp["cap0"].astype(float) * 100.0
                    else:
                        tmp["value"] = []
                else:
                    tmp = df_band[["cycle", metric_key]].dropna().copy()
                    tmp["value"] = tmp[metric_key].astype(float)
                    if mode == "soh_pct":
                        tmp["value"] = tmp["value"] * 100.0

                if not tmp.empty:
                    g = tmp.groupby("cycle")["value"]
                    med = g.median()
                    q25 = g.quantile(0.25)
                    q75 = g.quantile(0.75)
                    xs = [int(x) for x in med.index]
                    band = {
                        "x": xs,
                        "median": [float(med.loc[x]) for x in med.index],
                        "q25": [float(q25.loc[x]) for x in med.index],
                        "q75": [float(q75.loc[x]) for x in med.index],
                    }

            series = []
            for bid in bids:
                d1 = df[df["battery_id"] == str(bid)].sort_values("cycle").copy()
                if d1.empty or metric_key not in d1.columns:
                    series.append({"battery": bid, "x": [], "y": [], "missing": True})
                    continue

                x = d1["cycle"].astype(int).tolist()
                if mode == "cap_pct":
                    cap0 = float(d1["capacity_mean"].dropna().iloc[0]) if d1["capacity_mean"].dropna().shape[0] else float("nan")
                    if cap0 > 0:
                        y = (d1["capacity_mean"].astype(float) / cap0 * 100.0).tolist()
                    else:
                        y = []
                else:
                    vals = d1[metric_key].astype(float)
                    if mode == "soh_pct":
                        vals = vals * 100.0
                    y = vals.tolist()
                series.append({"battery": bid, "x": x, "y": [None if not _np.isfinite(v) else float(v) for v in y]})

            return _json(200, _clean({
                "ok": True,
                "source": str(csv_path),
                "metric": metric,
                "metric_key": metric_key,
                "metric_label": metric_label,
                "mode": mode,
                "cohort": cohort,
                "series": series,
                "band": band,
            }))
        except Exception as e:
            return _json(500, {"ok": False, "error": f"fixed4 compare failed: {e}"})

    m = _re.fullmatch(r"/api/live-reinfer/([^/]+)", path)
    if m:
        battery_id = m.group(1)
        r_ratio = str(request.query_params.get("r_ratio", request.query_params.get("rRatio", "0.1")))
        timeout = int(float(request.query_params.get("timeout", "360")))

        here = _Path(__file__).resolve()
        project = here.parents[2]
        repo = here.parents[3]
        candidates = [
            repo / "export_rul_dashboard_data_meta_fixed.py",
            repo / "export_rul_dashboard_data_meta_fixed_v4.py",
            repo / "run_bmaml_reinfer.py",
            repo / "run_live_reinfer.py",
            repo / "bmaml_svgd_reinfer.py",
            project / "export_rul_dashboard_data_meta_fixed.py",
            project / "backend" / "export_rul_dashboard_data_meta_fixed.py",
        ]
        script = next((p for p in candidates if p.exists()), None)
        if script is None:
            return _json(500, {"ok": False, "mode": "live_model_reinfer", "error": "No live model runner script found", "searched": [str(p) for p in candidates]})

        env = dict(_os.environ)
        env.update({
            "DEFAULT_R_RATIO": r_ratio,
            "R_RATIO": r_ratio,
            "BATTERY_ID": battery_id,
            "TARGET_BATTERY": battery_id,
            "LIVE_REINFER": "1",
            "PYTHONPATH": f"{repo}:{project}:{env.get('PYTHONPATH', '')}",
        })
        py = _sys.executable or "python3"
        cmds = [
            [py, str(script), "--battery", battery_id, "--r_ratio", r_ratio],
            [py, str(script), "--battery_id", battery_id, "--r_ratio", r_ratio],
            [py, str(script), "--target_battery", battery_id, "--r_ratio", r_ratio],
            [py, str(script), "--r_ratio", r_ratio],
            [py, str(script)],
        ]
        attempts = []
        started = _time.time()
        for cmd in cmds:
            try:
                proc = _subprocess.run(cmd, cwd=str(repo), env=env, text=True, capture_output=True, timeout=timeout)
                attempts.append({"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2500:], "stderr_tail": proc.stderr[-2500:]})
                if proc.returncode == 0:
                    return _json(200, {"ok": True, "mode": "live_model_reinfer", "battery": battery_id, "r_ratio": r_ratio, "script": str(script), "cmd": cmd, "elapsed_sec": round(_time.time() - started, 2), "stdout_tail": proc.stdout[-2500:], "stderr_tail": proc.stderr[-2500:]})
            except Exception as e:
                attempts.append({"cmd": cmd, "exception": str(e)})
        return _json(500, {"ok": False, "mode": "live_model_reinfer", "script": str(script), "error": "runner found but all invocations failed", "attempts": attempts})

    return await call_next(request)
# --- END HOTFIX_V19_FIXED4_COMPARE_AND_LIVE_REINFER ---


# --- HOTFIX_V18_LIVE_REINFER_ENDPOINT ---
@app.middleware("http")
async def _hotfix_v18_live_reinfer_middleware(request, call_next):
    import re as _re
    if request.method.upper() in {"GET", "POST"}:
        m = _re.fullmatch(r"/api/battery/([^/]+)/reinfer", request.url.path)
        if m:
            import os as _os
            import sys as _sys
            import time as _time
            import subprocess as _subprocess
            from pathlib import Path as _Path
            try:
                from fastapi.responses import JSONResponse as _JSONResponse
            except Exception:
                _JSONResponse = JSONResponse

            battery_id = m.group(1)
            r_ratio = str(request.query_params.get("r_ratio", request.query_params.get("rRatio", "0.1")))
            timeout = int(float(request.query_params.get("timeout", "360")))

            here = _Path(__file__).resolve()
            project = here.parents[2]
            repo = here.parents[3]
            candidates = [
                repo / "export_rul_dashboard_data_meta_fixed.py",
                repo / "export_rul_dashboard_data_meta_fixed_v4.py",
                repo / "run_bmaml_reinfer.py",
                repo / "run_live_reinfer.py",
                repo / "bmaml_svgd_reinfer.py",
                project / "export_rul_dashboard_data_meta_fixed.py",
                project / "backend" / "export_rul_dashboard_data_meta_fixed.py",
            ]
            script = next((p for p in candidates if p.exists()), None)
            if script is None:
                return _JSONResponse(status_code=500, content={
                    "ok": False,
                    "mode": "live_model_reinfer",
                    "battery": battery_id,
                    "r_ratio": r_ratio,
                    "error": "No live model runner script found. Expected export_rul_dashboard_data_meta_fixed.py or run_bmaml_reinfer.py.",
                    "searched": [str(p) for p in candidates],
                })

            env = dict(_os.environ)
            env.update({
                "DEFAULT_R_RATIO": r_ratio,
                "R_RATIO": r_ratio,
                "BATTERY_ID": battery_id,
                "TARGET_BATTERY": battery_id,
                "LIVE_REINFER": "1",
                "PYTHONPATH": f"{repo}:{project}:{env.get('PYTHONPATH', '')}",
            })
            py = _sys.executable or "python3"
            cmd_variants = [
                [py, str(script), "--battery", battery_id, "--r_ratio", r_ratio],
                [py, str(script), "--battery_id", battery_id, "--r_ratio", r_ratio],
                [py, str(script), "--target_battery", battery_id, "--r_ratio", r_ratio],
                [py, str(script), "--r_ratio", r_ratio],
                [py, str(script)],
            ]
            started = _time.time()
            attempts = []
            for cmd in cmd_variants:
                try:
                    proc = _subprocess.run(cmd, cwd=str(repo), env=env, text=True, capture_output=True, timeout=timeout)
                    attempts.append({"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2500:], "stderr_tail": proc.stderr[-2500:]})
                    if proc.returncode == 0:
                        return _JSONResponse(content={
                            "ok": True,
                            "mode": "live_model_reinfer",
                            "battery": battery_id,
                            "r_ratio": r_ratio,
                            "script": str(script),
                            "cmd": cmd,
                            "elapsed_sec": round(_time.time() - started, 2),
                            "stdout_tail": proc.stdout[-2500:],
                            "stderr_tail": proc.stderr[-2500:],
                            "note": "Live model/export runner completed. Frontend will refetch updated inference package.",
                        })
                except Exception as e:
                    attempts.append({"cmd": cmd, "exception": str(e)})
            return _JSONResponse(status_code=500, content={
                "ok": False,
                "mode": "live_model_reinfer",
                "battery": battery_id,
                "r_ratio": r_ratio,
                "script": str(script),
                "error": "Live model runner was found but all invocation patterns failed.",
                "attempts": attempts,
            })
    return await call_next(request)
# --- END HOTFIX_V18_LIVE_REINFER_ENDPOINT ---


# --- HOTFIX_V11_FIXED4_EXACT_DEGRADATION ---
# This middleware intentionally intercepts degradation-monitoring/degradation-report
# before the older FastAPI routes. It reproduces fixed4's capacity anomaly logic:
#   1) capacity band = per-battery normalized capacity% cohort band
#   2) anomaly = prefix min/max z-score up to the selected/current cycle
try:
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    JSONResponse = None  # type: ignore


def _hotfix_v11_find_nasa_features_csv() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "data" / "nasa_features_rul.csv",
        here.parents[2] / "analysis" / "nasa_features_rul.csv",
        Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/nasa_features_rul.csv"),
        Path("/Users/velocitygoal/battery-rul-v11/analysis/nasa_features_rul.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("nasa_features_rul.csv not found in expected project/data or analysis paths")


def _hotfix_v11_load_cycle_features_df() -> pd.DataFrame:
    csv_path = _hotfix_v11_find_nasa_features_csv()
    df = pd.read_csv(csv_path)
    if "battery_id" not in df.columns:
        if "battery" in df.columns:
            df["battery_id"] = df["battery"].astype(str)
        else:
            raise KeyError("CSV missing both battery_id and battery columns")
    if "cycle" not in df.columns:
        if "cycle_num" in df.columns:
            df = df.rename(columns={"cycle_num": "cycle"})
        else:
            raise KeyError("CSV missing both cycle and cycle_num columns")
    df["battery_id"] = df["battery_id"].astype(str)
    df["cycle"] = df["cycle"].astype(int)
    if "cycle_num" not in df.columns:
        df["cycle_num"] = df["cycle"]
    return df


def _hotfix_v11_robust_scale(q25: float, q75: float) -> float:
    return max(float(q75 - q25) / 1.349, 1e-9)


def _hotfix_v11_z(value: float, med: float, q25: float, q75: float) -> float:
    try:
        if not (np.isfinite(value) and np.isfinite(med) and np.isfinite(q25) and np.isfinite(q75)):
            return float("nan")
        return float((float(value) - float(med)) / _hotfix_v11_robust_scale(q25, q75))
    except Exception:
        return float("nan")


def _hotfix_v11_onset(z_values: list[float], cycles: list[int], thresh: float, direction: str, min_run: int) -> int | None:
    run = 0
    start_idx = None
    for i, z in enumerate(z_values):
        if not np.isfinite(z):
            ok = False
        elif direction == "pos":
            ok = z >= thresh
        else:
            ok = z <= -thresh
        if ok:
            if run == 0:
                start_idx = i
            run += 1
            if run >= min_run and start_idx is not None:
                return int(cycles[start_idx])
        else:
            run = 0
            start_idx = None
    return None


_HOTFIX_V11_DRIVER_TAGS = {
    "thermal_stress": ("고온/열 스트레스", "열관리 점검(팬/냉각)·고온 구간 제한"),
    "temperature_mean": ("고온 노출", "냉각/통풍·고온 운행 제한"),
    "temp_rise_cycle": ("셀 발열 증가", "열 runaway 위험 체크·냉각 강화"),
    "eff_c_rate": ("고 C-rate(고부하)", "가속/급속충전 제한·부하 분산"),
    "current_max": ("고부하(충전/회생)", "피크 전류 제한·회생제동 설정 조정"),
    "current_min": ("고부하(방전)", "피크 방전 전류 제한·부하 분산"),
    "voltage_min": ("깊은 방전(DoD↑)", "최저 SoC 제한·운영전략 조정"),
    "dvdt_max_abs": ("전압 급변", "BMS 로깅/센서 점검·전력 프로파일 확인"),
    "dTdt_max": ("온도 급상승", "열관리/센서 점검·운행 제한"),
}


def _hotfix_v11_expected_band(df: pd.DataFrame, metric: str) -> dict[int, dict[str, float]]:
    if metric not in df.columns:
        return {}
    tmp = df[["cycle", metric]].dropna()
    if tmp.empty:
        return {}
    g = tmp.groupby("cycle")[metric]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)
    return {
        int(c): {"median": float(med.loc[c]), "q1": float(q25.loc[c]), "q3": float(q75.loc[c])}
        for c in med.index
    }


def _hotfix_v11_capacity_pct_band(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    if "capacity_mean" not in df.columns:
        return {}
    tmp = df[["battery_id", "cycle", "capacity_mean"]].dropna().copy()
    if tmp.empty:
        return {}
    first = (
        tmp.sort_values(["battery_id", "cycle"])
        .groupby("battery_id")["capacity_mean"]
        .first()
        .rename("cap0")
    )
    tmp = tmp.join(first, on="battery_id")
    tmp = tmp[tmp["cap0"].astype(float) > 0]
    tmp["cap_pct"] = tmp["capacity_mean"].astype(float) / tmp["cap0"].astype(float) * 100.0
    g = tmp.groupby("cycle")["cap_pct"]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)
    return {
        int(c): {"median": float(med.loc[c]), "q1": float(q25.loc[c]), "q3": float(q75.loc[c])}
        for c in med.index
    }


def _hotfix_v11_build_degradation_payload(battery_id: str, r_ratio: float = 0.1, cycle: int | None = None) -> dict:
    df_all = _hotfix_v11_load_cycle_features_df()
    df_b = df_all[df_all["battery_id"] == str(battery_id)].sort_values("cycle").copy()
    if df_b.empty:
        raise HTTPException(status_code=404, detail=f"No data for battery {battery_id}")

    min_cycle = int(df_b["cycle"].min())
    max_cycle = int(df_b["cycle"].max())
    if cycle is None:
        current_cycle = max_cycle
    else:
        try:
            current_cycle = int(cycle)
        except Exception:
            current_cycle = max_cycle
        if current_cycle <= 0:
            current_cycle = min_cycle
        if current_cycle > max_cycle:
            current_cycle = max_cycle

    bands: dict[str, dict[int, dict[str, float]]] = {}
    for feat in [
        "dcr", "impedance_sum", "thermal_stress", "temperature_mean", "temp_rise_cycle",
        "eff_c_rate", "current_max", "current_min", "voltage_min", "dvdt_max_abs", "dTdt_max",
        "soh",
    ]:
        b = _hotfix_v11_expected_band(df_all, feat)
        if b:
            bands[feat] = b

    cap_band = _hotfix_v11_capacity_pct_band(df_all)
    if cap_band:
        # capacity_mean intentionally contains capacity% band for chart compatibility
        # because the React chart plots Capacity (% of initial), not raw Ah.
        bands["capacity_mean"] = cap_band
        bands["capacity_pct"] = cap_band

    cap0 = float(df_b.iloc[0]["capacity_mean"]) if "capacity_mean" in df_b.columns else float("nan")

    dcr_z_series: list[dict] = []
    cap_z_series: list[dict] = []
    for _, row in df_b.iterrows():
        cyc = int(row["cycle"])

        if "dcr" in row.index and "dcr" in bands and cyc in bands["dcr"] and pd.notna(row["dcr"]):
            b = bands["dcr"][cyc]
            z = _hotfix_v11_z(float(row["dcr"]), b["median"], b["q1"], b["q3"])
            dcr_z_series.append({"cycle": cyc, "z": z})

        if "capacity_mean" in row.index and cap_band and cyc in cap_band and pd.notna(row["capacity_mean"]) and np.isfinite(cap0) and cap0 > 0:
            cap_pct = float(row["capacity_mean"]) / cap0 * 100.0
            b = cap_band[cyc]
            z = _hotfix_v11_z(cap_pct, b["median"], b["q1"], b["q3"])
            cap_z_series.append({"cycle": cyc, "z": z})

    # fixed4 online-mode evidence: prefix only, up to current cycle.
    dcr_prefix = [p for p in dcr_z_series if int(p["cycle"]) <= current_cycle and np.isfinite(p.get("z", float("nan")))]
    cap_prefix = [p for p in cap_z_series if int(p["cycle"]) <= current_cycle and np.isfinite(p.get("z", float("nan")))]

    dcr_max_z = max([float(p["z"]) for p in dcr_prefix], default=float("-inf"))
    cap_min_z = min([float(p["z"]) for p in cap_prefix], default=float("inf"))

    dcr_onset = _hotfix_v11_onset([float(p["z"]) for p in dcr_prefix], [int(p["cycle"]) for p in dcr_prefix], 3.0, "pos", 2)
    cap_onset = _hotfix_v11_onset([float(p["z"]) for p in cap_prefix], [int(p["cycle"]) for p in cap_prefix], 3.0, "neg", 3)

    issues = []
    if np.isfinite(dcr_max_z) and dcr_max_z >= 4.0:
        issues.append({
            "type": "dcr_spike",
            "label": "Fault-like anomaly (DCR spike)",
            "severity": "HIGH" if abs(dcr_max_z) >= 6.0 else "MED",
            "onsetCycle": dcr_onset,
            "zValue": float(dcr_max_z),
        })
    if np.isfinite(cap_min_z) and cap_min_z <= -3.5:
        issues.append({
            "type": "capacity_drop",
            "label": "Accelerated degradation (Capacity drop)",
            "severity": "HIGH" if abs(cap_min_z) >= 6.0 else "MED",
            "onsetCycle": cap_onset,
            "zValue": float(cap_min_z),
        })

    ref_cycle = current_cycle
    if dcr_onset is not None:
        ref_cycle = int(dcr_onset)
    elif cap_onset is not None:
        ref_cycle = int(cap_onset)

    ref_rows = df_b[df_b["cycle"] == ref_cycle]
    ref_row = ref_rows.iloc[0] if not ref_rows.empty else df_b.iloc[-1]

    drivers = []
    for feat, (tag, action) in _HOTFIX_V11_DRIVER_TAGS.items():
        if feat not in df_all.columns or feat not in ref_row.index or feat not in bands:
            continue
        if ref_cycle not in bands[feat]:
            continue
        val = ref_row[feat]
        if pd.isna(val):
            continue
        b = bands[feat][ref_cycle]
        if float(b["q3"] - b["q1"]) < 1e-6:
            continue
        z = _hotfix_v11_z(float(val), b["median"], b["q1"], b["q3"])
        if not np.isfinite(z):
            continue
        drivers.append({
            "feature": feat,
            "label": tag,
            "tag": tag,
            "value": float(val),
            "z": float(z),
            "absZ": abs(float(z)),
            "recommendation": action,
            "action": action,
        })
    drivers = sorted(drivers, key=lambda d: d["absZ"], reverse=True)[:3]

    significant_drivers = [d for d in drivers if float(d.get("absZ", 0.0)) >= 3.0]
    has_early_warning = (not issues) and bool(significant_drivers)
    status = "major-anomaly" if issues else ("early-warning" if has_early_warning else "normal")

    lines = [
        f"# {battery_id} Degradation Report (fixed4-compatible)",
        f"- Current Cycle: {current_cycle}",
        f"- r_ratio: {r_ratio}",
        "",
        "## Findings",
    ]
    if not issues:
        lines.append("- No major deviation detected by KPI thresholds (DCR/Capacity%).")
    else:
        for it in issues:
            onset = it.get("onsetCycle")
            onset_txt = f"cycle {onset}" if onset is not None else "(onset 미확정)"
            lines.append(f"- {it['label']} · severity: {it['severity']} · onset: {onset_txt} · robust z: {it['zValue']:.2f}")

    if drivers:
        lines += ["", f"## Potential drivers around cycle {ref_cycle}"]
        for d in drivers:
            lines.append(f"- {d['label']} ({d['feature']}={d['value']:.4g}, z={d['z']:.2f}) — {d.get('recommendation','')}")

    df_out = df_b.copy()
    if "cycle_num" not in df_out.columns:
        df_out["cycle_num"] = df_out["cycle"]

    report = "\n".join(lines)

    return _sanitize_json({
        "battery": battery_id,
        "battery_id": battery_id,
        "r_ratio": float(r_ratio),
        "cycle": int(current_cycle),
        "cycles": [int(x) for x in df_b["cycle"].tolist()],
        "series": df_out.to_dict(orient="records"),
        "rows": df_out.to_dict(orient="records"),
        "bands": {feat: {int(k): v for k, v in stats.items()} for feat, stats in bands.items()},
        "z_scores": {"dcr": dcr_z_series, "capacity": cap_z_series},
        "zSeries": {"dcr": dcr_z_series, "capacity": cap_z_series},
        "dcr_max_z": None if not np.isfinite(dcr_max_z) else float(dcr_max_z),
        "cap_min_z": None if not np.isfinite(cap_min_z) else float(cap_min_z),
        "dcr_onset_cycle": dcr_onset,
        "cap_onset_cycle": cap_onset,
        "issues": issues,
        "majorAlerts": issues,
        "status": status,
        "earlyWarning": {
            "active": has_early_warning,
            "message": "핵심 KPI(DCR/Capacity%) 기준의 큰 이탈은 아직 없지만, 일부 스트레스 신호(driver)가 cohort 대비 outlier 입니다." if has_early_warning else None,
        },
        "drivers": drivers,
        "potentialDrivers": drivers,
        "normalMessage": "현재 선택된 배터리는 reference cohort의 기대 범위 내에서 큰 이탈이 관측되지 않았습니다.",
        "reportMarkdown": report,
        "markdown": report,
    })


@app.middleware("http")
async def _hotfix_v11_fixed4_degradation_middleware(request, call_next):
    if JSONResponse is not None:
        m = re.fullmatch(r"/api/battery/([^/]+)/(degradation-monitoring|degradation-report)", request.url.path)
        if m:
            try:
                battery_id = m.group(1)
                r_ratio = float(request.query_params.get("r_ratio", "0.1"))
                cycle_raw = request.query_params.get("cycle")
                cycle = int(cycle_raw) if cycle_raw not in (None, "", "null", "undefined") else None
                payload = _hotfix_v11_build_degradation_payload(battery_id=battery_id, r_ratio=r_ratio, cycle=cycle)
                return JSONResponse(content=payload)
            except HTTPException as e:
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
            except Exception as e:
                return JSONResponse(status_code=500, content={"detail": f"HOTFIX_V11 fixed4 degradation error: {e}"})
    return await call_next(request)
# --- END HOTFIX_V11_FIXED4_EXACT_DEGRADATION ---

def _sanitize_json(obj: Any) -> Any:
    # Convert NaN/Inf to None recursively so JSON encoding never crashes.
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [_sanitize_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    return obj

def _read_bytes_retry(p: Path, retries: int = 30) -> bytes:
    # Aggressive retry for Errno 35 (Resource deadlock avoided).
    for i in range(retries):
        try:
            with _FILE_LOCK:
                return p.read_bytes()
        except OSError as e:
            if getattr(e, "errno", None) == 35 and i < retries - 1:
                time.sleep(min(0.08 * (i + 1), 1.0))
                continue
            raise

def _battery_from_name(name: str) -> str | None:
    m = re.search(r"(B\d{4})", name)
    return m.group(1) if m else None

def _parse_r_from_name(name: str) -> float | None:
    # Matches your filenames:
    #   battery_B0018_r0p25.json  -> 0.25
    #   battery_B0018_r0p00.json  -> 0.00
    m = re.search(r"_r(\d+)p(\d+)", name)
    if m:
        return round(float(f"{m.group(1)}.{m.group(2)}"), 2)

    # Also accept dot format if ever present
    m = re.search(r"_r(\d+(?:\.\d+)?)", name)
    if m:
        v = float(m.group(1))
        if v > 1.0 and v <= 100.0:  # percent-like
            v /= 100.0
        return round(v, 2)

    return None

def _load_precomputed_all() -> None:
    # Load all allowed precomputed JSONs into memory at startup.
    _PRECOMP_PAYLOAD.clear()
    _PRECOMP_FILE.clear()
    _PRECOMP_R_BY_BAT.clear()

    precomp = getattr(settings, "PRECOMP_DIR", None)
    if not precomp:
        return
    root = Path(precomp)
    if not root.exists():
        return

    for fp in sorted(root.rglob("*.json")):
        bid = _battery_from_name(fp.name)
        if not bid:
            continue
        r = _parse_r_from_name(fp.name)
        if r is None or r not in _ALLOWED_SET:
            continue
        try:
            raw = _read_bytes_retry(fp)
            payload = _sanitize_json(json.loads(raw.decode("utf-8", errors="replace")))
        except Exception:
            continue
        key = (bid, r)
        _PRECOMP_PAYLOAD[key] = payload
        _PRECOMP_FILE[key] = str(fp.relative_to(root))

    for b in sorted({bid for (bid, _r) in _PRECOMP_PAYLOAD.keys()}):
        rs = sorted({r for (bid, r) in _PRECOMP_PAYLOAD.keys() if bid == b})
        _PRECOMP_R_BY_BAT[b] = rs if rs else ALLOWED_R

def _load_cref_map() -> dict:
    p = Path(settings.META_CREF_PATH)
    if not p.exists():
        return {}
    raw = _read_bytes_retry(p)
    obj = json.loads(raw.decode("utf-8", errors="replace"))
    return obj.get("c_ref_ahr_by_battery", {})

@app.on_event("startup")
def _startup():
    _load_precomputed_all()
    # Mount precomputed data directory for static file serving
    precomp = getattr(settings, "PRECOMP_DIR", None)
    if precomp and Path(precomp).exists():
        app.mount("/precomputed", StaticFiles(directory=precomp), name="precomputed")

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "precomputed_loaded": len(_PRECOMP_PAYLOAD),
        "precomputed_batteries": sorted(_PRECOMP_R_BY_BAT.keys()),
        "allowed_r": ALLOWED_R,
    }

@app.get("/api/batteries")
def batteries():
    if _PRECOMP_R_BY_BAT:
        return {"batteries": sorted(_PRECOMP_R_BY_BAT.keys())}
    return {"batteries": ["B0018", "B0042", "B0043"]}

@app.get("/api/battery/{battery_id}/r_ratios")
def battery_r_ratios(battery_id: str):
    rs = _PRECOMP_R_BY_BAT.get(battery_id)
    return {"battery": battery_id, "r_ratios": rs if rs else ALLOWED_R}

@app.get("/api/battery/{battery_id}/meta")
def battery_meta(battery_id: str):
    cref = _load_cref_map().get(battery_id)
    return {"battery": battery_id, "c_ref_ahr": cref}

@app.get("/api/battery/{battery_id}/cycles")
def battery_cycles(
    battery_id: str,
    start: int | None = None,
    end: int | None = None,
    stride: int = Query(default=1, ge=1, le=50),
    cols: str | None = None,
):
    cols_list = [c.strip() for c in cols.split(",")] if cols else None
    df = fetch_cycles(battery_id, start, end, stride, cols_list)
    if df.empty:
        raise HTTPException(status_code=404, detail="No rows found for this battery/range")
    try:
        df = add_eol_columns(df)
    except Exception:
        pass
    df = df.replace([float("inf"), float("-inf")], None)
    df = df.where(df.notna(), None)
    return {"rows": df.to_dict(orient="records")}

@app.get("/api/battery/{battery_id}/precomputed")
def battery_precomputed(
    battery_id: str,
    r_ratio: float = Query(default=0.25, ge=0.0, le=1.0),
):
    r = round(float(r_ratio), 2)
    key = (battery_id, r)
    payload = _PRECOMP_PAYLOAD.get(key)
    if payload is None:
        return {
            "battery": battery_id,
            "r_ratio": r,
            "error": "precomputed json not found (or not loaded).",
            "available_r": _PRECOMP_R_BY_BAT.get(battery_id, ALLOWED_R),
        }
    return {"battery": battery_id, "r_ratio": r, "file": _PRECOMP_FILE.get(key), "payload": payload}

@app.get("/api/precomputed/{tag}")
def precomputed_batch(tag: str):
    """Fetch all precomputed items for a specific tag (r-ratio).
    Tag format: 'r0p10', 'r0p20', etc."""
    m = re.search(r"r(\d+)p(\d+)", tag)
    if not m:
        raise HTTPException(status_code=400, detail="Invalid tag format. Use 'r0p10', 'r0p20', etc.")

    r = round(float(f"{m.group(1)}.{m.group(2)}"), 2)
    if r not in _ALLOWED_SET:
        raise HTTPException(status_code=400, detail=f"r_ratio {r} not in allowed set {ALLOWED_R}")

    items = []
    for (bid, ratio), payload in _PRECOMP_PAYLOAD.items():
        if ratio == r:
            items.append(payload)

    if not items:
        raise HTTPException(status_code=404, detail=f"No precomputed data found for tag {tag}")

    return {"tag": tag, "r_ratio": r, "items": items}



@app.post("/api/battery/{battery_id}/reinfer")
def battery_reinfer(
    battery_id: str,
    r_ratio: float = Query(default=0.10, ge=0.0, le=1.0),
    timeout: int = Query(default=900, ge=30, le=3600),
):
    import json
    import os
    import shutil as _shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    ckpt = Path("/Users/velocitygoal/battery-rul-v11/core_checkpoints/nasa_bmaml_best_re.pt")
    runner = Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/scripts/prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py")
    live_root = Path("/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/live_reinfer_results")
    pyroots = ['/Users/velocitygoal/battery_project/v11/server/backend', '/Users/velocitygoal/battery_project/v11', '/Users/velocitygoal/battery-rul-v11']

    if not ckpt.exists():
        raise HTTPException(status_code=500, detail=f"checkpoint not found: {ckpt}")
    if not runner.exists():
        raise HTTPException(status_code=500, detail=f"runner not found: {runner}")

    r = round(float(r_ratio), 2)
    tag = "r" + f"{r:.2f}".replace(".", "p")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = live_root / f"{battery_id}_{tag}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_json = out_dir / f"{battery_id}_viz_meta_{tag}.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(pyroots + [env.get("PYTHONPATH", "")])
    env["CHECKPOINT_PATH"] = str(ckpt)
    env["BMAML_CHECKPOINT"] = str(ckpt)
    env["PRECOMP_DIR"] = str(out_dir)

    direct_pythons = []
    for p in [
        "/opt/anaconda3/envs/battery-maml/bin/python",
        "/Users/velocitygoal/anaconda3/envs/battery-maml/bin/python",
        "/Users/velocitygoal/miniforge3/envs/battery-maml/bin/python",
        "/Users/velocitygoal/mambaforge/envs/battery-maml/bin/python",
    ]:
        if Path(p).exists() and p not in direct_pythons:
            direct_pythons.append(p)

    if sys.executable and sys.executable not in direct_pythons:
        direct_pythons.append(sys.executable)

    which_py = _shutil.which("python")
    if which_py and which_py not in direct_pythons:
        direct_pythons.append(which_py)

    base_args = [
        str(runner),
        "--ckpt", str(ckpt),
        "--eval_dataset", "from_ckpt",
        "--r_ratio", str(r),
        "--bids", str(battery_id),
        "--out_dir", str(out_dir),
        "--save_json", "1",
        "--save_batch_json", "1",
        "--min_support", "0",
        "--cap_before_eol", "1",
        "--ratio_base", "pos",
    ]

    attempts = []
    started = time.time()

    def finish_if_success(proc, label, cmd_repr):
        if proc.returncode == 0 and expected_json.exists():
            item = json.loads(expected_json.read_text(encoding="utf-8"))
            return {
                "ok": True,
                "success": True,
                "mode": "live_bmaml_reinfer_session_only",
                "battery": battery_id,
                "r_ratio": r,
                "tag": tag,
                "checkpoint": str(ckpt),
                "runner": str(runner),
                "executor": label,
                "cmd": cmd_repr,
                "out_dir": str(out_dir),
                "json_path": str(expected_json),
                "baseline_overwritten": False,
                "elapsed_sec": round(time.time() - started, 2),
                "item": item,
                "stdout_tail": proc.stdout[-2500:],
                "stderr_tail": proc.stderr[-2500:],
            }
        return None

    for py in direct_pythons:
        cmd = [py] + base_args
        try:
            proc = subprocess.run(cmd, cwd=pyroots[0], env=env, text=True, capture_output=True, timeout=int(timeout))
            attempts.append({"executor": py, "returncode": proc.returncode, "stdout_tail": proc.stdout[-1800:], "stderr_tail": proc.stderr[-1800:]})
            ok = finish_if_success(proc, py, cmd)
            if ok:
                return ok
        except Exception as e:
            attempts.append({"executor": py, "exception": str(e)})

    # Fallback through conda run. This matters because manual success used conda activate battery-maml.
    conda = _shutil.which("conda")
    if conda:
        cmd = [conda, "run", "-n", "battery-maml", "python"] + base_args
        try:
            proc = subprocess.run(cmd, cwd=pyroots[0], env=env, text=True, capture_output=True, timeout=int(timeout))
            attempts.append({"executor": "conda run -n battery-maml", "returncode": proc.returncode, "stdout_tail": proc.stdout[-1800:], "stderr_tail": proc.stderr[-1800:]})
            ok = finish_if_success(proc, "conda run -n battery-maml", cmd)
            if ok:
                return ok
        except Exception as e:
            attempts.append({"executor": "conda run -n battery-maml", "exception": str(e)})

    raise HTTPException(status_code=500, detail={
        "ok": False,
        "success": False,
        "mode": "live_bmaml_reinfer_session_only",
        "battery": battery_id,
        "r_ratio": r,
        "expected_json": str(expected_json),
        "baseline_overwritten": False,
        "error": "all runner attempts failed or expected JSON was not produced",
        "attempts": attempts,
    })


@app.get("/api/battery/{battery_id}/diagnostics")
def battery_diagnostics(
    battery_id: str,
    window: int = Query(default=200, ge=20, le=2000),
    z_thr: float = Query(default=-2.0),
    n_consec: int = Query(default=5, ge=1, le=50),
):
    cols = ["battery", "cycle_num", "pred_rul", "pred_rul_cycles", "true_rul", "true_eol", "rul_std"]
    df = fetch_cycles(battery_id, None, None, 1, cols)
    if df.empty:
        raise HTTPException(status_code=404, detail="No rows found")
    df = df.sort_values("cycle_num").tail(window)
    df = add_eol_columns(df)

    if "resid_rul" not in df.columns or "rul_std" not in df.columns:
        return {"battery": battery_id, "note": "residual/uncertainty not available with current columns"}

    resid = df["resid_rul"].to_numpy()
    sigma = df["rul_std"].to_numpy()
    cov = coverage_stats(resid, sigma)

    z = df.get("z_resid")
    z_arr = z.to_numpy() if z is not None else None
    alerts = None
    onset_cycle = None
    worst_z = None
    if z_arr is not None:
        worst_z = float(z_arr[np.isfinite(z_arr)].min()) if np.isfinite(z_arr).any() else None
        cond = np.isfinite(z_arr) & (z_arr < z_thr)
        idx = first_consecutive(cond, n_consec)
        if idx is not None:
            onset_cycle = int(df.iloc[idx]["cycle_num"])
        alerts = int(cond.sum())

    return {
        "battery": battery_id,
        **cov,
        "alerts_count": alerts,
        "worst_z": worst_z,
        "onset_cycle": onset_cycle,
        "window": int(window),
        "z_thr": float(z_thr),
        "n_consec": int(n_consec),
    }

@app.get("/api/battery/{battery_id}/feature-bands")
def battery_feature_bands(battery_id: str):
    """
    Compute feature statistics and z-scores for degradation monitoring.
    Uses all available batteries to establish expected bands (median ± IQR).
    """
    # Load data for current battery
    all_cols = [
        "cycle_num", "capacity_mean", "soh", "dcr", "impedance_sum",
        "temperature_mean", "thermal_stress", "dTdt_max", "dvdt_max_abs", "eff_c_rate",
        "temp_rise_cycle", "temperature_measured_max", "current_max",
        "dcr_growth", "impedance_growth"
    ]
    try:
        df_bat = _hotfix_fetch_cycles_any(battery_id, all_cols)
    except Exception as e:
        return {"battery": battery_id, "error": f"Could not load feature data: {str(e)}"}

    if df_bat.empty:
        return {"battery": battery_id, "error": f"No data for battery {battery_id}"}

    df_bat = df_bat.sort_values("cycle_num")

    # Load data for all other batteries to establish bands
    all_bids = ["B0018", "B0043", "B0042", "B0033"]
    dfs_all = []
    for bid in all_bids:
        try:
            df = _hotfix_fetch_cycles_any(bid, all_cols)
            if not df.empty:
                dfs_all.append(df)
        except Exception:
            pass

    if not dfs_all:
        return {"battery": battery_id, "error": "No reference data available"}

    df_all = pd.concat(dfs_all, ignore_index=True)

    # Compute bands for each feature
    features = [
        "capacity_mean", "soh", "dcr", "impedance_sum",
        "temperature_mean", "thermal_stress", "dTdt_max", "dvdt_max_abs", "eff_c_rate",
        "temp_rise_cycle", "temperature_measured_max", "current_max",
        "dcr_growth", "impedance_growth"
    ]
    bands = {}

    for feat in features:
        if feat not in df_all.columns:
            continue

        # Group by cycle and compute stats
        cycle_stats = []
        for cyc in sorted(df_all["cycle_num"].unique()):
            data = df_all[df_all["cycle_num"] == cyc][feat]
            if len(data) > 0:
                vals = data.dropna().values
                if len(vals) > 0:
                    med = float(np.median(vals))
                    q1 = float(np.percentile(vals, 25))
                    q3 = float(np.percentile(vals, 75))
                    iqr = q3 - q1
                    cycle_stats.append({
                        "cycle": int(cyc),
                        "median": med,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr if iqr > 0 else 1.0
                    })

        bands[feat] = cycle_stats

    # Compute z-scores for current battery
    z_scores = {}
    for feat in features:
        if feat not in df_bat.columns or feat not in bands:
            continue

        z_data = []
        band_dict = {s["cycle"]: s for s in bands[feat]}

        for _, row in df_bat.iterrows():
            cyc = int(row["cycle_num"])
            val = row[feat]

            if cyc in band_dict and not np.isnan(val):
                band = band_dict[cyc]
                z = (val - band["median"]) / band["iqr"] if band["iqr"] > 0 else 0.0
                z_data.append({"cycle": cyc, "z_score": float(z)})

        z_scores[feat] = z_data

    # Detect anomalies (|z| >= 2 for 3+ consecutive cycles)
    anomalies = {}
    for feat, z_data in z_scores.items():
        if not z_data:
            continue

        z_arr = np.array([abs(z["z_score"]) for z in z_data])
        cyc_arr = np.array([z["cycle"] for z in z_data])

        anomalies[feat] = {
            "max_z": float(np.max(z_arr)) if len(z_arr) > 0 else None,
            "onset_cycle": None
        }

        # Find first onset (3+ consecutive cycles with |z| >= 2)
        for i in range(len(z_arr) - 2):
            if z_arr[i] >= 2.0 and z_arr[i+1] >= 2.0 and z_arr[i+2] >= 2.0:
                anomalies[feat]["onset_cycle"] = int(cyc_arr[i])
                break

    return {
        "battery": battery_id,
        "bands": bands,
        "z_scores": z_scores,
        "anomalies": anomalies
    }


# --- HOTFIX V4: tolerant cycle fetch for degradation endpoints ---
def _hotfix_fetch_cycles_any(battery_id: str, preferred_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Degradation monitoring must work for all batteries visible in the UI.
    Try progressively safer fetches, then normalize the result so downstream logic can run.
    """
    attempts: list[list[str] | None] = []
    if preferred_cols:
        attempts.append(preferred_cols)
        core = [
            "cycle_num", "capacity_mean", "dcr", "soh", "impedance_sum",
            "thermal_stress", "temperature_mean", "temp_rise_cycle",
            "eff_c_rate", "current_max", "current_min", "voltage_min",
            "dvdt_max_abs", "dTdt_max",
        ]
        attempts.append([c for c in core if c in set(preferred_cols)])
    attempts.append(None)

    last_err: Exception | None = None
    for cols in attempts:
        try:
            df = fetch_cycles(battery_id, None, None, 1, cols)
            if df is not None and not df.empty:
                df = df.copy()
                if "cycle" in df.columns and "cycle_num" not in df.columns:
                    df["cycle_num"] = df["cycle"]
                if "battery" not in df.columns:
                    df["battery"] = battery_id
                if preferred_cols:
                    for c in preferred_cols:
                        if c not in df.columns:
                            df[c] = np.nan
                if "cycle_num" in df.columns:
                    df = df.sort_values("cycle_num")
                return df
        except Exception as e:
            last_err = e
            continue

    candidate_paths = []
    try:
        import os as _os
        for key in (
            "NASA_FEATURES_RUL_CSV", "NASA_FEATURES_PATH", "NASA_FEATURES_CSV",
            "FEATURES_CSV_PATH", "CSV_PATH", "DATA_PATH",
        ):
            v = _os.environ.get(key)
            if v:
                candidate_paths.append(Path(v))
        for attr in (
            "NASA_FEATURES_RUL_CSV", "NASA_FEATURES_PATH", "NASA_FEATURES_CSV",
            "FEATURES_CSV_PATH", "CSV_PATH", "DATA_PATH",
        ):
            v = getattr(settings, attr, None)
            if v:
                candidate_paths.append(Path(str(v)))
    except Exception:
        pass

    for p in candidate_paths:
        try:
            if not p.exists() or p.name != "nasa_features_rul.csv":
                continue
            df = pd.read_csv(p)
            if "battery_id" in df.columns:
                mask = df["battery_id"].astype(str) == str(battery_id)
            elif "battery" in df.columns:
                mask = df["battery"].astype(str) == str(battery_id)
            else:
                continue
            df = df.loc[mask].copy()
            if df.empty:
                continue
            if "cycle" in df.columns and "cycle_num" not in df.columns:
                df["cycle_num"] = df["cycle"]
            if "battery" not in df.columns:
                df["battery"] = battery_id
            if preferred_cols:
                for c in preferred_cols:
                    if c not in df.columns:
                        df[c] = np.nan
            if "cycle_num" in df.columns:
                df = df.sort_values("cycle_num")
            return df
        except Exception as e:
            last_err = e
            continue

    print(f"[HOTFIX V4] no cycle rows for {battery_id}; last_err={last_err}")
    return pd.DataFrame()
# --- END HOTFIX V4 ---

# =================================================
# Anomaly Report Endpoint (Fixed4 Logic)
# =================================================

def robust_scale_from_iqr(q1: float, q3: float) -> float:
    """Fixed4 robust z-score scaling: IQR / 1.349"""
    iqr = float(q3 - q1)
    return max(iqr / 1.349, 1e-9)


def compute_robust_z(value: float, median: float, q1: float, q3: float) -> float:
    """Fixed4 robust z-score: (value - median) / robust_scale"""
    scale = robust_scale_from_iqr(q1, q3)
    return float((value - median) / scale) if scale > 0 else 0.0


def find_onset_cycle(z_arr: np.ndarray, cycles_arr: np.ndarray, threshold: float, direction: str = "pos", min_run: int = 2) -> int | None:
    """Find first cycle where threshold is exceeded for min_run consecutive cycles"""
    if len(z_arr) == 0:
        return None

    if direction == "pos":
        cond = z_arr >= threshold
    else:  # "neg"
        cond = z_arr <= threshold

    idx = first_consecutive(cond, min_run)
    return int(cycles_arr[idx]) if idx is not None else None


@app.get("/api/battery/{battery_id}/degradation-report-legacy-disabled")
def battery_degradation_report(
    battery_id: str,
    r_ratio: float = Query(default=0.1, ge=0, le=1),
    cycle: int = Query(default=None),
):
    """
    Complete degradation report with rows, bands, and anomalies (for Degradation tab).

    Query parameters:
    - r_ratio: data split ratio (0.0-1.0)
    - cycle: specific cycle to analyze (defaults to latest)

    Returns:
    - rows: All cycle data for the battery
    - bands: Reference statistics (median, q1, q3) for each feature per cycle
    - zSeries: robust z-scores for DCR and Capacity%
    - majorAlerts: DCR spike or Capacity drop
    - earlyWarning: Driver outliers without major anomaly
    - potentialDrivers: Top 3 drivers by |z|
    - normalMessage: Message for normal state
    - markdown: Markdown report content
    """

    try:
        # Load battery data
        all_cols = [
            "cycle_num", "capacity_mean", "dcr", "soh", "impedance_sum",
            "thermal_stress", "temperature_mean", "temp_rise_cycle",
            "eff_c_rate", "current_max", "current_min", "voltage_min",
            "dvdt_max_abs", "dTdt_max"
        ]
        df_bat = _hotfix_fetch_cycles_any(battery_id, all_cols)

        if df_bat.empty:
            raise HTTPException(status_code=404, detail=f"No data for battery {battery_id}")

        df_bat = df_bat.sort_values("cycle_num")

        # Use provided cycle or default to latest
        if cycle is None:
            cycle = int(df_bat["cycle_num"].max())
        else:
            # Ensure cycle exists in data
            if cycle not in df_bat["cycle_num"].values:
                cycle = int(df_bat["cycle_num"].max())

        # Load cohort reference data
        cohort_batteries = ["B0018", "B0043", "B0042", "B0033"]
        dfs_all = []
        for bid in cohort_batteries:
            try:
                df = _hotfix_fetch_cycles_any(bid, all_cols)
                if not df.empty:
                    dfs_all.append(df)
            except Exception:
                pass

        if not dfs_all:
            dfs_all = [df_bat.copy()]
            print(f"[HOTFIX V4] using selected battery as reference fallback for {battery_id}")

        df_all = pd.concat(dfs_all, ignore_index=True)

        # Compute reference bands (median, q1, q3) per cycle
        driver_features = [
            "thermal_stress", "temperature_mean", "temp_rise_cycle",
            "eff_c_rate", "current_max", "current_min", "voltage_min",
            "dvdt_max_abs", "dTdt_max"
        ]
        all_features = ["dcr", "capacity_mean", "impedance_sum"] + driver_features

        bands = {}
        for feat in all_features:
            if feat not in df_all.columns:
                continue

            cycle_stats = {}
            for cyc in sorted(df_all["cycle_num"].unique()):
                data = df_all[df_all["cycle_num"] == cyc][feat].dropna().values
                if len(data) > 0:
                    med = float(np.median(data))
                    q1 = float(np.percentile(data, 25))
                    q3 = float(np.percentile(data, 75))
                    cycle_stats[int(cyc)] = {"median": med, "q1": q1, "q3": q3}

            bands[feat] = cycle_stats

        # ===== CALCULATE ROBUST Z-SCORES =====

        # Get initial capacity for normalization
        init_cap = float(df_bat.iloc[0]["capacity_mean"])

        # DCR z-scores
        dcrZSeries = []
        dcr_z_array = []
        dcr_cycles_array = []
        for _, row in df_bat.iterrows():
            cyc = int(row["cycle_num"])
            if cyc in bands.get("dcr", {}) and not pd.isna(row["dcr"]):
                band = bands.get("dcr", {})[cyc]
                z = compute_robust_z(row["dcr"], band["median"], band["q1"], band["q3"])
                dcrZSeries.append({"cycle": cyc, "z": float(z)})
                dcr_z_array.append(z)
                dcr_cycles_array.append(cyc)

        dcr_z_array = np.array(dcr_z_array)
        dcr_cycles_array = np.array(dcr_cycles_array)

        # Capacity% z-scores
        capZSeries = []
        cap_z_array = []
        cap_cycles_array = []
        for _, row in df_bat.iterrows():
            cyc = int(row["cycle_num"])
            if cyc in bands.get("capacity_mean", {}) and not pd.isna(row["capacity_mean"]):
                band = bands.get("capacity_mean", {})[cyc]

                # Convert to percentage
                cap_pct = (row["capacity_mean"] / init_cap) * 100
                med_pct = (band["median"] / init_cap) * 100
                q1_pct = (band["q1"] / init_cap) * 100
                q3_pct = (band["q3"] / init_cap) * 100

                z = compute_robust_z(cap_pct, med_pct, q1_pct, q3_pct)
                capZSeries.append({"cycle": cyc, "z": float(z)})
                cap_z_array.append(z)
                cap_cycles_array.append(cyc)

        cap_z_array = np.array(cap_z_array)
        cap_cycles_array = np.array(cap_cycles_array)

        # ===== MAJOR ANOMALY DETECTION =====

        max_dcr_z = float(np.nanmax(dcr_z_array)) if len(dcr_z_array) > 0 else float('-inf')
        min_cap_z = float(np.nanmin(cap_z_array)) if len(cap_z_array) > 0 else float('inf')

        major_alerts = []

        # DCR spike
        if max_dcr_z >= 4.0:
            dcr_onset = find_onset_cycle(dcr_z_array, dcr_cycles_array, 3.0, "pos", min_run=2)
            major_alerts.append({
                "type": "dcr_spike",
                "label": "Fault-like anomaly (DCR spike)",
                "severity": "HIGH" if abs(max_dcr_z) >= 6.0 else "MED",
                "onsetCycle": dcr_onset,
                "zValue": float(max_dcr_z)
            })

        # Capacity drop
        if min_cap_z <= -3.5:
            cap_onset = find_onset_cycle(cap_z_array, cap_cycles_array, -3.0, "neg", min_run=3)
            major_alerts.append({
                "type": "capacity_drop",
                "label": "Accelerated degradation (Capacity drop)",
                "severity": "HIGH" if abs(min_cap_z) >= 6.0 else "MED",
                "onsetCycle": cap_onset,
                "zValue": float(min_cap_z)
            })

        # ===== DRIVER FEATURES & EARLY WARNING =====

        ref_cycle = major_alerts[0]["onsetCycle"] if major_alerts and major_alerts[0].get("onsetCycle") else cycle
        ref_row = df_bat[df_bat["cycle_num"] == ref_cycle].iloc[0] if any(df_bat["cycle_num"] == ref_cycle) else df_bat.iloc[-1]

        drivers_list = []
        for feat in driver_features:
            if feat in ref_row.index and ref_cycle in bands.get(feat, {}):
                val = ref_row[feat]
                if not np.isnan(val):
                    band = bands[feat][ref_cycle]
                    z = compute_robust_z(val, band["median"], band["q1"], band["q3"])
                    drivers_list.append({
                        "feature": feat,
                        "label": {
                            "thermal_stress": "고온/열 스트레스",
                            "temperature_mean": "고온 노출",
                            "temp_rise_cycle": "셀 발열 증가",
                            "eff_c_rate": "고 C-rate(고부하)",
                            "current_max": "고부하(충전/회생)",
                            "current_min": "고부하(방전)",
                            "voltage_min": "깊은 방전(DoD↑)",
                            "dvdt_max_abs": "전압 급변",
                            "dTdt_max": "온도 급상승"
                        }.get(feat, feat),
                        "value": float(val),
                        "z": float(z),
                        "absZ": abs(float(z)),
                        "recommendation": {
                            "thermal_stress": "열관리 점검(팬/냉각)·고온 구간 제한",
                            "temperature_mean": "냉각/통풍·고온 운행 제한",
                            "temp_rise_cycle": "열 runaway 위험 체크·냉각 강화",
                            "eff_c_rate": "가속/급속충전 제한·부하 분산",
                            "current_max": "피크 전류 제한·회생제동 설정 조정",
                            "current_min": "피크 방전 전류 제한·부하 분산",
                            "voltage_min": "최저 SoC 제한·운영전략 조정",
                            "dvdt_max_abs": "BMS 로깅/센서 점검·전력 프로파일 확인",
                            "dTdt_max": "열관리/센서 점검·운행 제한"
                        }.get(feat, "")
                    })

        # Sort by absZ and get top 3
        drivers_list.sort(key=lambda x: x["absZ"], reverse=True)
        potential_drivers = drivers_list[:3]

        # Early warning detection
        significant_drivers = [d for d in drivers_list if d["absZ"] >= 3.0]
        has_early_warning = len(major_alerts) == 0 and len(significant_drivers) > 0

        early_warning = {
            "active": has_early_warning,
            "message": "핵심 KPI(DCR/Capacity%) 기준의 '큰 이탈'은 아직 없지만, 일부 스트레스 신호(driver)가 cohort 대비 outlier 입니다. (early warning)" if has_early_warning else None
        }

        # Markdown report
        markdown_lines = [
            f"# Battery anomaly report — {battery_id}",
            f"- current_cycle: {cycle}",
            f"- r_ratio: {r_ratio}",
            "",
            "## RUL Status",
            "- (RUL data not available in this endpoint)",
            "",
            "## Findings (robust z-score)"
        ]

        if len(major_alerts) == 0:
            markdown_lines.append("- No major deviation detected by KPI thresholds (DCR/Capacity%).")
        else:
            for alert in major_alerts:
                markdown_lines.append(f"- {alert['label']} | onset=cycle {alert.get('onsetCycle', 'unknown')} | robust z={alert['zValue']:.2f}")

        markdown_lines.extend([
            "",
            f"## Potential drivers around cycle {ref_cycle}",
        ])

        if len(drivers_list) == 0:
            markdown_lines.append("- (insufficient cohort stats)")
        else:
            for d in drivers_list[:5]:
                markdown_lines.append(f"- {d['label']} ({d['feature']}={d['value']:.4g}, z={d['z']:.2f}) — {d['recommendation']}")

        markdown_lines.append("")
        markdown_lines.append("> Note: This is a demo heuristic. In production, thresholds should be tuned with cohort definition, sensor QA, and physics constraints.")

        markdown = "\n".join(markdown_lines)

        # Return response with rows and bands
        return _sanitize_json({
            "battery_id": battery_id,
            "r_ratio": r_ratio,
            "cycle": cycle,
            "rows": df_bat.to_dict(orient="records"),
            "bands": {feat: {int(k): v for k, v in stats.items()} for feat, stats in bands.items()},
            "zSeries": {
                "dcr": dcrZSeries,
                "capacity": capZSeries
            },
            "majorAlerts": major_alerts,
            "earlyWarning": early_warning,
            "potentialDrivers": [
                {
                    "label": d["label"],
                    "feature": d["feature"],
                    "value": d["value"],
                    "z": d["z"],
                    "recommendation": d["recommendation"]
                }
                for d in potential_drivers
            ],
            "normalMessage": "현재 선택된 배터리는 reference cohort의 기대 범위 내에서 큰 이탈이 관측되지 않았습니다.",
            "markdown": markdown
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating anomaly report: {str(e)}")


# =================================================
# UNIFIED DEGRADATION MONITORING ENDPOINT
# =================================================
@app.get("/api/battery/{battery_id}/degradation-monitoring")
def battery_degradation_monitoring(
    battery_id: str,
    r_ratio: float = Query(default=0.1, ge=0, le=1),
    cycle: int = Query(default=None),
):
    """
    Fixed4-compatible degradation monitoring payload.

    Important details copied from fixed4:
    - Capacity anomaly uses battery-normalized Capacity(% of initial) cohort band.
    - Major issue decision uses prefix/current online view: rows with cycle <= current cycle.
    - Capacity issue threshold: min z <= -3.5.
    - Capacity onset: z <= -3.0 for 3 consecutive cycles; onset may be None.
    - DCR issue threshold: max z >= 4.0.
    - DCR onset: z >= 3.0 for 2 consecutive cycles; onset may be None.
    """
    try:
        all_cols = [
            "battery", "cycle_num", "capacity_mean", "dcr", "soh", "impedance_sum",
            "thermal_stress", "temperature_mean", "temp_rise_cycle",
            "eff_c_rate", "current_max", "current_min", "voltage_min",
            "dvdt_max_abs", "dTdt_max", "dcr_growth", "impedance_growth"
        ]

        def _fetch_one_bid(bid: str) -> pd.DataFrame:
            """Try strict cols, then all cols. Always return cycle_num + required feature columns."""
            attempts = []
            for cols in (all_cols, None):
                try:
                    df = fetch_cycles(bid, None, None, 1, cols)
                    attempts.append((cols, len(df) if df is not None else 0))
                    if df is not None and not df.empty:
                        df = df.copy()
                        if "battery" not in df.columns:
                            df["battery"] = bid
                        for c in all_cols:
                            if c not in df.columns and c not in {"battery"}:
                                df[c] = np.nan
                        return df
                except Exception:
                    continue
            return pd.DataFrame(columns=all_cols)

        df_bat_full = _fetch_one_bid(battery_id)
        if df_bat_full.empty:
            raise HTTPException(status_code=404, detail=f"No data for battery {battery_id}")

        df_bat_full = df_bat_full.sort_values("cycle_num").reset_index(drop=True)

        if cycle is None:
            cycle = int(df_bat_full["cycle_num"].max())
        else:
            cycle = int(cycle)
            available = sorted(int(x) for x in df_bat_full["cycle_num"].dropna().unique())
            if available:
                if cycle not in available:
                    # Clamp to closest available cycle instead of failing.
                    cycle = min(available, key=lambda x: abs(x - cycle))

        # Fixed4 online mode semantics: anomaly decision for current view uses prefix <= cycle.
        df_bat_prefix = df_bat_full[df_bat_full["cycle_num"].astype(int) <= int(cycle)].copy()
        if df_bat_prefix.empty:
            df_bat_prefix = df_bat_full.copy()

        cohort_batteries = ["B0018", "B0043", "B0042", "B0033", "B0055"]
        dfs_all = []
        for bid in cohort_batteries:
            df = _fetch_one_bid(bid)
            if df is not None and not df.empty:
                df = df.copy()
                df["battery_id"] = str(bid)
                dfs_all.append(df)

        if not dfs_all:
            raise HTTPException(status_code=404, detail="No reference cohort data available")

        df_all = pd.concat(dfs_all, ignore_index=True)
        if "battery_id" not in df_bat_full.columns:
            df_bat_full["battery_id"] = str(battery_id)
        if "battery_id" not in df_bat_prefix.columns:
            df_bat_prefix["battery_id"] = str(battery_id)

        driver_features = [
            "thermal_stress", "temperature_mean", "temp_rise_cycle",
            "eff_c_rate", "current_max", "current_min", "voltage_min",
            "dvdt_max_abs", "dTdt_max"
        ]
        raw_band_features = ["dcr", "capacity_mean", "impedance_sum"] + driver_features

        # Raw expected bands for DCR/impedance/drivers.
        bands = {}
        for feat in raw_band_features:
            if feat not in df_all.columns:
                continue
            cycle_stats = {}
            for cyc in sorted(df_all["cycle_num"].dropna().astype(int).unique()):
                vals = df_all[df_all["cycle_num"].astype(int) == int(cyc)][feat].astype(float).dropna().values
                if len(vals) > 0:
                    cycle_stats[int(cyc)] = {
                        "median": float(np.median(vals)),
                        "q1": float(np.percentile(vals, 25)),
                        "q3": float(np.percentile(vals, 75)),
                    }
            bands[feat] = cycle_stats

        # Fixed4 compute_capacity_pct_band(): normalize each battery by its own first capacity.
        cap_pct_band = {}
        if "capacity_mean" in df_all.columns:
            tmp = df_all[["battery_id", "cycle_num", "capacity_mean"]].copy()
            tmp = tmp.dropna(subset=["battery_id", "cycle_num", "capacity_mean"])
            if not tmp.empty:
                first_cap = (
                    tmp.sort_values(["battery_id", "cycle_num"])
                    .groupby("battery_id")["capacity_mean"]
                    .first()
                    .rename("cap0")
                )
                tmp = tmp.join(first_cap, on="battery_id")
                tmp = tmp[tmp["cap0"].astype(float) > 0]
                tmp["cap_pct"] = tmp["capacity_mean"].astype(float) / tmp["cap0"].astype(float) * 100.0
                for cyc in sorted(tmp["cycle_num"].dropna().astype(int).unique()):
                    vals = tmp[tmp["cycle_num"].astype(int) == int(cyc)]["cap_pct"].astype(float).dropna().values
                    if len(vals) > 0:
                        cap_pct_band[int(cyc)] = {
                            "median": float(np.median(vals)),
                            "q1": float(np.percentile(vals, 25)),
                            "q3": float(np.percentile(vals, 75)),
                            "unit": "pct",
                        }
        bands["capacity_pct"] = cap_pct_band

        # DCR z over full series.
        dcrZSeries = []
        for _, row in df_bat_full.iterrows():
            cyc = int(row["cycle_num"])
            try:
                val = float(row["dcr"])
            except Exception:
                continue
            if cyc in bands.get("dcr", {}) and np.isfinite(val):
                b = bands["dcr"][cyc]
                z = compute_robust_z(val, b["median"], b["q1"], b["q3"])
                dcrZSeries.append({"cycle": cyc, "z": float(z)})

        # Capacity z over full series using Fixed4 cap_pct band.
        capZSeries = []
        try:
            init_cap = float(df_bat_full.iloc[0]["capacity_mean"])
        except Exception:
            init_cap = float("nan")
        if np.isfinite(init_cap) and init_cap > 0:
            for _, row in df_bat_full.iterrows():
                cyc = int(row["cycle_num"])
                try:
                    cap_pct = float(row["capacity_mean"]) / init_cap * 100.0
                except Exception:
                    continue
                if cyc in cap_pct_band and np.isfinite(cap_pct):
                    b = cap_pct_band[cyc]
                    z = compute_robust_z(cap_pct, b["median"], b["q1"], b["q3"])
                    capZSeries.append({"cycle": cyc, "z": float(z)})

        def _prefix_points(points, upto_cycle: int):
            return [p for p in points if p.get("cycle") is not None and int(p["cycle"]) <= int(upto_cycle) and np.isfinite(float(p.get("z", np.nan)))]

        dcr_prefix = _prefix_points(dcrZSeries, int(cycle))
        cap_prefix = _prefix_points(capZSeries, int(cycle))

        dcr_vals = np.array([float(p["z"]) for p in dcr_prefix], dtype=float)
        dcr_cycles = np.array([int(p["cycle"]) for p in dcr_prefix], dtype=int)
        cap_vals = np.array([float(p["z"]) for p in cap_prefix], dtype=float)
        cap_cycles = np.array([int(p["cycle"]) for p in cap_prefix], dtype=int)

        max_dcr_z = float(np.nanmax(dcr_vals)) if dcr_vals.size else float("nan")
        min_cap_z = float(np.nanmin(cap_vals)) if cap_vals.size else float("nan")

        major_alerts = []
        dcr_onset = None
        cap_onset = None

        if np.isfinite(max_dcr_z) and max_dcr_z >= 4.0:
            dcr_onset = find_onset_cycle(dcr_vals, dcr_cycles, 3.0, "pos", min_run=2)
            major_alerts.append({
                "type": "dcr_spike",
                "label": "Fault-like anomaly (DCR spike)",
                "severity": "HIGH" if abs(max_dcr_z) >= 6.0 else "MED",
                "onsetCycle": dcr_onset,
                "zValue": float(max_dcr_z),
            })

        if np.isfinite(min_cap_z) and min_cap_z <= -3.5:
            cap_onset = find_onset_cycle(cap_vals, cap_cycles, -3.0, "neg", min_run=3)
            major_alerts.append({
                "type": "capacity_drop",
                "label": "Accelerated degradation (Capacity drop)",
                "severity": "HIGH" if abs(min_cap_z) >= 6.0 else "MED",
                "onsetCycle": cap_onset,
                "zValue": float(min_cap_z),
            })

        # Fixed4 ref cycle: DCR onset first, then capacity onset, then current cycle.
        ref_cycle = int(cycle)
        if dcr_onset is not None:
            ref_cycle = int(dcr_onset)
        elif cap_onset is not None:
            ref_cycle = int(cap_onset)

        matching_rows = df_bat_full[df_bat_full["cycle_num"].astype(int) == int(ref_cycle)]
        ref_row = matching_rows.iloc[0] if len(matching_rows) else df_bat_prefix.iloc[-1]

        driver_labels = {
            "thermal_stress": "고온/열 스트레스",
            "temperature_mean": "고온 노출",
            "temp_rise_cycle": "셀 발열 증가",
            "eff_c_rate": "고 C-rate(고부하)",
            "current_max": "고부하(충전/회생)",
            "current_min": "고부하(방전)",
            "voltage_min": "깊은 방전(DoD↑)",
            "dvdt_max_abs": "전압 급변",
            "dTdt_max": "온도 급상승",
        }
        driver_actions = {
            "thermal_stress": "열관리 점검(팬/냉각)·고온 구간 제한",
            "temperature_mean": "냉각/통풍·고온 운행 제한",
            "temp_rise_cycle": "열 runaway 위험 체크·냉각 강화",
            "eff_c_rate": "가속/급속충전 제한·부하 분산",
            "current_max": "피크 전류 제한·회생제동 설정 조정",
            "current_min": "피크 방전 전류 제한·부하 분산",
            "voltage_min": "최저 SoC 제한·운영전략 조정",
            "dvdt_max_abs": "BMS 로깅/센서 점검·전력 프로파일 확인",
            "dTdt_max": "열관리/센서 점검·운행 제한",
        }

        drivers_list = []
        for feat in driver_features:
            if feat not in ref_row.index or int(ref_cycle) not in bands.get(feat, {}):
                continue
            try:
                val = float(ref_row[feat])
            except Exception:
                continue
            if not np.isfinite(val):
                continue
            b = bands[feat][int(ref_cycle)]
            z = compute_robust_z(val, b["median"], b["q1"], b["q3"])
            drivers_list.append({
                "feature": feat,
                "label": driver_labels.get(feat, feat),
                "value": float(val),
                "z": float(z),
                "absZ": abs(float(z)),
                "recommendation": driver_actions.get(feat, ""),
            })
        drivers_list.sort(key=lambda x: x["absZ"], reverse=True)
        potential_drivers = drivers_list[:3]

        significant_drivers = [d for d in drivers_list if np.isfinite(d.get("z", np.nan)) and abs(float(d["z"])) >= 3.0]
        has_early_warning = len(major_alerts) == 0 and len(significant_drivers) > 0
        status = "major-anomaly" if major_alerts else ("early-warning" if has_early_warning else "normal")

        markdown_lines = [
            f"# Battery anomaly report — {battery_id}",
            f"- current_cycle: {cycle}",
            f"- r_ratio: {r_ratio}",
            "",
            "## Findings (robust z-score)",
        ]
        if len(major_alerts) == 0:
            markdown_lines.append("- No major deviation detected by KPI thresholds (DCR/Capacity%).")
        else:
            for alert in major_alerts:
                onset_txt = f"cycle {alert['onsetCycle']}" if alert.get("onsetCycle") is not None else "(onset 미확정)"
                markdown_lines.append(f"- {alert['label']} | severity={alert['severity']} | onset={onset_txt} | robust z={alert['zValue']:.2f}")

        markdown_lines += ["", f"## Potential drivers around cycle {ref_cycle}"]
        if len(drivers_list) == 0:
            markdown_lines.append("- (insufficient cohort stats)")
        else:
            for d in drivers_list[:5]:
                markdown_lines.append(f"- {d['label']} ({d['feature']}={d['value']:.4g}, z={d['z']:.2f}) — {d.get('recommendation', '')}")
        markdown_lines.append("")
        markdown_lines.append("> Note: demo heuristic based on robust z-score; tune thresholds/cohorts for production.")
        markdown = "\n".join(markdown_lines)

        return _sanitize_json({
            "battery": battery_id,
            "r_ratio": r_ratio,
            "cycle": int(cycle),
            "cycles": [int(x) for x in df_bat_full["cycle_num"].tolist()],
            "series": df_bat_full.to_dict(orient="records"),
            "bands": {feat: {int(k): v for k, v in stats.items()} for feat, stats in bands.items()},
            "z_scores": {
                "dcr": dcrZSeries,
                "capacity": capZSeries,
            },
            "dcr_max_z": float(max_dcr_z) if np.isfinite(max_dcr_z) else None,
            "cap_min_z": float(min_cap_z) if np.isfinite(min_cap_z) else None,
            "dcr_onset_cycle": dcr_onset,
            "cap_onset_cycle": cap_onset,
            "issues": major_alerts,
            "status": status,
            "earlyWarning": {
                "active": has_early_warning,
                "message": "핵심 KPI(DCR/Capacity%) 기준의 '큰 이탈'은 아직 없지만, 일부 스트레스 신호(driver)가 cohort 대비 outlier 입니다. (early warning)" if has_early_warning else None,
            },
            "drivers": potential_drivers,
            "reportMarkdown": markdown,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in degradation monitoring: {str(e)}")



@app.get("/api/fixed4/shap-v23")
@app.get("/api/fixed4/shap-v22")
@app.get("/api/fixed4/shap-v21")
@app.get("/api/fixed4/shap-v20")
def fixed4_shap_global_importance():
    import json
    import math
    from pathlib import Path

    p = Path("/Users/velocitygoal/battery_project/v11/deep_learning/core/shap_outputs/bmaml_shap_seq_feature_importance.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"SHAP global importance JSON not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))

    def is_num(v):
        try:
            return math.isfinite(float(v))
        except Exception:
            return False

    def list_items(xs):
        out = []
        for x in xs or []:
            if not isinstance(x, dict):
                continue
            name = x.get("feature") or x.get("name") or x.get("feature_name") or x.get("column") or x.get("col")
            val = x.get("importance", x.get("value", x.get("mean_abs_shap", x.get("abs_mean", x.get("shap", None)))))
            if name is not None and is_num(val):
                out.append({"feature": str(name), "importance": float(val)})
        return out

    def parse(o):
        if isinstance(o, list):
            got = list_items(o)
            if got:
                return got
        if not isinstance(o, dict):
            return []
        if o and all(is_num(v) for v in o.values()):
            return [{"feature": str(k), "importance": float(v)} for k, v in o.items()]
        for key in ["items", "data", "global_importance", "feature_importance", "shap_importance"]:
            got = list_items(o.get(key))
            if got:
                return got
        for nk in ["feature_names", "features", "names", "columns"]:
            names = o.get(nk)
            if not isinstance(names, list):
                continue
            for vk in ["importance", "importances", "values", "mean_abs_shap", "shap_values", "global_importance"]:
                vals = o.get(vk)
                if isinstance(vals, list) and len(vals) == len(names):
                    out = []
                    for a, b in zip(names, vals):
                        if is_num(b):
                            out.append({"feature": str(a), "importance": float(b)})
                    if out:
                        return out
        for v in o.values():
            got = parse(v)
            if got:
                return got
        return []

    items = parse(obj)
    items = [x for x in items if x.get("feature") and is_num(x.get("importance"))]
    items.sort(key=lambda x: float(x["importance"]), reverse=True)
    return {
        "ok": True,
        "source": str(p),
        "kind": "global_model_feature_importance",
        "note": "Global BMAML/sequence-model feature importance; not a cycle-local anomaly driver.",
        "items": items,
    }


# --- HOTFIX_V29_BOTTOM_REINFER_SHAP_OVERRIDE ---
@app.middleware("http")
async def _v29_bottom_reinfer_shap_override(request, call_next):
    import json as _json
    import math as _math
    import os as _os
    import re as _re
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path
    try:
        from fastapi.responses import JSONResponse as _JSONResponse
    except Exception:
        _JSONResponse = JSONResponse

    path = request.url.path
    method = request.method.upper()

    CKPT = _CKPT_PATH
    RUNNER = _SCRIPTS_PATH
    LIVE_DIR = _resolve_path(
        "/app/data/live_reinfer_results",
        "/Users/velocitygoal/battery-rul-v11/battery_rul_fastapi_react_mvp/data/live_reinfer_results"
    )
    PYROOTS = [str(_APP_ROOT), str(_APP_ROOT / "backend"), str(_APP_ROOT / "deep_learning")]
    SHAP_CANDIDATES = _SHAP_CANDIDATES

    def resp(code, payload):
        return _JSONResponse(status_code=code, content=payload)

    def fnum(v, default=None):
        try:
            x = float(v)
            return x if _math.isfinite(x) else default
        except Exception:
            return default

    def battery_from_request():
        patterns = [
            r"/api/battery/([^/]+)/reinfer",
            r"/api/battery/([^/]+)/re-inference",
            r"/api/live-reinfer-v\d+/([^/]+)",
            r"/api/live-reinfer/([^/]+)",
        ]
        for pat in patterns:
            m = _re.fullmatch(pat, path)
            if m:
                return m.group(1)
        if path in {"/api/reinfer", "/api/re-inference", "/api/live-reinfer", "/api/live-reinfer-v29"}:
            return request.query_params.get("battery") or request.query_params.get("battery_id") or request.query_params.get("bid")
        return None

    bid = battery_from_request()
    if bid and method in {"GET", "POST"}:
        if not CKPT.exists():
            return resp(500, {"ok": False, "error": "checkpoint not found", "path": str(CKPT)})
        if not RUNNER.exists():
            return resp(500, {"ok": False, "error": "runner not found", "path": str(RUNNER)})

        r = round(float(fnum(request.query_params.get("r_ratio", request.query_params.get("rRatio", "0.1")), 0.1)), 2)
        tag = "r" + f"{r:.2f}".replace(".", "p")
        run_id = _time.strftime("%Y%m%d_%H%M%S")
        out_dir = LIVE_DIR / f"{bid}_{tag}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        expected_json = out_dir / f"{bid}_viz_meta_{tag}.json"
        timeout = int(fnum(request.query_params.get("timeout", "900"), 900))

        env = dict(_os.environ)
        env["PYTHONPATH"] = ":".join(PYROOTS + [env.get("PYTHONPATH", "")])
        env["CHECKPOINT_PATH"] = str(CKPT)
        env["BMAML_CHECKPOINT"] = str(CKPT)
        env["PRECOMP_DIR"] = str(out_dir)

        args = [
            str(RUNNER),
            "--ckpt", str(CKPT),
            "--eval_dataset", "from_ckpt",
            "--r_ratio", str(r),
            "--bids", str(bid),
            "--out_dir", str(out_dir),
            "--save_json", "1",
            "--save_batch_json", "1",
            "--min_support", "0",
            "--cap_before_eol", "1",
            "--ratio_base", "pos",
        ]

        commands = []
        conda = _shutil.which("conda")
        if conda:
            commands.append(("conda run -n battery-maml", [conda, "run", "-n", "battery-maml", "python"] + args))
        for py in [
            "/opt/anaconda3/envs/battery-maml/bin/python",
            "/Users/velocitygoal/anaconda3/envs/battery-maml/bin/python",
            "/Users/velocitygoal/miniforge3/envs/battery-maml/bin/python",
            "/Users/velocitygoal/mambaforge/envs/battery-maml/bin/python",
        ]:
            if _Path(py).exists():
                commands.append((py, [py] + args))
        if _sys.executable:
            commands.append((_sys.executable, [_sys.executable] + args))
        py = _shutil.which("python")
        if py:
            commands.append((py, [py] + args))

        deduped = []
        seen = set()
        for label, cmd in commands:
            key = tuple(cmd)
            if key not in seen:
                deduped.append((label, cmd))
                seen.add(key)

        attempts = []
        started = _time.time()
        for label, cmd in deduped:
            try:
                proc = _subprocess.run(cmd, cwd=PYROOTS[0], env=env, text=True, capture_output=True, timeout=timeout)
                attempts.append({"executor": label, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2400:], "stderr_tail": proc.stderr[-2400:]})
                if proc.returncode == 0 and expected_json.exists():
                    item = _json.loads(expected_json.read_text(encoding="utf-8"))
                    return resp(200, {
                        "ok": True,
                        "success": True,
                        "mode": "live_bmaml_reinfer_session_only",
                        "battery": bid,
                        "r_ratio": r,
                        "tag": tag,
                        "executor": label,
                        "checkpoint": str(CKPT),
                        "runner": str(RUNNER),
                        "out_dir": str(out_dir),
                        "json_path": str(expected_json),
                        "baseline_overwritten": False,
                        "elapsed_sec": round(_time.time() - started, 2),
                        "item": item,
                        "stdout_tail": proc.stdout[-2400:],
                        "stderr_tail": proc.stderr[-2400:],
                    })
            except Exception as e:
                attempts.append({"executor": label, "exception": str(e)})

        return resp(500, {
            "ok": False,
            "success": False,
            "error": "all runner attempts failed or expected JSON was not produced",
            "battery": bid,
            "r_ratio": r,
            "expected_json": str(expected_json),
            "baseline_overwritten": False,
            "attempts": attempts,
        })

    if path in {"/api/fixed4/shap-current", "/api/fixed4/shap-v29", "/api/fixed4/shap-v28", "/api/fixed4/shap-v27", "/api/fixed4/shap-v26"}:
        p = next((_Path(x) for x in SHAP_CANDIDATES if _Path(x).exists()), None)
        if p is None:
            return resp(404, {"ok": False, "error": "SHAP global importance JSON not found", "searched": SHAP_CANDIDATES})
        try:
            obj = _json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            return resp(500, {"ok": False, "error": f"failed to read SHAP JSON: {e}", "path": str(p)})

        def is_num(v):
            try:
                return _math.isfinite(float(v))
            except Exception:
                return False

        def list_items(xs):
            out = []
            for x in xs or []:
                if not isinstance(x, dict):
                    continue
                name = x.get("feature") or x.get("name") or x.get("feature_name") or x.get("column") or x.get("col")
                val = x.get("importance", x.get("value", x.get("mean_abs_shap", x.get("abs_mean", x.get("shap", None)))))
                if name is not None and is_num(val):
                    out.append({"feature": str(name), "importance": float(val)})
            return out

        def parse(o):
            if isinstance(o, list):
                got = list_items(o)
                if got:
                    return got
            if not isinstance(o, dict):
                return []
            if o and all(is_num(v) for v in o.values()):
                return [{"feature": str(k), "importance": float(v)} for k, v in o.items()]
            for key in ["items", "data", "global_importance", "feature_importance", "shap_importance"]:
                got = list_items(o.get(key))
                if got:
                    return got
            for nk in ["feature_names", "features", "names", "columns"]:
                names = o.get(nk)
                if not isinstance(names, list):
                    continue
                for vk in ["importance", "importances", "values", "mean_abs_shap", "shap_values", "global_importance"]:
                    vals = o.get(vk)
                    if isinstance(vals, list) and len(vals) == len(names):
                        out = []
                        for a, b in zip(names, vals):
                            if is_num(b):
                                out.append({"feature": str(a), "importance": float(b)})
                        if out:
                            return out
            for v in o.values():
                got = parse(v)
                if got:
                    return got
            return []

        items = parse(obj)
        items = [x for x in items if x.get("feature") and is_num(x.get("importance"))]
        items.sort(key=lambda x: float(x["importance"]), reverse=True)
        return resp(200, {"ok": True, "source": str(p), "kind": "global_model_feature_importance", "items": items})

    return await call_next(request)
# --- END HOTFIX_V29_BOTTOM_REINFER_SHAP_OVERRIDE ---

# === HF FINAL LIVE REINFER OVERRIDE ===
# Purpose:
# - Intercept stale reinfer middlewares/routes.
# - Use existing HF-copied runners.
# - Return actual subprocess stdout/stderr instead of hiding the failure.

@app.middleware("http")
async def _hf_final_live_reinfer_override(request, call_next):
    import os as _os
    import sys as _sys
    import re as _re
    import json as _json
    import time as _time
    import glob as _glob
    import subprocess as _subprocess
    from pathlib import Path as _Path
    from fastapi.responses import JSONResponse as _JSONResponse

    path = request.url.path

    battery = None
    m = (
        _re.fullmatch(r"/api/battery/([^/]+)/reinfer", path)
        or _re.fullmatch(r"/api/live-reinfer-v\d+/([^/]+)", path)
        or _re.fullmatch(r"/api/live-reinfer/([^/]+)", path)
    )

    if m:
        battery = m.group(1)
    elif path in {"/api/reinfer", "/api/re-inference", "/api/live-reinfer"}:
        try:
            body = await request.body()
            payload = _json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}
        battery = (
            payload.get("battery")
            or payload.get("battery_id")
            or payload.get("id")
            or request.query_params.get("battery")
            or request.query_params.get("battery_id")
        )
    else:
        return await call_next(request)

    if not battery:
        return _JSONResponse({
            "ok": False,
            "mode": "hf_final_live_reinfer",
            "error": "battery id missing",
            "path": path,
        }, status_code=400)

    rr = (
        request.query_params.get("r_ratio")
        or request.query_params.get("r")
        or request.query_params.get("ratio")
        or "0.1"
    )

    timeout = int(request.query_params.get("timeout") or 900)

    repo = _Path("/app") if _Path("/app").exists() else _Path.cwd()
    backend = repo / "backend"
    deep_learning = repo / "deep_learning"

    ckpt_candidates = [
        repo / "core_checkpoints" / "nasa_bmaml_best_re.pt",
        backend / "core_checkpoints" / "nasa_bmaml_best_re.pt",
        _Path("/Users/velocitygoal/battery-rul-dashboard/core_checkpoints/nasa_bmaml_best_re.pt"),
    ]
    ckpt = next((x for x in ckpt_candidates if x.exists()), ckpt_candidates[0])

    out_dirs = [
        repo / "data" / "live_reinfer_results",
        _Path("/tmp") / "battery_rul_live_reinfer_results",
    ]
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    script_candidates = [
        repo / "run_bmaml_reinfer.py",
        repo / "export_rul_dashboard_data_meta_fixed.py",
        repo / "scripts" / "run_bmaml_reinfer.py",
        repo / "scripts" / "export_rul_dashboard_data_meta_fixed.py",
        repo / "scripts" / "prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py",
        _Path("/Users/velocitygoal/battery-rul-dashboard/run_bmaml_reinfer.py"),
        _Path("/Users/velocitygoal/battery-rul-dashboard/export_rul_dashboard_data_meta_fixed.py"),
        _Path("/Users/velocitygoal/battery-rul-dashboard/scripts/prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py"),
    ]

    scripts = [x for x in script_candidates if x.exists()]
    if not scripts:
        return _JSONResponse({
            "ok": False,
            "mode": "hf_final_live_reinfer",
            "error": "No runner script found",
            "searched": [str(x) for x in script_candidates],
            "checkpoint_exists": ckpt.exists(),
            "checkpoint": str(ckpt),
        }, status_code=500)

    env = dict(_os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = ":".join([
        str(repo),
        str(backend),
        str(deep_learning),
        env.get("PYTHONPATH", ""),
    ])

    def _latest_json(paths):
        files = []
        for d in paths:
            try:
                files.extend(_glob.glob(str(d / "**" / "*.json"), recursive=True))
            except Exception:
                pass
        files = [f for f in files if _Path(f).is_file()]
        if not files:
            return None, None
        latest = max(files, key=lambda f: _Path(f).stat().st_mtime)
        try:
            return latest, _json.loads(_Path(latest).read_text())
        except Exception:
            return latest, None

    attempts = []
    started = _time.time()

    for script in scripts:
        for out_dir in out_dirs:
            cmd_variants = [
                [
                    _sys.executable, str(script),
                    "--battery", str(battery),
                    "--r_ratio", str(rr),
                    "--ckpt", str(ckpt),
                    "--out_dir", str(out_dir),
                ],
                [
                    _sys.executable, str(script),
                    "--battery_id", str(battery),
                    "--r_ratio", str(rr),
                    "--ckpt", str(ckpt),
                    "--out_dir", str(out_dir),
                ],
                [
                    _sys.executable, str(script),
                    "--battery", str(battery),
                    "--r-ratio", str(rr),
                    "--checkpoint", str(ckpt),
                    "--output-dir", str(out_dir),
                ],
                [
                    _sys.executable, str(script),
                    "--ckpt", str(ckpt),
                    "--eval_dataset", "from_ckpt",
                    "--r_ratio", str(rr),
                    "--out_dir", str(out_dir),
                ],
            ]

            for cmd in cmd_variants:
                try:
                    proc = _subprocess.run(
                        cmd,
                        cwd=str(repo),
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                    )

                    attempt = {
                        "script": str(script),
                        "cmd": cmd,
                        "cwd": str(repo),
                        "out_dir": str(out_dir),
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-4000:],
                        "stderr_tail": proc.stderr[-4000:],
                    }
                    attempts.append(attempt)

                    if proc.returncode == 0:
                        latest_path, latest_payload = _latest_json(out_dirs)

                        base = {
                            "ok": True,
                            "mode": "hf_final_live_reinfer",
                            "battery": battery,
                            "battery_id": battery,
                            "r_ratio": rr,
                            "r_ratio_input": rr,
                            "script": str(script),
                            "cmd": cmd,
                            "cwd": str(repo),
                            "checkpoint": str(ckpt),
                            "checkpoint_exists": ckpt.exists(),
                            "elapsed_sec": round(_time.time() - started, 2),
                            "latest_json": latest_path,
                            "stdout_tail": proc.stdout[-4000:],
                            "stderr_tail": proc.stderr[-4000:],
                        }

                        if isinstance(latest_payload, dict):
                            merged = dict(latest_payload)
                            merged.update(base)
                            merged["data"] = latest_payload
                            return _JSONResponse(merged, status_code=200)

                        return _JSONResponse(base, status_code=200)

                except Exception as e:
                    attempts.append({
                        "script": str(script),
                        "cmd": cmd,
                        "cwd": str(repo),
                        "out_dir": str(out_dir),
                        "exception": repr(e),
                    })

    return _JSONResponse({
        "ok": False,
        "mode": "hf_final_live_reinfer",
        "error": "runner found but all invocations failed",
        "battery": battery,
        "battery_id": battery,
        "r_ratio": rr,
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "scripts": [str(x) for x in scripts],
        "elapsed_sec": round(_time.time() - started, 2),
        "attempts": attempts[-12:],
    }, status_code=500)

