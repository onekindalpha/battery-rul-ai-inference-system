import os
import re
import sys
import json
import math
import time
import subprocess
from pathlib import Path
from urllib.parse import parse_qs

from hf_app import app as downstream_app


class HFFrontApp:
    def __init__(self, downstream):
        self.downstream = downstream

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.downstream(scope, receive, send)

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "")
        qs = parse_qs(scope.get("query_string", b"").decode("utf-8", errors="ignore"))

        root = Path("/app") if Path("/app").exists() else Path.cwd()

        precomp_dir = root / "data" / "precomputed_from_export_v2"
        ckpt = root / "core_checkpoints" / "nasa_bmaml_best_re.pt"
        shap_json = root / "deep_learning" / "core" / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"
        shap_png = root / "deep_learning" / "core" / "shap_outputs" / "bmaml_shap_seq_feature_importance.png"

        async def send_bytes(status, body, content_type):
            headers = [
                (b"content-type", content_type.encode("utf-8")),
                (b"access-control-allow-origin", b"*"),
                (b"cache-control", b"no-store"),
            ]
            await send({"type": "http.response.start", "status": status, "headers": headers})

            if method == "HEAD":
                body = b""

            await send({"type": "http.response.body", "body": body})

        async def send_json(status, obj):
            def clean(x):
                if isinstance(x, float):
                    if math.isnan(x) or math.isinf(x):
                        return None
                    return x
                if isinstance(x, list):
                    return [clean(v) for v in x]
                if isinstance(x, dict):
                    return {str(k): clean(v) for k, v in x.items()}
                return x

            body = json.dumps(clean(obj), ensure_ascii=False).encode("utf-8")
            await send_bytes(status, body, "application/json; charset=utf-8")

        def q1(name, default=None):
            vals = qs.get(name)
            return vals[0] if vals else default

        def ratio_to_tag(ratio):
            ratio = str(ratio or "0.1")
            if ratio.startswith("r"):
                return ratio
            try:
                return f"r{float(ratio):.2f}".replace(".", "p")
            except Exception:
                return "r0p10"

        def tag_to_ratio(tag):
            tag = str(tag)
            return tag[1:].replace("p", ".") if tag.startswith("r") else tag

        def read_json(fp):
            try:
                return json.loads(Path(fp).read_text())
            except Exception:
                return None

        def as_float(v):
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        def first_not_none(*vals):
            for v in vals:
                if v is not None:
                    return v
            return None

        def unwrap(raw):
            if not isinstance(raw, dict):
                return {}
            for key in ["data", "payload", "result", "pack"]:
                v = raw.get(key)
                if isinstance(v, dict):
                    return v
            return raw

        def get_pred_std(raw):
            raw = unwrap(raw)
            pred = raw.get("pred")

            std_seq = []
            mean_seq = []

            if isinstance(pred, dict):
                std_seq = (
                    pred.get("std")
                    or pred.get("sigma")
                    or pred.get("uncertainty")
                    or pred.get("y_std")
                    or []
                )
                mean_seq = (
                    pred.get("mean")
                    or pred.get("mu")
                    or pred.get("value")
                    or pred.get("rul")
                    or pred.get("y")
                    or []
                )

            elif isinstance(pred, list):
                for row in pred:
                    if isinstance(row, dict):
                        std_seq.append(first_not_none(
                            row.get("std"),
                            row.get("sigma"),
                            row.get("uncertainty"),
                            row.get("y_std"),
                        ))
                        mean_seq.append(first_not_none(
                            row.get("mean"),
                            row.get("mu"),
                            row.get("pred"),
                            row.get("rul"),
                            row.get("y"),
                            row.get("value"),
                        ))
                    else:
                        mean_seq.append(row)

            if not isinstance(std_seq, list):
                std_seq = []
            if not isinstance(mean_seq, list):
                mean_seq = []

            idx = first_not_none(
                raw.get("currentCycleIndex"),
                raw.get("current_cycle_index"),
                raw.get("current_idx"),
                raw.get("q_pos"),
            )

            try:
                idx = int(idx)
            except Exception:
                idx = len(std_seq) - 1 if std_seq else len(mean_seq) - 1

            if idx < 0:
                idx = 0

            if std_seq:
                idx = min(idx, len(std_seq) - 1)
            elif mean_seq:
                idx = min(idx, len(mean_seq) - 1)

            std_val = as_float(std_seq[idx]) if std_seq and idx < len(std_seq) else None
            pred_val = as_float(mean_seq[idx]) if mean_seq and idx < len(mean_seq) else None

            return raw, std_seq, mean_seq, idx, std_val, pred_val

        def normalize_precomputed(battery, tag):
            battery = battery.upper()
            fp = precomp_dir / f"{battery}_viz_meta_{tag}.json"
            raw0 = read_json(fp)

            if not isinstance(raw0, dict):
                return None, {
                    "ok": False,
                    "error": "precomputed file not found or unreadable",
                    "file": str(fp),
                    "exists": fp.exists(),
                    "battery": battery,
                    "tag": tag,
                }

            raw, std_seq, mean_seq, idx, std_val, pred_val = get_pred_std(raw0)

            uncertainty_2sigma = 2.0 * std_val if std_val is not None else None

            confidence = None
            confidence_percent = None
            if uncertainty_2sigma is not None:
                # Smaller uncertainty => higher confidence.
                confidence = 1.0 / (1.0 + max(float(uncertainty_2sigma), 0.0))
                confidence_percent = round(confidence * 100.0, 2)

            out = dict(raw)
            metrics = out.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}

            metrics.update({
                "uncertainty": uncertainty_2sigma,
                "prediction_uncertainty": uncertainty_2sigma,
                "uncertainty_2sigma": uncertainty_2sigma,
                "confidence": confidence,
                "confidence_percent": confidence_percent,
            })

            out.update({
                "ok": True,
                "source": "hf_front_app_exact_precomputed_from_export_v2",
                "battery": battery,
                "battery_id": battery,
                "tag": tag,
                "r_ratio": tag_to_ratio(tag),
                "r_ratio_input": tag_to_ratio(tag),
                "source_file": str(fp),

                "pred_std": std_seq,
                "currentCycleIndex": idx,
                "current_cycle_index": idx,

                "uncertainty": uncertainty_2sigma,
                "prediction_uncertainty": uncertainty_2sigma,
                "pred_uncertainty": uncertainty_2sigma,
                "uncertainty_2sigma": uncertainty_2sigma,
                "uncertainty_plus_minus": uncertainty_2sigma,
                "uncertainty_band": uncertainty_2sigma,

                "confidence": confidence,
                "confidence_percent": confidence_percent,
                "prediction_confidence": confidence,
                "prediction_confidence_percent": confidence_percent,

                "current_pred_rul": first_not_none(
                    raw.get("current_pred_rul"),
                    raw.get("pred_rul"),
                    pred_val,
                ),
                "pred_rul": first_not_none(
                    raw.get("pred_rul"),
                    raw.get("current_pred_rul"),
                    pred_val,
                ),

                "metrics": metrics,
            })

            inner = dict(out)
            out["data"] = inner
            out["payload"] = inner
            out["result"] = inner
            return out, None

        def shap_payload():
            raw = read_json(shap_json)
            items = []

            candidates = []
            if isinstance(raw, dict):
                candidates = [
                    raw.get("items"),
                    raw.get("data"),
                    raw.get("features"),
                    raw.get("global_importance"),
                    raw.get("feature_importance"),
                    raw.get("mean_abs_shap"),
                    raw.get("importance"),
                ]
            elif isinstance(raw, list):
                candidates = [raw]

            chosen = None
            for cand in candidates:
                if isinstance(cand, dict):
                    chosen = [{"feature": k, "importance": v} for k, v in cand.items()]
                    break
                if isinstance(cand, list) and cand:
                    chosen = cand
                    break

            if isinstance(chosen, list):
                for i, row in enumerate(chosen):
                    if isinstance(row, dict):
                        feature = first_not_none(
                            row.get("feature"),
                            row.get("name"),
                            row.get("column"),
                            row.get("feature_name"),
                            f"feature_{i}",
                        )
                        value = first_not_none(
                            row.get("importance"),
                            row.get("mean_abs_shap"),
                            row.get("mean_abs"),
                            row.get("value"),
                            row.get("shap"),
                        )
                    else:
                        feature = f"feature_{i}"
                        value = row

                    fv = as_float(value)
                    if feature is not None and fv is not None:
                        items.append({
                            "feature": str(feature),
                            "importance": fv,
                            "mean_abs_shap": fv,
                            "value": fv,
                        })

            items = sorted(items, key=lambda x: abs(float(x.get("importance") or 0)), reverse=True)

            # HF_SHAP_FEATURE_NAME_REMAP_V1
            # Existing SHAP JSON is found, but older parser may label rows as feature_0, feature_1, ...
            # Map those indices back to the actual sequence feature order used by the BMAML runner.
            _known_feature_cols = [
                "anchor_cycle", "capacity_mean", "ambient_temp_c", "voltage_measured_mean",
                "voltage_min", "voltage_max", "voltage_std", "re_ohm_interp", "rct_ohm_interp",
                "temperature_measured_max", "temperature_mean", "temperature_min", "temperature_std",
                "temp_rise_cycle", "eff_c_rate", "current_mean", "current_std", "current_min",
                "current_max", "dvdt_max_abs", "dTdt_max", "soh", "capacity_derivative", "cap_vel",
                "regen_strength", "impedance_sum", "impedance_growth", "dcr", "dcr_growth",
                "lli", "lam", "temp_rise", "thermal_stress", "Residual", "IMF1", "IMF2",
                "IMF3", "IMF4", "IMF5", "IMF6"
            ]

            _fixed_items = []
            for _row in items:
                _row = dict(_row)
                _feature = str(_row.get("feature") or "")
                _m = re.fullmatch(r"feature_(\d+)", _feature)
                if _m:
                    _idx = int(_m.group(1))
                    if 0 <= _idx < len(_known_feature_cols):
                        _row["feature"] = _known_feature_cols[_idx]
                _fixed_items.append(_row)
            items = _fixed_items

            return {
                "ok": bool(items),
                "source": "hf_front_app_exact_shap_outputs",
                "json_file": str(shap_json),
                "json_exists": shap_json.exists(),
                "image_file": str(shap_png),
                "image_exists": shap_png.exists(),
                "image_url": "/api/fixed4/shap-image",
                "png_url": "/api/fixed4/shap-image",
                "items": items,
                "data": items,
                "features": items,
                "global_importance": items,
            }

        # HF_REAL_LIVE_JSON_HELPERS_V1
        def find_new_live_json_files(started_ts, battery):
            candidates = []
            search_dirs = [
                precomp_dir,
                root / "data" / "live_reinfer_results",
                root / "data" / "reinfer",
                root / "data" / "live",
                root / "outputs",
                Path("/tmp") / "battery_rul_live_reinfer_results",
            ]

            for d in search_dirs:
                try:
                    d.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                try:
                    for fp in d.rglob("*.json"):
                        if not fp.is_file():
                            continue
                        name = fp.name.upper()
                        if battery.upper() not in name and "B00" not in name:
                            continue
                        try:
                            if fp.stat().st_mtime >= started_ts - 1:
                                candidates.append(fp)
                        except Exception:
                            pass
                except Exception:
                    pass

            candidates = sorted(
                candidates,
                key=lambda x: x.stat().st_mtime if x.exists() else 0,
                reverse=True,
            )
            return candidates

        def make_live_item_response(live_payload, battery, tag, source_file, meta=None):
            meta = meta or {}

            if isinstance(live_payload, dict):
                item = dict(live_payload)
            else:
                item = {"raw": live_payload}

            item["battery"] = battery.upper()
            item["battery_id"] = battery.upper()
            item["tag"] = tag
            item["r_ratio"] = tag_to_ratio(tag)
            item["r_ratio_input"] = tag_to_ratio(tag)
            item["source_file"] = str(source_file) if source_file else None

            # Critical aliases for frontend.
            response = dict(item)
            response.update(meta)
            response["ok"] = True
            response["mode"] = "hf_front_app_real_live_reinfer_success"
            response["battery"] = battery.upper()
            response["battery_id"] = battery.upper()
            response["tag"] = tag
            response["r_ratio"] = tag_to_ratio(tag)
            response["r_ratio_input"] = tag_to_ratio(tag)
            response["source_file"] = str(source_file) if source_file else None

            response["item"] = item
            response["item_json"] = item
            response["live_item"] = item
            response["live_item_json"] = item
            response["reinfer_item"] = item
            response["reinfer_item_json"] = item
            response["items"] = [item]

            # Non-recursive wrappers.
            response["data"] = item
            response["payload"] = item
            response["result"] = {
                "ok": True,
                "item": item,
                "item_json": item,
                "items": [item],
            }

            return response


        async def run_reinfer(battery, tag):
            battery = str(battery).upper()
            r_ratio = tag_to_ratio(tag)

            live_out_dir = root / "data" / "live_reinfer_results"
            live_out_dir.mkdir(parents=True, exist_ok=True)

            script_candidates = [
                root / "scripts" / "export_rul_dashboard_data_meta_fixed.py",
                root / "export_rul_dashboard_data_meta_fixed.py",
                root / "scripts" / "prefix_inference_viz_meta_restored_v3_pyc_patched_json_v3.py",
            ]
            scripts = [x for x in script_candidates if x.exists()]

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = ":".join([
                str(root),
                str(root / "backend"),
                str(root / "deep_learning"),
                env.get("PYTHONPATH", ""),
            ])

            timeout = int(q1("timeout", "900"))
            attempts = []
            started = time.time()

            def find_new_live_json_files():
                search_dirs = [
                    live_out_dir,
                    root / "data" / "live_reinfer_results",
                    root / "data" / "precomputed_from_export_v2",
                    root / "data" / "precomputed",
                    Path("/tmp") / "battery_rul_live_reinfer_results",
                ]

                out = []
                for d in search_dirs:
                    try:
                        if not d.exists():
                            continue
                        for fp in d.rglob("*.json"):
                            if not fp.is_file():
                                continue
                            name = fp.name.upper()
                            if battery not in name and "B00" not in name:
                                continue
                            try:
                                if fp.stat().st_mtime >= started - 1:
                                    out.append(fp)
                            except Exception:
                                pass
                    except Exception:
                        pass

                return sorted(out, key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)

            def make_live_response(payload, source_file, meta):
                item = dict(payload) if isinstance(payload, dict) else {"raw": payload}

                item.update({
                    "battery": battery,
                    "battery_id": battery,
                    "tag": tag,
                    "r_ratio": r_ratio,
                    "r_ratio_input": r_ratio,
                    "source_file": str(source_file) if source_file else None,
                })

                # HF_INSERT_LIVE_STD_UNCERTAINTY_V2
                pred_obj = item.get("pred")
                std_seq = None
                
                if isinstance(pred_obj, dict):
                    std_seq = (
                        pred_obj.get("std")
                        or pred_obj.get("sigma")
                        or pred_obj.get("y_std")
                        or pred_obj.get("uncertainty")
                    )
                elif isinstance(pred_obj, list):
                    vals = []
                    for row in pred_obj:
                        if isinstance(row, dict):
                            vals.append(
                                row.get("std")
                                if row.get("std") is not None else
                                row.get("sigma")
                                if row.get("sigma") is not None else
                                row.get("y_std")
                                if row.get("y_std") is not None else
                                row.get("uncertainty")
                            )
                    std_seq = vals
                
                idx = (
                    item.get("currentCycleIndex")
                    if item.get("currentCycleIndex") is not None else
                    item.get("current_cycle_index")
                    if item.get("current_cycle_index") is not None else
                    item.get("q_pos")
                )
                
                try:
                    idx = int(idx)
                except Exception:
                    idx = None
                
                std_val = None
                if isinstance(std_seq, list) and std_seq:
                    if idx is None:
                        idx = len(std_seq) - 1
                    idx = max(0, min(idx, len(std_seq) - 1))
                    try:
                        std_val = float(std_seq[idx])
                    except Exception:
                        std_val = None
                
                if std_val is not None:
                    uncertainty = 2.0 * std_val
                    confidence = 1.0 / (1.0 + max(float(uncertainty), 0.0))
                    confidence_percent = round(confidence * 100.0, 2)
                
                    item["current_pred_std"] = std_val
                    item["uncertainty"] = uncertainty
                    item["prediction_uncertainty"] = uncertainty
                    item["pred_uncertainty"] = uncertainty
                    item["uncertainty_2sigma"] = uncertainty
                    item["uncertainty_plus_minus"] = uncertainty
                    item["uncertainty_band"] = uncertainty
                    item["confidence"] = confidence
                    item["confidence_percent"] = confidence_percent
                    item["prediction_confidence"] = confidence
                    item["prediction_confidence_percent"] = confidence_percent
                
                    metrics = item.get("metrics")
                    if not isinstance(metrics, dict):
                        metrics = {}
                    metrics.update({
                        "current_pred_std": std_val,
                        "uncertainty": uncertainty,
                        "prediction_uncertainty": uncertainty,
                        "uncertainty_2sigma": uncertainty,
                        "confidence": confidence,
                        "confidence_percent": confidence_percent,
                    })
                    item["metrics"] = metrics
                
                    if isinstance(pred_obj, dict):
                        pred_obj["current_std"] = std_val
                        pred_obj["currentCycleIndex"] = idx
                        pred_obj["uncertainty_2sigma"] = uncertainty
                        pred_obj["confidence"] = confidence
                        pred_obj["confidence_percent"] = confidence_percent
                        item["pred"] = pred_obj
                
                resp = dict(item)
                resp.update(meta)
                resp.update({
                    "ok": True,
                    "mode": "hf_front_app_real_live_reinfer_success",
                    "battery": battery,
                    "battery_id": battery,
                    "tag": tag,
                    "r_ratio": r_ratio,
                    "r_ratio_input": r_ratio,
                    "source_file": str(source_file) if source_file else None,

                    # frontend compatibility
                    "item": item,
                    "item_json": item,
                    "live_item": item,
                    "live_item_json": item,
                    "reinfer_item": item,
                    "reinfer_item_json": item,
                    "items": [item],
                    "data": item,
                    "payload": item,
                    "result": {
                        "ok": True,
                        "item": item,
                        "item_json": item,
                        "items": [item],
                    },
                })
                return resp

            if not scripts:
                return await send_json(500, {
                    "ok": False,
                    "mode": "hf_front_app_reinfer_failed_no_runner",
                    "error": "No reinference runner script found",
                    "searched": [str(x) for x in script_candidates],
                    "checkpoint": str(ckpt),
                    "checkpoint_exists": ckpt.exists(),
                })

            for script in scripts:
                cmd = [
                    sys.executable, str(script),
                    "--ckpt", str(ckpt),
                    "--eval_dataset", "from_ckpt",
                    "--r_ratio", r_ratio,
                    "--bids", battery,
                    "--out_dir", str(live_out_dir),
                    "--save_json", "1",
                    "--save_batch_json", "1",
                ]

                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(root),
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                    )

                    attempts.append({
                        "script": str(script),
                        "cmd": cmd,
                        "cwd": str(root),
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-5000:],
                        "stderr_tail": proc.stderr[-5000:],
                    })

                    if proc.returncode == 0:
                        files = find_new_live_json_files()

                        for fp in files:
                            payload = read_json(fp)
                            if isinstance(payload, dict):
                                return await send_json(200, make_live_response(payload, fp, {
                                    "checkpoint": str(ckpt),
                                    "checkpoint_exists": ckpt.exists(),
                                    "elapsed_sec": round(time.time() - started, 2),
                                    "runner_script": str(script),
                                    "runner_cmd": cmd,
                                    "stdout_tail": proc.stdout[-5000:],
                                    "stderr_tail": proc.stderr[-5000:],
                                    "new_live_json_candidates": [str(x) for x in files[:10]],
                                }))

                        return await send_json(500, {
                            "ok": False,
                            "mode": "hf_front_app_runner_success_but_no_live_item_json",
                            "error": "Runner returned 0 but no newly generated live JSON was found",
                            "battery": battery,
                            "tag": tag,
                            "checkpoint": str(ckpt),
                            "checkpoint_exists": ckpt.exists(),
                            "elapsed_sec": round(time.time() - started, 2),
                            "runner_script": str(script),
                            "runner_cmd": cmd,
                            "stdout_tail": proc.stdout[-5000:],
                            "stderr_tail": proc.stderr[-5000:],
                            "searched_out_dir": str(live_out_dir),
                            "new_live_json_candidates": [str(x) for x in files[:10]],
                        })

                except Exception as e:
                    attempts.append({
                        "script": str(script),
                        "cmd": cmd,
                        "cwd": str(root),
                        "exception": repr(e),
                    })

            return await send_json(500, {
                "ok": False,
                "mode": "hf_front_app_real_reinfer_failed_no_fake_success",
                "error": "Real runner failed",
                "battery": battery,
                "tag": tag,
                "checkpoint": str(ckpt),
                "checkpoint_exists": ckpt.exists(),
                "scripts": [str(x) for x in scripts],
                "elapsed_sec": round(time.time() - started, 2),
                "attempts": attempts[-10:],
            })

        if method == "OPTIONS":
            return await send_json(204, {"ok": True})

        # SHAP JSON
        if path in {
            "/api/fixed4/shap-current",
            "/api/fixed4/shap-global",
            "/api/fixed4/shap",
            "/api/shap-current",
            "/api/shap-global",
        }:
            return await send_json(200, shap_payload())

        # SHAP image
        if path in {
            "/api/fixed4/shap-image",
            "/api/fixed4/shap-current.png",
            "/api/fixed4/shap-global.png",
            "/api/shap-image",
        }:
            if shap_png.exists():
                return await send_bytes(200, shap_png.read_bytes(), "image/png")
            return await send_json(404, {
                "ok": False,
                "error": "SHAP PNG not found",
                "image_file": str(shap_png),
            })

        # Battery precomputed
        m = re.fullmatch(r"/api/battery/([^/]+)/precomputed", path)
        if m:
            battery = m.group(1).upper()
            tag = ratio_to_tag(q1("r_ratio", q1("r", q1("ratio", "0.1"))))
            pack, err = normalize_precomputed(battery, tag)
            if isinstance(pack, dict):
                return await send_json(200, pack)
            return await send_json(404, err)

        # Bundle precomputed
        m = re.fullmatch(r"/api/precomputed/(r[0-9]+p[0-9]+)", path)
        if m:
            tag = m.group(1)
            by_battery = {}
            errors = {}

            for bid in ["B0018", "B0042", "B0043"]:
                pack, err = normalize_precomputed(bid, tag)
                if isinstance(pack, dict):
                    by_battery[bid] = pack
                else:
                    errors[bid] = err

            selected = by_battery.get("B0043") or by_battery.get("B0018") or next(iter(by_battery.values()), {})

            out = {
                "ok": bool(by_battery),
                "source": "hf_front_app_exact_bundle",
                "tag": tag,
                "r_ratio": tag_to_ratio(tag),
                "r_ratio_input": tag_to_ratio(tag),
                "batteries": by_battery,
                "by_battery": by_battery,
                "battery": by_battery,
                "items": list(by_battery.values()),
                "errors": errors,
                "current": selected,
                "explainability": selected,
            }

            if isinstance(selected, dict):
                for key in [
                    "support", "query", "pred", "pred_std",
                    "uncertainty", "prediction_uncertainty", "pred_uncertainty",
                    "uncertainty_2sigma", "uncertainty_plus_minus",
                    "confidence", "confidence_percent",
                    "prediction_confidence", "prediction_confidence_percent",
                    "current_pred_rul", "pred_rul", "current_true_rul",
                    "current_cycle_effective", "currentCycleIndex",
                    "current_cycle_index", "metrics",
                ]:
                    out[key] = selected.get(key)

            return await send_json(200, out)

        # Reinference
        m = (
            re.fullmatch(r"/api/battery/([^/]+)/reinfer", path)
            or re.fullmatch(r"/api/live-reinfer-v[0-9]+/([^/]+)", path)
            or re.fullmatch(r"/api/live-reinfer/([^/]+)", path)
        )
        if m:
            battery = m.group(1).upper()
            tag = ratio_to_tag(q1("r_ratio", q1("r", q1("ratio", "0.1"))))
            return await run_reinfer(battery, tag)

        return await self.downstream(scope, receive, send)


app = HFFrontApp(downstream_app)

# === HF LIVE STD RESPONSE POSTPROCESS V1 ===
# Do not edit run_reinfer internals. Post-process successful live reinference JSON.
# uncertainty = 2 * item.pred.std[currentCycleIndex]
# confidence = 1 / (1 + uncertainty)

if "_HF_LIVE_STD_RESPONSE_POSTPROCESS_V1" not in globals():
    _HF_LIVE_STD_RESPONSE_POSTPROCESS_V1 = True

    class HFLiveStdResponsePostprocess:
        def __init__(self, downstream):
            self.downstream = downstream

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                return await self.downstream(scope, receive, send)

            import re
            import json
            import math

            path = scope.get("path", "")
            is_reinfer = bool(
                re.fullmatch(r"/api/battery/[^/]+/reinfer", path)
                or re.fullmatch(r"/api/live-reinfer-v[0-9]+/[^/]+", path)
                or re.fullmatch(r"/api/live-reinfer/[^/]+", path)
            )

            if not is_reinfer:
                return await self.downstream(scope, receive, send)

            captured = {
                "start": None,
                "body": b"",
            }

            async def capture_send(message):
                if message["type"] == "http.response.start":
                    captured["start"] = message
                elif message["type"] == "http.response.body":
                    captured["body"] += message.get("body", b"")
                    if not message.get("more_body", False):
                        await self._flush(captured, send)
                else:
                    await send(message)

            await self.downstream(scope, receive, capture_send)

        async def _flush(self, captured, send):
            import json
            import math

            start = captured.get("start") or {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
            body = captured.get("body") or b""

            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                await send(start)
                await send({"type": "http.response.body", "body": body})
                return

            def as_float(v):
                try:
                    if v is None:
                        return None
                    return float(v)
                except Exception:
                    return None

            def clean(x):
                if isinstance(x, float):
                    if math.isnan(x) or math.isinf(x):
                        return None
                    return x
                if isinstance(x, list):
                    return [clean(v) for v in x]
                if isinstance(x, dict):
                    return {str(k): clean(v) for k, v in x.items()}
                return x

            item = data.get("item")
            if not isinstance(item, dict):
                item = data.get("item_json")
            if not isinstance(item, dict):
                item = data.get("live_item_json")
            if not isinstance(item, dict):
                item = data if isinstance(data, dict) else {}

            pred = item.get("pred")
            std_seq = None

            if isinstance(pred, dict):
                std_seq = (
                    pred.get("std")
                    or pred.get("sigma")
                    or pred.get("y_std")
                    or pred.get("uncertainty")
                )
            elif isinstance(pred, list):
                vals = []
                for row in pred:
                    if isinstance(row, dict):
                        vals.append(
                            row.get("std")
                            if row.get("std") is not None else
                            row.get("sigma")
                            if row.get("sigma") is not None else
                            row.get("y_std")
                            if row.get("y_std") is not None else
                            row.get("uncertainty")
                        )
                std_seq = vals

            idx = (
                item.get("currentCycleIndex")
                if item.get("currentCycleIndex") is not None else
                item.get("current_cycle_index")
                if item.get("current_cycle_index") is not None else
                item.get("q_pos")
            )

            try:
                idx = int(idx)
            except Exception:
                idx = None

            std_val = None
            if isinstance(std_seq, list) and std_seq:
                if idx is None:
                    idx = len(std_seq) - 1
                idx = max(0, min(idx, len(std_seq) - 1))
                std_val = as_float(std_seq[idx])

            if std_val is not None:
                uncertainty = 2.0 * std_val
                confidence = 1.0 / (1.0 + max(float(uncertainty), 0.0))
                confidence_percent = round(confidence * 100.0, 2)

                item["current_pred_std"] = std_val
                item["uncertainty"] = uncertainty
                item["prediction_uncertainty"] = uncertainty
                item["pred_uncertainty"] = uncertainty
                item["uncertainty_2sigma"] = uncertainty
                item["uncertainty_plus_minus"] = uncertainty
                item["uncertainty_band"] = uncertainty
                item["confidence"] = confidence
                item["confidence_percent"] = confidence_percent
                item["prediction_confidence"] = confidence
                item["prediction_confidence_percent"] = confidence_percent

                metrics = item.get("metrics")
                if not isinstance(metrics, dict):
                    metrics = {}
                metrics.update({
                    "current_pred_std": std_val,
                    "uncertainty": uncertainty,
                    "prediction_uncertainty": uncertainty,
                    "uncertainty_2sigma": uncertainty,
                    "confidence": confidence,
                    "confidence_percent": confidence_percent,
                })
                item["metrics"] = metrics

                if isinstance(pred, dict):
                    pred["current_std"] = std_val
                    pred["currentCycleIndex"] = idx
                    pred["uncertainty_2sigma"] = uncertainty
                    pred["confidence"] = confidence
                    pred["confidence_percent"] = confidence_percent
                    item["pred"] = pred

                data["item"] = item
                data["item_json"] = item
                data["live_item"] = item
                data["live_item_json"] = item
                data["reinfer_item"] = item
                data["reinfer_item_json"] = item
                data["items"] = [item]
                data["data"] = item
                data["payload"] = item

                data["current_pred_std"] = std_val
                data["uncertainty"] = uncertainty
                data["prediction_uncertainty"] = uncertainty
                data["pred_uncertainty"] = uncertainty
                data["uncertainty_2sigma"] = uncertainty
                data["confidence"] = confidence
                data["confidence_percent"] = confidence_percent
                data["prediction_confidence"] = confidence
                data["prediction_confidence_percent"] = confidence_percent

            new_body = json.dumps(clean(data), ensure_ascii=False).encode("utf-8")

            headers = []
            for k, v in start.get("headers", []):
                lk = k.lower()
                if lk == b"content-length":
                    continue
                headers.append((k, v))
            headers.append((b"content-length", str(len(new_body)).encode("utf-8")))

            start["headers"] = headers

            await send(start)
            await send({"type": "http.response.body", "body": new_body})

    app = HFLiveStdResponsePostprocess(app)

# === HF UI LIVE PRECOMPUTED AND SHAP NAME FIX V1 ===
# Purpose:
# - UI may call GET /api/battery/{battery}/precomputed after POST /reinfer.
# - Make that GET prefer /app/data/live_reinfer_results/{battery}_viz_meta_{tag}.json.
# - Also parse SHAP feature names instead of returning feature_0, feature_1, ...

if "_HF_UI_LIVE_PRECOMPUTED_AND_SHAP_NAME_FIX_V1" not in globals():
    _HF_UI_LIVE_PRECOMPUTED_AND_SHAP_NAME_FIX_V1 = True

    class HFULivePrecomputedAndShapNameFix:
        def __init__(self, downstream):
            self.downstream = downstream

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                return await self.downstream(scope, receive, send)

            import re
            import json
            import math
            from pathlib import Path
            from urllib.parse import parse_qs

            method = scope.get("method", "GET").upper()
            path = scope.get("path", "")
            qs = parse_qs(scope.get("query_string", b"").decode("utf-8", errors="ignore"))

            root = Path("/app") if Path("/app").exists() else Path.cwd()

            def q1(name, default=None):
                vals = qs.get(name)
                return vals[0] if vals else default

            def ratio_to_tag(v):
                v = str(v or "0.1")
                if v.startswith("r"):
                    return v
                try:
                    return f"r{float(v):.2f}".replace(".", "p")
                except Exception:
                    return "r0p10"

            def tag_to_ratio(tag):
                tag = str(tag)
                return tag[1:].replace("p", ".") if tag.startswith("r") else tag

            def read_json(fp):
                try:
                    return json.loads(Path(fp).read_text())
                except Exception:
                    return None

            def as_float(v):
                try:
                    if v is None:
                        return None
                    return float(v)
                except Exception:
                    return None

            def clean(x):
                if isinstance(x, float):
                    if math.isnan(x) or math.isinf(x):
                        return None
                    return x
                if isinstance(x, list):
                    return [clean(v) for v in x]
                if isinstance(x, dict):
                    return {str(k): clean(v) for k, v in x.items()}
                return x

            async def send_json(status, obj):
                body = json.dumps(clean(obj), ensure_ascii=False).encode("utf-8")
                headers = [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"access-control-allow-origin", b"*"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(body)).encode("utf-8")),
                ]
                await send({"type": "http.response.start", "status": status, "headers": headers})
                await send({"type": "http.response.body", "body": body})

            def add_uncertainty_from_pred_std(payload, battery=None, tag=None, source_file=None, source=None):
                item = dict(payload) if isinstance(payload, dict) else {"raw": payload}

                if battery:
                    item["battery"] = str(battery).upper()
                    item["battery_id"] = str(battery).upper()
                if tag:
                    item["tag"] = tag
                    item["r_ratio"] = tag_to_ratio(tag)
                    item["r_ratio_input"] = tag_to_ratio(tag)
                if source_file:
                    item["source_file"] = str(source_file)
                if source:
                    item["source"] = source

                pred = item.get("pred")
                std_seq = None

                if isinstance(pred, dict):
                    std_seq = (
                        pred.get("std")
                        or pred.get("sigma")
                        or pred.get("y_std")
                        or pred.get("uncertainty")
                    )
                elif isinstance(pred, list):
                    vals = []
                    for row in pred:
                        if isinstance(row, dict):
                            vals.append(
                                row.get("std")
                                if row.get("std") is not None else
                                row.get("sigma")
                                if row.get("sigma") is not None else
                                row.get("y_std")
                                if row.get("y_std") is not None else
                                row.get("uncertainty")
                            )
                    std_seq = vals

                idx = (
                    item.get("currentCycleIndex")
                    if item.get("currentCycleIndex") is not None else
                    item.get("current_cycle_index")
                    if item.get("current_cycle_index") is not None else
                    item.get("q_pos")
                )

                try:
                    idx = int(idx)
                except Exception:
                    idx = None

                std_val = None
                if isinstance(std_seq, list) and std_seq:
                    if idx is None:
                        idx = len(std_seq) - 1
                    idx = max(0, min(idx, len(std_seq) - 1))
                    std_val = as_float(std_seq[idx])

                if std_val is not None:
                    uncertainty = 2.0 * std_val
                    confidence = 1.0 / (1.0 + max(float(uncertainty), 0.0))
                    confidence_percent = round(confidence * 100.0, 2)

                    item["current_pred_std"] = std_val
                    item["uncertainty"] = uncertainty
                    item["prediction_uncertainty"] = uncertainty
                    item["pred_uncertainty"] = uncertainty
                    item["uncertainty_2sigma"] = uncertainty
                    item["uncertainty_plus_minus"] = uncertainty
                    item["uncertainty_band"] = uncertainty
                    item["confidence"] = confidence
                    item["confidence_percent"] = confidence_percent
                    item["prediction_confidence"] = confidence
                    item["prediction_confidence_percent"] = confidence_percent

                    metrics = item.get("metrics")
                    if not isinstance(metrics, dict):
                        metrics = {}
                    metrics.update({
                        "current_pred_std": std_val,
                        "uncertainty": uncertainty,
                        "prediction_uncertainty": uncertainty,
                        "uncertainty_2sigma": uncertainty,
                        "confidence": confidence,
                        "confidence_percent": confidence_percent,
                    })
                    item["metrics"] = metrics

                    if isinstance(pred, dict):
                        pred["current_std"] = std_val
                        pred["currentCycleIndex"] = idx
                        pred["uncertainty_2sigma"] = uncertainty
                        pred["confidence"] = confidence
                        pred["confidence_percent"] = confidence_percent
                        item["pred"] = pred

                item["item"] = item
                item["item_json"] = item
                item["data"] = item
                item["payload"] = item
                item["result"] = {"ok": True, "item": item, "item_json": item, "items": [item]}
                item["items"] = [item]
                item["ok"] = True
                return item

            def load_precomputed_preferring_live(battery, tag):
                battery = str(battery).upper()
                live_fp = root / "data" / "live_reinfer_results" / f"{battery}_viz_meta_{tag}.json"
                pre_fp = root / "data" / "precomputed_from_export_v2" / f"{battery}_viz_meta_{tag}.json"

                if live_fp.exists():
                    raw = read_json(live_fp)
                    if isinstance(raw, dict):
                        return add_uncertainty_from_pred_std(
                            raw, battery=battery, tag=tag, source_file=live_fp,
                            source="hf_live_precomputed_prefers_live_reinfer_results"
                        )

                if pre_fp.exists():
                    raw = read_json(pre_fp)
                    if isinstance(raw, dict):
                        return add_uncertainty_from_pred_std(
                            raw, battery=battery, tag=tag, source_file=pre_fp,
                            source="hf_live_precomputed_fallback_precomputed_from_export_v2"
                        )

                return None

            known_feature_cols = [
                "anchor_cycle", "capacity_mean", "ambient_temp_c", "voltage_measured_mean",
                "voltage_min", "voltage_max", "voltage_std", "re_ohm_interp", "rct_ohm_interp",
                "temperature_measured_max", "temperature_mean", "temperature_min", "temperature_std",
                "temp_rise_cycle", "eff_c_rate", "current_mean", "current_std", "current_min",
                "current_max", "dvdt_max_abs", "dTdt_max", "soh", "capacity_derivative", "cap_vel",
                "regen_strength", "impedance_sum", "impedance_growth", "dcr", "dcr_growth",
                "lli", "lam", "temp_rise", "thermal_stress", "Residual", "IMF1", "IMF2",
                "IMF3", "IMF4", "IMF5", "IMF6"
            ]

            def shap_payload():
                shap_json = root / "deep_learning" / "core" / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"
                shap_png = root / "deep_learning" / "core" / "shap_outputs" / "bmaml_shap_seq_feature_importance.png"

                raw = read_json(shap_json)
                items = []

                def add_item(feature, value):
                    fv = as_float(value)
                    if feature is not None and fv is not None:
                        items.append({
                            "feature": str(feature),
                            "importance": fv,
                            "mean_abs_shap": fv,
                            "value": fv,
                        })

                if isinstance(raw, dict):
                    # Case 1: dict mapping feature -> value
                    for key in ["feature_importance", "global_importance", "mean_abs_shap", "importance"]:
                        val = raw.get(key)
                        if isinstance(val, dict):
                            for f, v in val.items():
                                add_item(f, v)

                    # Case 2: separate names and values arrays
                    name_keys = ["feature_names", "features", "columns", "seq_feature_names", "input_features"]
                    value_keys = ["values", "importances", "importance_values", "mean_abs_values", "mean_abs_shap_values", "shap_values", "importance"]

                    names = None
                    for k in name_keys:
                        v = raw.get(k)
                        if isinstance(v, list) and all(isinstance(x, str) for x in v):
                            names = v
                            break

                    values = None
                    for k in value_keys:
                        v = raw.get(k)
                        if isinstance(v, list) and all(as_float(x) is not None for x in v):
                            values = v
                            break

                    if names and values:
                        for f, v in zip(names, values):
                            add_item(f, v)

                    # Case 3: list of row dicts
                    for key in ["items", "data", "rows"]:
                        rows = raw.get(key)
                        if isinstance(rows, list):
                            for i, row in enumerate(rows):
                                if isinstance(row, dict):
                                    f = row.get("feature") or row.get("name") or row.get("column") or row.get("feature_name")
                                    v = (
                                        row.get("importance")
                                        if row.get("importance") is not None else
                                        row.get("mean_abs_shap")
                                        if row.get("mean_abs_shap") is not None else
                                        row.get("mean_abs")
                                        if row.get("mean_abs") is not None else
                                        row.get("value")
                                        if row.get("value") is not None else
                                        row.get("shap")
                                    )
                                    add_item(f or (known_feature_cols[i] if i < len(known_feature_cols) else f"feature_{i}"), v)
                                else:
                                    add_item(known_feature_cols[i] if i < len(known_feature_cols) else f"feature_{i}", row)

                    # Case 4: raw numeric list under an unknown key, use known feature order
                    if not items:
                        for key, val in raw.items():
                            if isinstance(val, list) and val and all(as_float(x) is not None for x in val):
                                for i, v in enumerate(val):
                                    add_item(known_feature_cols[i] if i < len(known_feature_cols) else f"feature_{i}", v)
                                break

                elif isinstance(raw, list):
                    for i, row in enumerate(raw):
                        if isinstance(row, dict):
                            f = row.get("feature") or row.get("name") or row.get("column") or row.get("feature_name")
                            v = (
                                row.get("importance")
                                if row.get("importance") is not None else
                                row.get("mean_abs_shap")
                                if row.get("mean_abs_shap") is not None else
                                row.get("mean_abs")
                                if row.get("mean_abs") is not None else
                                row.get("value")
                                if row.get("value") is not None else
                                row.get("shap")
                            )
                            add_item(f or (known_feature_cols[i] if i < len(known_feature_cols) else f"feature_{i}"), v)
                        else:
                            add_item(known_feature_cols[i] if i < len(known_feature_cols) else f"feature_{i}", row)

                # de-duplicate by feature
                dedup = {}
                for it in items:
                    dedup[it["feature"]] = it
                items = sorted(dedup.values(), key=lambda x: abs(float(x["importance"])), reverse=True)

                return {
                    "ok": bool(items),
                    "source": "hf_shap_exact_json_with_feature_names_v1",
                    "json_file": str(shap_json),
                    "json_exists": shap_json.exists(),
                    "image_file": str(shap_png),
                    "image_exists": shap_png.exists(),
                    "image_url": "/api/fixed4/shap-image",
                    "items": items,
                    "data": items,
                    "features": items,
                    "global_importance": items,
                }

            # Prefer live JSON for Explainability GET.
            m = re.fullmatch(r"/api/battery/([^/]+)/precomputed", path)
            if m and method in {"GET", "POST"}:
                battery = m.group(1)
                tag = ratio_to_tag(q1("r_ratio", q1("r", q1("ratio", "0.1"))))
                payload = load_precomputed_preferring_live(battery, tag)
                if payload is not None:
                    return await send_json(200, payload)

            # Bundle route, used by Overview/initial app load in some frontend paths.
            m = re.fullmatch(r"/api/precomputed/(r[0-9]+p[0-9]+)", path)
            if m and method in {"GET", "POST"}:
                tag = m.group(1)
                by_battery = {}
                for bid in ["B0018", "B0042", "B0043"]:
                    payload = load_precomputed_preferring_live(bid, tag)
                    if payload is not None:
                        by_battery[bid] = payload

                selected = by_battery.get("B0043") or by_battery.get("B0018") or next(iter(by_battery.values()), {})
                out = {
                    "ok": bool(by_battery),
                    "source": "hf_bundle_prefers_live_reinfer_results",
                    "tag": tag,
                    "r_ratio": tag_to_ratio(tag),
                    "r_ratio_input": tag_to_ratio(tag),
                    "batteries": by_battery,
                    "by_battery": by_battery,
                    "battery": by_battery,
                    "items": list(by_battery.values()),
                    "current": selected,
                    "explainability": selected,
                }
                if isinstance(selected, dict):
                    for k, v in selected.items():
                        if k not in out:
                            out[k] = v
                return await send_json(200, out)

            # SHAP endpoint with real feature names.
            if path in {
                "/api/fixed4/shap-current",
                "/api/fixed4/shap-global",
                "/api/fixed4/shap",
                "/api/shap-current",
                "/api/shap-global",
            } and method in {"GET", "POST"}:
                return await send_json(200, shap_payload())

            return await self.downstream(scope, receive, send)

    app = HFULivePrecomputedAndShapNameFix(app)

# === HF DARK SHAP IMAGE OVERRIDE V1 ===
# Serve dark SHAP PNG if present. Does not modify existing SHAP logic.

if "_HF_DARK_SHAP_IMAGE_OVERRIDE_V1" not in globals():
    _HF_DARK_SHAP_IMAGE_OVERRIDE_V1 = True

    class HFDarkShapImageOverrideV1:
        def __init__(self, downstream):
            self.downstream = downstream

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                return await self.downstream(scope, receive, send)

            from pathlib import Path

            method = scope.get("method", "GET").upper()
            path = scope.get("path", "")

            if path not in {
                "/api/fixed4/shap-image",
                "/api/fixed4/shap-current.png",
                "/api/fixed4/shap-global.png",
                "/api/shap-image",
            }:
                return await self.downstream(scope, receive, send)

            root = Path("/app") if Path("/app").exists() else Path.cwd()
            dark = root / "deep_learning" / "core" / "shap_outputs" / "bmaml_shap_seq_feature_importance_dark.png"
            raw = root / "deep_learning" / "core" / "shap_outputs" / "bmaml_shap_seq_feature_importance.png"

            fp = dark if dark.exists() else raw

            if not fp.exists():
                body = b'{"ok":false,"error":"SHAP image not found"}'
                await send({
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return

            body = fp.read_bytes()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"image/png"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": b"" if method == "HEAD" else body})

    app = HFDarkShapImageOverrideV1(app)

# === HF SAFE LIVE PRECOMPUTED OVERRIDE V2 ===
# Fix GET /api/battery/{battery}/precomputed 500.
# Frontend reads this after live reinference, so prefer live_reinfer_results.
# No self-referential dicts.

if "_HF_SAFE_LIVE_PRECOMPUTED_OVERRIDE_V2" not in globals():
    _HF_SAFE_LIVE_PRECOMPUTED_OVERRIDE_V2 = True

    class HFSafeLivePrecomputedOverrideV2:
        def __init__(self, downstream):
            self.downstream = downstream

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                return await self.downstream(scope, receive, send)

            import re
            import json
            import math
            from pathlib import Path
            from urllib.parse import parse_qs

            method = scope.get("method", "GET").upper()
            path = scope.get("path", "")
            qs = parse_qs(scope.get("query_string", b"").decode("utf-8", errors="ignore"))

            root = Path("/app") if Path("/app").exists() else Path.cwd()

            def q1(name, default=None):
                vals = qs.get(name)
                return vals[0] if vals else default

            def ratio_to_tag(v):
                v = str(v or "0.1")
                if v.startswith("r"):
                    return v
                try:
                    return f"r{float(v):.2f}".replace(".", "p")
                except Exception:
                    return "r0p10"

            def tag_to_ratio(tag):
                tag = str(tag)
                return tag[1:].replace("p", ".") if str(tag).startswith("r") else str(tag)

            def read_json(fp):
                try:
                    return json.loads(Path(fp).read_text())
                except Exception:
                    return None

            def as_float(v):
                try:
                    if v is None:
                        return None
                    return float(v)
                except Exception:
                    return None

            def clean(x):
                if isinstance(x, float):
                    if math.isnan(x) or math.isinf(x):
                        return None
                    return x
                if isinstance(x, list):
                    return [clean(v) for v in x]
                if isinstance(x, dict):
                    return {str(k): clean(v) for k, v in x.items()}
                return x

            async def send_json(status, obj):
                body = json.dumps(clean(obj), ensure_ascii=False).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"access-control-allow-origin", b"*"),
                        (b"cache-control", b"no-store"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                })
                await send({"type": "http.response.body", "body": body})

            def add_live_uncertainty(payload, battery, tag, fp, source):
                item = dict(payload)

                item["ok"] = True
                item["battery"] = battery
                item["battery_id"] = battery
                item["tag"] = tag
                item["r_ratio"] = tag_to_ratio(tag)
                item["r_ratio_input"] = tag_to_ratio(tag)
                item["source"] = source
                item["source_file"] = str(fp)

                pred = item.get("pred")
                std_seq = None

                if isinstance(pred, dict):
                    std_seq = pred.get("std") or pred.get("sigma") or pred.get("y_std") or pred.get("uncertainty")
                elif isinstance(pred, list):
                    vals = []
                    for row in pred:
                        if isinstance(row, dict):
                            vals.append(
                                row.get("std")
                                if row.get("std") is not None else
                                row.get("sigma")
                                if row.get("sigma") is not None else
                                row.get("y_std")
                                if row.get("y_std") is not None else
                                row.get("uncertainty")
                            )
                    std_seq = vals

                idx = (
                    item.get("currentCycleIndex")
                    if item.get("currentCycleIndex") is not None else
                    item.get("current_cycle_index")
                    if item.get("current_cycle_index") is not None else
                    item.get("q_pos")
                )

                try:
                    idx = int(idx)
                except Exception:
                    idx = None

                std_val = None
                if isinstance(std_seq, list) and std_seq:
                    if idx is None:
                        idx = len(std_seq) - 1
                    idx = max(0, min(idx, len(std_seq) - 1))
                    std_val = as_float(std_seq[idx])

                if std_val is not None:
                    uncertainty = 2.0 * std_val
                    confidence = 1.0 / (1.0 + max(float(uncertainty), 0.0))
                    confidence_percent = round(confidence * 100.0, 2)

                    item["current_pred_std"] = std_val
                    item["uncertainty"] = uncertainty
                    item["prediction_uncertainty"] = uncertainty
                    item["pred_uncertainty"] = uncertainty
                    item["uncertainty_2sigma"] = uncertainty
                    item["uncertainty_plus_minus"] = uncertainty
                    item["uncertainty_band"] = uncertainty
                    item["confidence"] = confidence
                    item["confidence_percent"] = confidence_percent
                    item["prediction_confidence"] = confidence
                    item["prediction_confidence_percent"] = confidence_percent

                    metrics = item.get("metrics")
                    if not isinstance(metrics, dict):
                        metrics = {}
                    metrics.update({
                        "current_pred_std": std_val,
                        "uncertainty": uncertainty,
                        "prediction_uncertainty": uncertainty,
                        "uncertainty_2sigma": uncertainty,
                        "confidence": confidence,
                        "confidence_percent": confidence_percent,
                    })
                    item["metrics"] = metrics

                    if isinstance(pred, dict):
                        pred = dict(pred)
                        pred["current_std"] = std_val
                        pred["currentCycleIndex"] = idx
                        pred["uncertainty_2sigma"] = uncertainty
                        pred["confidence"] = confidence
                        pred["confidence_percent"] = confidence_percent
                        item["pred"] = pred

                # wrappers must be copies, not item["item"] = item
                wrapper_item = dict(item)
                out = dict(item)
                out["item"] = wrapper_item
                out["item_json"] = wrapper_item
                out["live_item"] = wrapper_item
                out["live_item_json"] = wrapper_item
                out["items"] = [wrapper_item]
                out["data"] = wrapper_item
                out["payload"] = wrapper_item
                out["result"] = {"ok": True, "item": wrapper_item, "item_json": wrapper_item, "items": [wrapper_item]}
                return out

            def load_pack(battery, tag):
                battery = battery.upper()
                live = root / "data" / "live_reinfer_results" / f"{battery}_viz_meta_{tag}.json"
                pre = root / "data" / "precomputed_from_export_v2" / f"{battery}_viz_meta_{tag}.json"

                if live.exists():
                    raw = read_json(live)
                    if isinstance(raw, dict):
                        return add_live_uncertainty(raw, battery, tag, live, "hf_safe_live_precomputed_override_v2_live")

                if pre.exists():
                    raw = read_json(pre)
                    if isinstance(raw, dict):
                        return add_live_uncertainty(raw, battery, tag, pre, "hf_safe_live_precomputed_override_v2_precomputed")

                return None

            m = re.fullmatch(r"/api/battery/([^/]+)/precomputed", path)
            if m and method in {"GET", "POST"}:
                battery = m.group(1).upper()
                tag = ratio_to_tag(q1("r_ratio", q1("r", q1("ratio", "0.1"))))
                pack = load_pack(battery, tag)
                if pack is not None:
                    return await send_json(200, pack)
                return await send_json(404, {
                    "ok": False,
                    "error": "precomputed/live pack not found",
                    "battery": battery,
                    "tag": tag,
                })

            m = re.fullmatch(r"/api/precomputed/(r[0-9]+p[0-9]+)", path)
            if m and method in {"GET", "POST"}:
                tag = m.group(1)
                by_battery = {}
                for bid in ["B0018", "B0042", "B0043"]:
                    pack = load_pack(bid, tag)
                    if pack is not None:
                        by_battery[bid] = pack

                selected = by_battery.get("B0043") or by_battery.get("B0018") or next(iter(by_battery.values()), {})
                out = {
                    "ok": bool(by_battery),
                    "source": "hf_safe_live_precomputed_override_v2_bundle",
                    "tag": tag,
                    "r_ratio": tag_to_ratio(tag),
                    "r_ratio_input": tag_to_ratio(tag),
                    "batteries": by_battery,
                    "by_battery": by_battery,
                    "battery": by_battery,
                    "items": list(by_battery.values()),
                    "current": selected,
                    "explainability": selected,
                }
                if isinstance(selected, dict):
                    for k, v in selected.items():
                        if k not in out:
                            out[k] = v
                return await send_json(200, out)

            return await self.downstream(scope, receive, send)

    app = HFSafeLivePrecomputedOverrideV2(app)
