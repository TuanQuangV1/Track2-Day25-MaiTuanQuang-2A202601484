"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


def hardware_unit_economics(catalog_rows: list[dict]) -> list[dict]:
    """Compute unit economic metrics ($/GB VRAM, $/(TB/s) BW, TFLOPs/$) across GPU catalog.

    Essential for right-sizing memory-bound vs compute-bound workloads (FinOps Extension 2).
    """
    results = []
    for row in catalog_rows:
        od = float(row.get("on_demand_hr", 0.0))
        vram = float(row.get("hbm_gb", 0.0))
        bw = float(row.get("peak_bw_tbs", 0.0))
        tflops = float(row.get("peak_tflops_fp16", 0.0))
        watts = float(row.get("watts", 0.0))

        cost_per_gb_vram_hr = (od / vram) if vram > 0 else float("inf")
        cost_per_tbs_bw_hr = (od / bw) if bw > 0 else float("inf")
        tflops_per_dollar_hr = (tflops / od) if od > 0 else 0.0
        watts_per_tflop = (watts / tflops) if tflops > 0 else 0.0

        results.append({
            "gpu_type": row.get("gpu_type"),
            "on_demand_hr": od,
            "hbm_gb": vram,
            "peak_bw_tbs": bw,
            "peak_tflops_fp16": tflops,
            "cost_per_gb_vram_hr": round(cost_per_gb_vram_hr, 4),
            "cost_per_tbs_bw_hr": round(cost_per_tbs_bw_hr, 4),
            "tflops_per_dollar_hr": round(tflops_per_dollar_hr, 2),
            "watts_per_tflop": round(watts_per_tflop, 4),
        })
    return results


def recommend_rightsize_mbu(
    current_gpu: str,
    achieved_bw_tbs: float,
    vram_needed_gb: float,
    catalog_by_type: dict,
) -> dict:
    """Recommend a cheaper GPU replacement if current GPU is over-provisioned for memory bandwidth.

    Finds the cheapest GPU in catalog that satisfies:
      1. peak_bw_tbs >= achieved_bw_tbs * 1.15 (15% headroom)
      2. hbm_gb >= vram_needed_gb
    """
    cur_info = catalog_by_type.get(current_gpu)
    if not cur_info:
        return {"recommended_gpu": current_gpu, "savings_hr": 0.0, "savings_pct": 0.0}

    cur_price = float(cur_info["on_demand_hr"])
    best_gpu = current_gpu
    best_price = cur_price

    for gtype, info in catalog_by_type.items():
        price = float(info["on_demand_hr"])
        bw = float(info["peak_bw_tbs"])
        vram = float(info["hbm_gb"])

        if price < best_price and bw >= achieved_bw_tbs * 1.15 and vram >= vram_needed_gb:
            best_gpu = gtype
            best_price = price

    savings_hr = cur_price - best_price
    savings_pct = (savings_hr / cur_price * 100.0) if cur_price > 0 else 0.0
    return {
        "current_gpu": current_gpu,
        "recommended_gpu": best_gpu,
        "current_price_hr": cur_price,
        "recommended_price_hr": best_price,
        "savings_hr": round(savings_hr, 3),
        "savings_pct": round(savings_pct, 1),
    }
