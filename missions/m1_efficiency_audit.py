"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: Right-sizing analysis based on MBU & $/GB-VRAM ---
    price_cat_rows = load_csv("price_catalog.csv")
    hw_econ = metrics.hardware_unit_economics(price_cat_rows)

    rightsize_recs = []
    monthly_rightsize_savings = 0.0
    for s in summary:
        gtype = s["gpu_type"]
        cur_bw = num(cat[gtype]["peak_bw_tbs"]) * s["mbu"]
        if s["mbu"] < 0.60 and s["mfu"] < 0.35 and gtype in ("H100", "A100", "A10G"):
            vram_needed = num(cat[gtype]["hbm_gb"]) * 0.5
            rec = metrics.recommend_rightsize_mbu(gtype, cur_bw, vram_needed, cat)
            if rec["savings_hr"] > 0:
                rightsize_recs.append({"gpu_id": s["gpu_id"], **rec})
                monthly_rightsize_savings += rec["savings_hr"] * 24 * 30

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n--- Extension 2: Hardware Unit Economics ($/GB-VRAM, $/(TB/s) BW) ---")
        print(f"{'GPU Type':10}{'$/hr':>7}{'VRAM(GB)':>10}{'BW(TB/s)':>10}{'$/GB-hr':>10}{'$/(TB/s)-hr':>14}{'TFLOPs/$':>10}")
        for h in hw_econ:
            print(f"{h['gpu_type']:10}${h['on_demand_hr']:>6.2f}{h['hbm_gb']:>10.0f}{h['peak_bw_tbs']:>10.2f}${h['cost_per_gb_vram_hr']:>9.4f}${h['cost_per_tbs_bw_hr']:>13.4f}{h['tflops_per_dollar_hr']:>10.1f}")

        if rightsize_recs:
            print("\n--- Extension 2: MBU Right-Sizing Recommendations ---")
            for r in rightsize_recs:
                print(f"  {r['gpu_id']}: {r['current_gpu']} (${r['current_price_hr']}/hr) -> {r['recommended_gpu']} (${r['recommended_price_hr']}/hr) | Save: ${r['savings_hr']}/hr ({r['savings_pct']}%)")
            print(f"  Total potential monthly right-sizing savings: ${monthly_rightsize_savings:,.0f}/month")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "hardware_economics": hw_econ,
        "rightsize_recommendations": rightsize_recs,
        "monthly_rightsize_savings": round(monthly_rightsize_savings, 2),
    }


if __name__ == "__main__":
    run()

