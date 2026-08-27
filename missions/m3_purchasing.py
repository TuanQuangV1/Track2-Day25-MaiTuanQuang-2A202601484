"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []

    # Tracking for Extension 5: Carbon-aware scheduling for interruptible workloads
    total_interruptible_kwh = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = int(num(j.get("days", 30)))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Use enhanced recommend_tier supporting gpu_type & job_days
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=job_days)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
                     "days": job_days, "interruptible": interruptible})

        if interruptible:
            watts = num(c["watts"])
            total_interruptible_kwh += (watts * ngpu * hpd * DAYS) / 1000.0

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # --- Extension 5: 5-Region Carbon & Cost Comparison ---
    region_comparison = []
    base_region = "us-east-1"
    base_carbon = (total_interruptible_kwh * sustainability.REGION_CARBON[base_region]) / 1000.0
    base_elec_cost = total_interruptible_kwh * sustainability.REGION_PRICE_KWH[base_region]

    for reg, c_int in sustainability.REGION_CARBON.items():
        price_kwh = sustainability.REGION_PRICE_KWH[reg]
        cost = total_interruptible_kwh * price_kwh
        carbon_kg = (total_interruptible_kwh * c_int) / 1000.0
        saved_carbon = base_carbon - carbon_kg
        saved_cost = base_elec_cost - cost
        region_comparison.append({
            "region": reg,
            "price_kwh": price_kwh,
            "carbon_intensity_g_kwh": c_int,
            "monthly_elec_cost": round(cost, 2),
            "monthly_carbon_kg": round(carbon_kg, 1),
            "saved_carbon_kg": round(saved_carbon, 1),
            "saved_cost_usd": round(saved_cost, 2),
        })

    region_comparison.sort(key=lambda x: x["monthly_carbon_kg"])

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n--- Extension 5: Carbon-Aware Scheduling (Interruptible Workloads) ---")
        print(f"Total interruptible monthly energy: {total_interruptible_kwh:,.0f} kWh")
        print(f"{'Region':18}{'$/kWh':>8}{'gCO2/kWh':>10}{'Elec Cost':>12}{'Carbon (kg)':>14}{'CO2 Saved vs US-East':>22}")
        for reg in region_comparison:
            print(f"{reg['region']:18}${reg['price_kwh']:>7.3f}{reg['carbon_intensity_g_kwh']:>10}${reg['monthly_elec_cost']:>11.2f}{reg['monthly_carbon_kg']:>14.1f} kg{reg['saved_carbon_kg']:>19.1f} kg")

        best_clean = region_comparison[0]
        cheapest_reg = min(region_comparison, key=lambda x: x["monthly_elec_cost"])
        print(f"\n  * Cleanest Region: {best_clean['region']} ({best_clean['carbon_intensity_g_kwh']} gCO2/kWh) -> Cuts carbon by {best_clean['saved_carbon_kg']:,.1f} kg ({best_clean['saved_carbon_kg']/base_carbon:.1%})")
        print(f"  * Cheapest Electricity: {cheapest_reg['region']} (${cheapest_reg['price_kwh']}/kWh) -> Cuts power bill by ${cheapest_reg['saved_cost_usd']:,.2f}/mo")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "extension_carbon_scheduling": {
            "total_interruptible_kwh": round(total_interruptible_kwh, 2),
            "region_comparison": region_comparison,
        }
    }


if __name__ == "__main__":
    run()

