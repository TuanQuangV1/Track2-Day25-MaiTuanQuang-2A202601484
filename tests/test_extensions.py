"""Unit tests for Lab 25 FinOps Extensions (Rubric Part D)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, metrics, sustainability


def test_recommend_tier_enhanced():
    # Test spot tier recommendation with GPU type
    assert pricing.recommend_tier(6, True, gpu_type="H100") == "spot"
    assert pricing.recommend_tier(6, True, gpu_type="A10G") == "spot"

    # Test reserved tier recommendation with high duty cycle
    assert pricing.recommend_tier(20, False, gpu_type="H100") == "reserved"

    # Test on-demand for low duty cycle non-interruptible
    assert pricing.recommend_tier(4, False, gpu_type="H100") == "on_demand"

    # Test custom high interruption rate forcing on-demand/fallback
    assert pricing.recommend_tier(6, True, interrupt_rate=0.25) == "on_demand"


def test_cache_is_worth_it_and_break_even():
    # Break-even reads formula: write_cost / (price_in * (1 - read_discount))
    # E.g., write=$2.70, price_in=$3.00, discount=0.10 (90% off -> save $2.70/M per read)
    # BE = 2.70 / (3.00 * 0.9) = 1.0 read
    be = pricing.cache_break_even_reads(write_cost_per_m=2.70, price_in_per_m=3.00, read_discount=0.10)
    assert abs(be - 1.0) < 1e-6

    # Test cache_is_worth_it decision
    assert pricing.cache_is_worth_it(avg_cache_reads=2.0, write_cost_per_m=2.70, read_discount=0.10, price_in_per_m=3.00) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=0.5, write_cost_per_m=2.70, read_discount=0.10, price_in_per_m=3.00) is False


def test_hardware_unit_economics_and_mbu_rightsizing():
    catalog_sample = [
        {"gpu_type": "H100", "on_demand_hr": 2.50, "hbm_gb": 80, "peak_bw_tbs": 3.35, "peak_tflops_fp16": 990, "watts": 700},
        {"gpu_type": "A100", "on_demand_hr": 1.79, "hbm_gb": 80, "peak_bw_tbs": 2.00, "peak_tflops_fp16": 312, "watts": 400},
        {"gpu_type": "A10G", "on_demand_hr": 1.00, "hbm_gb": 24, "peak_bw_tbs": 0.60, "peak_tflops_fp16": 125, "watts": 150},
    ]
    econ = metrics.hardware_unit_economics(catalog_sample)
    assert len(econ) == 3
    # Check cost per GB VRAM for H100: 2.50 / 80 = 0.03125 $/GB-hr
    h100 = next(e for e in econ if e["gpu_type"] == "H100")
    assert abs(h100["cost_per_gb_vram_hr"] - 0.03125) < 1e-4

    cat_map = {row["gpu_type"]: row for row in catalog_sample}
    # Current is H100 ($2.50/hr, 3.35 TB/s, 80GB), but workload only needs 1.5 TB/s BW and 40GB VRAM
    # A100 ($1.79/hr, 2.0 TB/s, 80GB) satisfies the requirements and is cheaper
    rec = metrics.recommend_rightsize_mbu("H100", achieved_bw_tbs=1.5, vram_needed_gb=40, catalog_by_type=cat_map)
    assert rec["recommended_gpu"] == "A100"
    assert rec["savings_hr"] > 0
    assert rec["savings_pct"] > 0


def test_sustainability_region_comparison():
    wh = 10000.0  # 10 kWh
    results = sustainability.compare_regions(wh)
    assert len(results) >= 5

    cleanest = min(results, key=lambda x: x["carbon_intensity_g_kwh"])
    assert cleanest["region"] == "europe-north1"

    cheapest = min(results, key=lambda x: x["price_kwh"])
    assert cheapest["region"] == "us-east-wa"

    # Check workload calculator
    calc = sustainability.calculate_workload_energy_carbon(gpu_watts=700, hours=10, num_gpus=2, region="us-east-1")
    # 700W * 2 * 10h = 14,000 Wh = 14 kWh
    assert abs(calc["kwh"] - 14.0) < 1e-2
    assert calc["energy_cost_usd"] > 0
    assert calc["carbon_kg"] > 0
