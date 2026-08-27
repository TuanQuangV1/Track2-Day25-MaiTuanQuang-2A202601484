"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
# Illustrative cache write cost ($/1M tokens) snapshot
CACHE_WRITE_COST = {"small": 0.25, "large": 3.75}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Tracking for Extension 4: Reasoning budget & energy breakdown
    reasoning_reqs = non_reasoning_reqs = 0
    reasoning_tokens = non_reasoning_tokens = 0
    reasoning_cost = non_reasoning_cost = 0.0
    reasoning_wh = non_reasoning_wh = 0.0

    # Tracking for Extension 3: Cache analysis
    total_cached_tokens = 0
    total_input_tokens = 0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))

        req_tokens = inp + out
        total_tokens += req_tokens
        total_input_tokens += inp
        total_cached_tokens += cached

        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        req_base = pricing.request_cost(inp, out, lin, lout)
        base_cost += req_base

        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        req_opt = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        opt_cost += req_opt

        # Energy calculation
        req_wh = sustainability.wh_per_query(req_tokens, is_reasoning=is_reasoning)

        if is_reasoning:
            reasoning_reqs += 1
            reasoning_tokens += req_tokens
            reasoning_cost += req_opt
            reasoning_wh += req_wh
        else:
            non_reasoning_reqs += 1
            non_reasoning_tokens += req_tokens
            non_reasoning_cost += req_opt
            non_reasoning_wh += req_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # --- Extension 3 Metrics: Cache economics ---
    cache_be_small = pricing.cache_break_even_reads(
        CACHE_WRITE_COST["small"], MODEL_PRICES["small"][0], read_discount=0.10
    )
    cache_be_large = pricing.cache_break_even_reads(
        CACHE_WRITE_COST["large"], MODEL_PRICES["large"][0], read_discount=0.10
    )
    overall_cache_hit_rate = (total_cached_tokens / total_input_tokens) if total_input_tokens else 0.0
    estimated_avg_reads = 4.5
    cache_worth_small = pricing.cache_is_worth_it(estimated_avg_reads, CACHE_WRITE_COST["small"], 0.10, MODEL_PRICES["small"][0])
    cache_worth_large = pricing.cache_is_worth_it(estimated_avg_reads, CACHE_WRITE_COST["large"], 0.10, MODEL_PRICES["large"][0])

    # --- Extension 4 Metrics: Reasoning budget ---
    n_total = len(rows)
    reasoning_req_pct = (reasoning_reqs / n_total * 100.0) if n_total else 0.0
    reasoning_cost_pct = (reasoning_cost / opt_cost * 100.0) if opt_cost else 0.0
    reasoning_energy_pct = (reasoning_wh / (reasoning_wh + non_reasoning_wh) * 100.0) if (reasoning_wh + non_reasoning_wh) else 0.0

    capped_reasoning_count = int(n_total * 0.05)
    excess_reasoning = max(0, reasoning_reqs - capped_reasoning_count)
    avg_cost_reasoning = reasoning_cost / reasoning_reqs if reasoning_reqs else 0.0
    avg_cost_non_reasoning = non_reasoning_cost / non_reasoning_reqs if non_reasoning_reqs else 0.0
    avg_wh_reasoning = reasoning_wh / reasoning_reqs if reasoning_reqs else 0.0
    avg_wh_non_reasoning = non_reasoning_wh / non_reasoning_reqs if non_reasoning_reqs else 0.0

    reasoning_cap_savings_daily = excess_reasoning * (avg_cost_reasoning - avg_cost_non_reasoning)
    reasoning_cap_wh_saved_daily = excess_reasoning * (avg_wh_reasoning - avg_wh_non_reasoning)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n--- Extension 3: Prompt Caching Economics ---")
        print(f"  Overall input cache hit rate: {overall_cache_hit_rate:.1%}")
        print(f"  Break-even cache reads required: Small Model = {cache_be_small:.2f} reads | Large Model = {cache_be_large:.2f} reads")
        print(f"  Is cache economically beneficial at ~{estimated_avg_reads} avg reads? Small: {cache_worth_small} | Large: {cache_worth_large}")

        print("\n--- Extension 4: Reasoning Budget & Energy Surge ---")
        print(f"  Reasoning traffic: {reasoning_reqs} reqs ({reasoning_req_pct:.1f}% of traffic)")
        print(f"  Reasoning cost: ${reasoning_cost:.2f} ({reasoning_cost_pct:.1f}% of optimized inference bill)")
        print(f"  Reasoning energy: {reasoning_wh/1000.0:.2f} kWh ({reasoning_energy_pct:.1f}% of total inference power)")
        print(f"  If capped to 5% traffic with threshold routing:")
        print(f"    Daily savings: ${reasoning_cap_savings_daily:.2f}/day (${reasoning_cap_savings_daily*30:.0f}/month) | Power saved: {reasoning_cap_wh_saved_daily/1000.0:.2f} kWh/day")

    return {
        "baseline_daily": round(base_cost, 2),
        "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3),
        "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1),
        "total_tokens": total_tokens,
        "extension_cache": {
            "overall_cache_hit_rate": round(overall_cache_hit_rate, 3),
            "break_even_reads_small": round(cache_be_small, 2),
            "break_even_reads_large": round(cache_be_large, 2),
            "cache_worth_small": cache_worth_small,
            "cache_worth_large": cache_worth_large,
        },
        "extension_reasoning": {
            "reasoning_req_pct": round(reasoning_req_pct, 1),
            "reasoning_cost_pct": round(reasoning_cost_pct, 1),
            "reasoning_energy_pct": round(reasoning_energy_pct, 1),
            "monthly_cap_savings_usd": round(reasoning_cap_savings_daily * 30, 2),
            "monthly_cap_kwh_saved": round(reasoning_cap_wh_saved_daily * 30 / 1000.0, 2),
        }
    }


if __name__ == "__main__":
    run()

