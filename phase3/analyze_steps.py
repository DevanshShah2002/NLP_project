"""
analyze_steps.py
-----------------
Analyzes average LLM calls / steps / attempts per query
including retry overhead from the pipeline results JSON.

Usage:
    python analyze_steps.py                        # uses default file
    python analyze_steps.py phase3_results.json    # specify file
"""

import json
import sys
from collections import defaultdict

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "phase3_results_all.json"

def get_hop_type(qid: str) -> str:
    for hop in ["4hop3", "4hop2", "4hop1", "3hop2", "3hop1", "2hop"]:
        if qid.startswith(hop):
            return hop
    return "unknown"

def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    n = len(results)

    # Fix hop_type from ID — same logic as check_results.py
    for r in results:
        r["hop_type"] = get_hop_type(r["id"])

    # ── Global counters ────────────────────────────────────────────────────
    total_steps          = 0
    total_attempts       = 0
    total_extra_attempts = 0
    queries_with_retry   = 0
    steps_with_retry     = 0

    attempt_dist = defaultdict(int)

    hop_stats = defaultdict(lambda: {
        "count": 0, "total_steps": 0,
        "total_attempts": 0, "total_extra_attempts": 0,
        "queries_with_retry": 0,
    })
    k_stats = defaultdict(lambda: {
        "count": 0, "total_steps": 0,
        "total_attempts": 0, "total_extra_attempts": 0,
    })

    for r in results:
        steps    = r.get("steps", [])
        hop_type = r["hop_type"]
        k        = r.get("k", len(steps))

        num_steps      = len(steps)
        query_attempts = 0
        query_extras   = 0
        had_retry      = False

        for step in steps:
            att = step.get("attempts", 1)
            query_attempts += att
            extras          = att - 1
            query_extras   += extras
            attempt_dist[att] += 1
            total_steps += 1
            if att > 1:
                steps_with_retry += 1
                had_retry = True

        total_attempts       += query_attempts
        total_extra_attempts += query_extras
        if had_retry:
            queries_with_retry += 1

        hs = hop_stats[hop_type]
        hs["count"]                += 1
        hs["total_steps"]          += num_steps
        hs["total_attempts"]       += query_attempts
        hs["total_extra_attempts"] += query_extras
        if had_retry:
            hs["queries_with_retry"] += 1

        ks = k_stats[k]
        ks["count"]                += 1
        ks["total_steps"]          += num_steps
        ks["total_attempts"]       += query_attempts
        ks["total_extra_attempts"] += query_extras

    avg_steps             = total_steps    / n
    avg_attempts          = total_attempts / n
    avg_attempts_per_step = total_attempts / total_steps if total_steps else 0
    avg_extras_per_query  = total_extra_attempts / n

    print(f"\nTotal queries loaded : {n}")

    print("\n" + "=" * 60)
    print("  LLM CALL ANALYSIS  (answering LLM only)")
    print("=" * 60)
    print(f"  Total step executions         : {total_steps}")
    print(f"  Total attempts (all steps)    : {total_attempts}")
    print(f"  Total extra attempts (retries): {total_extra_attempts}")
    print()
    print(f"  Avg steps per query           : {avg_steps:.3f}")
    print(f"  Avg LLM calls per query       : {avg_attempts:.3f}   <- main number")
    print(f"  Avg attempts per step         : {avg_attempts_per_step:.3f}")
    print(f"  Avg extra (retry) calls/query : {avg_extras_per_query:.3f}")
    print()
    print(f"  Queries with at least 1 retry : {queries_with_retry}  ({queries_with_retry/n*100:.1f}%)")
    print(f"  Steps that needed retry       : {steps_with_retry}  ({steps_with_retry/total_steps*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  ATTEMPT DISTRIBUTION (per step)")
    print("=" * 60)
    print(f"  {'Attempts':>8} | {'Steps':>6} | {'%':>6} | bar")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'─'*20}")
    for att in sorted(attempt_dist):
        cnt = attempt_dist[att]
        pct = cnt / total_steps * 100
        bar = "█" * int(pct / 2)
        print(f"  {att:>8} | {cnt:>6} | {pct:>5.1f}% | {bar}")

    print("\n" + "=" * 60)
    print("  AVG LLM CALLS BY HOP TYPE")
    print("=" * 60)
    print(f"  {'Hop':>6} | {'N':>5} | {'Avg Steps':>9} | {'Avg LLM Calls':>13} | {'Avg Retries':>11} | {'% w/ Retry':>10}")
    print(f"  {'─'*6}-+-{'─'*5}-+-{'─'*9}-+-{'─'*13}-+-{'─'*11}-+-{'─'*10}")
    for hop in ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3", "unknown"]:
        hs = hop_stats.get(hop)
        if not hs or hs["count"] == 0:
            continue
        hn    = hs["count"]
        avg_s = hs["total_steps"]          / hn
        avg_a = hs["total_attempts"]       / hn
        avg_e = hs["total_extra_attempts"] / hn
        pct_r = hs["queries_with_retry"]   / hn * 100
        print(f"  {hop:>6} | {hn:>5} | {avg_s:>9.2f} | {avg_a:>13.2f} | {avg_e:>11.2f} | {pct_r:>9.1f}%")

    print("\n" + "=" * 60)
    print("  AVG LLM CALLS BY k (sub-question count)")
    print("=" * 60)
    print(f"  {'k':>4} | {'N':>5} | {'Avg Steps':>9} | {'Avg LLM Calls':>13} | {'Avg Retries':>11}")
    print(f"  {'─'*4}-+-{'─'*5}-+-{'─'*9}-+-{'─'*13}-+-{'─'*11}")
    for k in sorted(k_stats):
        ks    = k_stats[k]
        kn    = ks["count"]
        avg_s = ks["total_steps"]          / kn
        avg_a = ks["total_attempts"]       / kn
        avg_e = ks["total_extra_attempts"] / kn
        print(f"  {k:>4} | {kn:>5} | {avg_s:>9.2f} | {avg_a:>13.2f} | {avg_e:>11.2f}")

    print("\n" + "=" * 60)
    print("  COST SUMMARY")
    print("=" * 60)
    overhead_pct = (total_extra_attempts / total_steps) * 100
    print(f"  Answering LLM calls per query : {avg_attempts:.3f}")
    print(f"  Judge LLM call per query      : 1.000  (at minimum)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total LLM calls per query     : ~{avg_attempts + 1:.3f}")
    print()
    print(f"  Retry overhead                : {overhead_pct:.1f}% of all step executions")
    print(f"  Without retries would be      : {avg_steps:.3f} answering calls/query")
    print(f"  Retry cost added              : +{avg_extras_per_query:.3f} calls/query")
    print("=" * 60)

if __name__ == "__main__":
    main()