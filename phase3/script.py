"""
check_results.py
-----------------
Quick result checker — reads pipeline results JSON and prints
overall + per-hop metrics without re-running anything.

Usage:
    python check_results.py                          # uses default file
    python check_results.py phase3_results.json      # specify file
"""

import json
import sys
from collections import defaultdict

# ── Input file ─────────────────────────────────────────────────────────────
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "phase3_results_all.json"

def get_hop_type(qid: str) -> str:
    for hop in ["4hop3","4hop2","4hop1","3hop2","3hop1","2hop"]:
        if qid.startswith(hop):
            return hop
    return "unknown"

def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    n       = len(results)
    print(f"Total results: {n}\n")

    # Fix hop_type if missing
    for r in results:
        if not r.get("hop_type") or r["hop_type"] == "unknown":
            r["hop_type"] = get_hop_type(r["id"])

    by_hop = defaultdict(list)
    for r in results:
        by_hop[r["hop_type"]].append(r)

    # ── Overall ───────────────────────────────────────────────────────────
    total_em     = sum(r["metrics"]["em"]        for r in results)
    total_f1     = sum(r["metrics"]["f1"]        for r in results)
    total_prec   = sum(r["metrics"]["precision"] for r in results)
    total_recall = sum(r["metrics"]["recall"]    for r in results)
    total_jacc   = sum(r["judge"]["accuracy"]     for r in results)
    total_jcomp  = sum(r["judge"]["completeness"] for r in results)
    not_found    = sum(1 for r in results if "NOT FOUND" in r["final_answer"].upper())

    print("=" * 55)
    print("OVERALL METRICS")
    print("=" * 55)
    print(f"  EM              : {total_em/n:.4f}  ({total_em}/{n})")
    print(f"  F1              : {total_f1/n:.4f}")
    print(f"  Precision       : {total_prec/n:.4f}")
    print(f"  Recall          : {total_recall/n:.4f}")
    print(f"  Count           : {n}")
    print(f"  Judge Accuracy  : {total_jacc/n:.2f} / 100")
    print(f"  Judge Complete  : {total_jcomp/n:.2f} / 100")
    print(f"  Final NOT FOUND : {not_found} ({not_found/n*100:.1f}%)")

    # ── Per hop-type ──────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("PER HOP-TYPE METRICS")
    print("=" * 55)
    print(f"  {'Hop':6s} | {'N':>4} | {'EM':>6} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'J.Acc':>6}")
    print("  " + "-" * 55)

    for hop in ["2hop","3hop1","3hop2","4hop1","4hop2","4hop3"]:
        hrs = by_hop.get(hop, [])
        if not hrs: continue
        hn  = len(hrs)
        em  = sum(r["metrics"]["em"]        for r in hrs) / hn
        f1  = sum(r["metrics"]["f1"]        for r in hrs) / hn
        pr  = sum(r["metrics"]["precision"] for r in hrs) / hn
        rec = sum(r["metrics"]["recall"]    for r in hrs) / hn
        ja  = sum(r["judge"]["accuracy"]    for r in hrs) / hn
        nf  = sum(1 for r in hrs if "NOT FOUND" in r["final_answer"].upper())
        print(f"  {hop:6s} | {hn:4d} | {em:6.3f} | {f1:6.3f} | {pr:6.3f} | {rec:6.3f} | {ja:6.1f}  NF={nf}")

    # ── k breakdown ───────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("ACCURACY BY k (sub-question count)")
    print("=" * 55)
    k_data = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        k = r["k"]
        k_data[k]["total"] += 1
        if r["metrics"]["em"] == 1:
            k_data[k]["correct"] += 1
    print(f"  {'k':>3} | {'correct':>8} | {'total':>6} | {'EM%':>7} | bar")
    print(f"  {'─'*50}")
    for k in sorted(k_data):
        c   = k_data[k]["correct"]
        t   = k_data[k]["total"]
        pct = c/t*100
        bar = "█" * int(pct / 5)
        print(f"  {k:3d} | {c:8d} | {t:6d} | {pct:6.1f}% | {bar}")

    # ── F1 distribution ───────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("F1 DISTRIBUTION")
    print("=" * 55)
    for label, fn in [
        ("0.00  (no overlap)", lambda f: f == 0.0),
        ("0.01–0.29  (low)",   lambda f: 0 < f < 0.3),
        ("0.30–0.59  (part)",  lambda f: 0.3 <= f < 0.6),
        ("0.60–0.99  (close)", lambda f: 0.6 <= f < 1.0),
        ("1.00  (exact)",      lambda f: f == 1.0),
    ]:
        cnt = sum(1 for r in results if fn(r["metrics"]["f1"]))
        bar = "█" * min(cnt // 5, 40)
        print(f"  F1={label}: {cnt:4d} ({cnt/n*100:5.1f}%)  {bar}")

    print("=" * 55)

if __name__ == "__main__":
    main()