"""
generate_summary.py
--------------------
Reads pipeline_results_bge_reranker.json and prints a full summary.

Usage:
    python generate_summary.py
"""

import json
from collections import defaultdict, Counter

INPUT_FILE = "pipeline_results_bge_reranker.json"

def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    n       = len(results)

    by_hop: dict[str, list] = defaultdict(list)
    for r in results:
        by_hop[r.get("hop_type", "unknown")].append(r)

    hop_order = ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]

    print("=" * 70)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 70)

    # ── Per hop-type ──────────────────────────────────────────────────────
    print("\n── Per hop-type metrics ──\n")
    print(f"  {'Hop':6s} | {'N':>4} | {'EM':>6} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} "
          f"| {'J.Acc':>6} | {'J.Comp':>6} | {'NOT_FOUND':>10}")
    print("  " + "-" * 75)

    for hop in hop_order:
        hrs = by_hop.get(hop, [])
        if not hrs:
            continue
        hn      = len(hrs)
        em      = sum(r["metrics"]["em"]        for r in hrs) / hn
        f1      = sum(r["metrics"]["f1"]        for r in hrs) / hn
        prec    = sum(r["metrics"]["precision"] for r in hrs) / hn
        rec     = sum(r["metrics"]["recall"]    for r in hrs) / hn
        jacc    = sum(r["judge"]["accuracy"]     for r in hrs) / hn
        jcomp   = sum(r["judge"]["completeness"] for r in hrs) / hn
        nf      = sum(1 for r in hrs if "NOT FOUND" in r["final_answer"].upper())
        print(f"  {hop:6s} | {hn:4d} | {em:6.3f} | {f1:6.3f} | {prec:6.3f} | {rec:6.3f} "
              f"| {jacc:6.1f} | {jcomp:6.1f} | {nf:4d} ({nf/hn*100:.1f}%)")

    # ── Overall ───────────────────────────────────────────────────────────
    agg = data.get("aggregate_metrics", {})
    print(f"\n{'  ':6s}   {'─'*65}")
    print(f"  {'TOTAL':6s} | {n:4d} | {agg['em']:6.4f} | {agg['f1']:6.4f} | "
          f"{agg['precision']:6.4f} | {agg['recall']:6.4f} "
          f"| {agg['judge_accuracy_avg']:6.1f} | {agg['judge_completeness_avg']:6.1f} |")

    # ── F1 distribution ───────────────────────────────────────────────────
    print("\n── F1 distribution ──\n")
    f1_buckets: dict[str, int] = {}
    for r in results:
        f1 = r["metrics"]["f1"]
        if f1 == 0.0:       b = "0.00       (no overlap)"
        elif f1 < 0.3:      b = "0.01–0.29  (low)"
        elif f1 < 0.6:      b = "0.30–0.59  (partial)"
        elif f1 < 1.0:      b = "0.60–0.99  (close)"
        else:               b = "1.00       (exact)"
        f1_buckets[b] = f1_buckets.get(b, 0) + 1
    for b in sorted(f1_buckets):
        cnt = f1_buckets[b]
        bar = "█" * min(cnt // 5, 50)
        print(f"  F1={b}: {cnt:4d} ({cnt/n*100:5.1f}%)  {bar}")

    # ── Error analysis ────────────────────────────────────────────────────
    bad       = [r for r in results if r["metrics"]["em"] == 0]
    not_found = [r for r in bad if "NOT FOUND" in r["final_answer"].upper()]
    wrong     = [r for r in bad if "NOT FOUND" not in r["final_answer"].upper()
                                and r["final_answer"].strip() != ""]

    # Intermediate NOT FOUND
    inter_nf = sum(
        1 for r in results
        for s in r["steps"]
        if "NOT FOUND" in s["intermediate_answer"].upper()
    )
    total_steps = sum(len(r["steps"]) for r in results)

    print("\n── Error analysis ──\n")
    print(f"  EM=1 correct          : {len(results)-len(bad):4d} ({(len(results)-len(bad))/n*100:.1f}%)")
    print(f"  EM=0 wrong total      : {len(bad):4d} ({len(bad)/n*100:.1f}%)")
    print(f"    ↳ Final NOT FOUND   : {len(not_found):4d} ({len(not_found)/n*100:.1f}%)")
    print(f"    ↳ Wrong answer      : {len(wrong):4d} ({len(wrong)/n*100:.1f}%)")
    print(f"  Intermediate NOT FOUND: {inter_nf:4d}/{total_steps} steps ({inter_nf/total_steps*100:.1f}%)")

    # ── k distribution of correct vs wrong ───────────────────────────────
    print("\n── k distribution: correct (EM=1) vs wrong (EM=0) ──\n")
    k_correct: dict[int, int] = defaultdict(int)
    k_wrong:   dict[int, int] = defaultdict(int)
    for r in results:
        if r["metrics"]["em"] == 1:
            k_correct[r["k"]] += 1
        else:
            k_wrong[r["k"]] += 1
    all_ks = sorted(set(list(k_correct.keys()) + list(k_wrong.keys())))
    print(f"  {'k':>3} | {'correct':>8} | {'wrong':>8} | {'accuracy':>9}")
    print("  " + "-" * 38)
    for k in all_ks:
        c   = k_correct.get(k, 0)
        w   = k_wrong.get(k, 0)
        tot = c + w
        acc = c / tot * 100 if tot else 0
        print(f"  {k:3d} | {c:8d} | {w:8d} | {acc:8.1f}%")

    # ── Retrieval quality ─────────────────────────────────────────────────
    print("\n── Retrieval quality (supporting para in top-2?) ──\n")
    hit  = 0
    miss = 0
    for r in results:
        for step in r["steps"]:
            has_support = any(
                p["is_supporting"] for p in step["retrieved_paragraphs"]
            )
            if has_support:
                hit += 1
            else:
                miss += 1
    print(f"  Steps with ≥1 supporting para retrieved : {hit:4d}/{total_steps} ({hit/total_steps*100:.1f}%)")
    print(f"  Steps with NO supporting para retrieved  : {miss:4d}/{total_steps} ({miss/total_steps*100:.1f}%)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()