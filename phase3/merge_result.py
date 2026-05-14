"""
merge_results.py
────────────────
Merges all phase3_results_*.json batch files into:
  - phase3_results_all.json       (all results + re-computed aggregate & per-hop metrics)
  - phase3_summary_all.txt        (full aggregated summary, same format as per-batch summaries)

Usage (run from the phase3 directory):
    python merge_results.py

    # Or point to a specific dir:
    python merge_results.py --dir /work/qgi899/Final_project/phase3

    # Dry-run: just print what files would be merged without writing output:
    python merge_results.py --dry-run
"""

import json
import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

# ── hop type repair (same logic as pipeline) ─────────────────────────────────
HOP_TYPES = ["4hop3", "4hop2", "4hop1", "3hop2", "3hop1", "2hop"]

def get_hop_type(qid: str) -> str:
    for hop in HOP_TYPES:
        if qid.startswith(hop):
            return hop
    return "unknown"


# ── collect & sort batch files ────────────────────────────────────────────────
def collect_batch_files(directory: str):
    pattern = os.path.join(directory, "phase3_results_*.json")
    files   = glob.glob(pattern)

    # Exclude the merged output itself if re-running
    files = [f for f in files if "phase3_results_all.json" not in f]

    def sort_key(path):
        name = os.path.basename(path)
        m = re.search(r"_(\d+)-", name)
        return int(m.group(1)) if m else 0

    files.sort(key=sort_key)
    return files


# ── main merge ────────────────────────────────────────────────────────────────
def merge(directory, dry_run=False):
    files = collect_batch_files(directory)
    if not files:
        print(f"No phase3_results_*.json files found in: {directory}")
        return

    print(f"Found {len(files)} batch files:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print()

    if dry_run:
        print("Dry-run mode — exiting without writing.")
        return

    # ── load and merge all results ────────────────────────────────────────────
    all_results = []
    seen_ids = set()
    total_skipped            = 0
    batch_meta = []
    load_errors = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            load_errors.append(f"{os.path.basename(path)}: {e}")
            continue

        agg      = data.get("aggregate_metrics", {})
        results  = data.get("results", [])
        skipped  = agg.get("skipped", 0)
        total_skipped += skipped

        batch_meta.append({
            "file":      os.path.basename(path),
            "batch":     agg.get("batch", "?"),
            "count":     len(results),
            "skipped":   skipped,
        })

        dup = 0
        for r in results:
            # Repair hop_type if missing or unknown
            if not r.get("hop_type") or r["hop_type"] == "unknown":
                r["hop_type"] = get_hop_type(r["id"])

            qid = r["id"]
            if qid in seen_ids:
                dup += 1
                continue
            seen_ids.add(qid)
            all_results.append(r)

        if dup:
            print(f"  WARNING: {os.path.basename(path)} had {dup} duplicate question IDs (skipped)")

    if load_errors:
        print("\nLoad errors:")
        for e in load_errors:
            print(f"  {e}")

    print(f"\nTotal unique results merged : {len(all_results)}")
    print(f"Total skipped (from batches): {total_skipped}")

    # ── re-compute per-hop metrics from scratch ───────────────────────────────
    by_hop = defaultdict(list)
    for r in all_results:
        by_hop[r["hop_type"]].append(r)

    per_hop_metrics = {}
    for hop, hrs in by_hop.items():
        hn = len(hrs)
        per_hop_metrics[hop] = {
            "count":                  hn,
            "em":                     round(sum(r["metrics"]["em"]        for r in hrs) / hn, 4),
            "f1":                     round(sum(r["metrics"]["f1"]        for r in hrs) / hn, 4),
            "precision":              round(sum(r["metrics"]["precision"] for r in hrs) / hn, 4),
            "recall":                 round(sum(r["metrics"]["recall"]    for r in hrs) / hn, 4),
            "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in hrs) / hn, 2),
            "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in hrs) / hn, 2),
        }

    n = len(all_results)
    aggregate = {
        "batch":     "all",
        "em":        round(sum(r["metrics"]["em"]        for r in all_results) / n, 4) if n else 0,
        "f1":        round(sum(r["metrics"]["f1"]        for r in all_results) / n, 4) if n else 0,
        "precision": round(sum(r["metrics"]["precision"] for r in all_results) / n, 4) if n else 0,
        "recall":    round(sum(r["metrics"]["recall"]    for r in all_results) / n, 4) if n else 0,
        "count":     n,
        "skipped":   total_skipped,
        "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in all_results) / n, 2) if n else 0,
        "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in all_results) / n, 2) if n else 0,
    }

    # ── write merged JSON ─────────────────────────────────────────────────────
    out_json = os.path.join(directory, "phase3_results_all.json")
    print(f"\nWriting merged JSON → {out_json} ...")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate_metrics": aggregate,
            "per_hop_metrics":   per_hop_metrics,
            "batch_files":       batch_meta,
            "results":           all_results,
        }, f, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(out_json) / 1e6
    print(f"  Done — {size_mb:.1f} MB")

    # ── build and write summary ───────────────────────────────────────────────
    summary = build_summary(all_results, total_skipped, batch_meta)
    out_txt = os.path.join(directory, "phase3_summary_all.txt")
    print(f"Writing summary       → {out_txt} ...")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary)
    print("  Done\n")

    print(summary)


# ── summary builder ───────────────────────────────────────────────────────────
def build_summary(results, total_skipped, batch_meta):
    n     = len(results)
    lines = []
    def add(s=""): lines.append(s)

    # Hop-type groups
    by_hop = defaultdict(list)
    for r in results:
        by_hop[r["hop_type"]].append(r)

    add("=" * 70)
    add("PHASE 3 — ADAPTIVE RAG FULL PIPELINE RESULTS  [ALL BATCHES MERGED]")
    add(f"Total processed: {n}  |  Total skipped: {total_skipped}  |  Batches: {len(batch_meta)}")
    add("=" * 70)
    add()

    # ── batch inventory ───────────────────────────────────────────────────────
    add("── Batch inventory ──")
    add()
    add(f"  {'File':45s} | {'Count':>6} | {'Skipped':>7}")
    add("  " + "-" * 62)
    for bm in batch_meta:
        add(f"  {bm['file']:45s} | {bm['count']:6d} | {bm['skipped']:7d}")
    add("  " + "-" * 62)
    add(f"  {'TOTAL':45s} | {n:6d} | {total_skipped:7d}")
    add()

    # ── per-hop summary table ─────────────────────────────────────────────────
    add("── Per-Hop Metrics ──")
    add()
    add(f"  {'Hop':6s} | {'N':>5} | {'EM':>6} | {'F1':>6} | "
        f"{'Prec':>6} | {'Recall':>6} | {'J.Acc':>6} | {'J.Comp':>6} | {'NOT_FOUND':>14}")
    add("  " + "-" * 85)

    total_em = total_f1 = total_prec = total_recall = 0.0
    total_jacc = total_jcomp = 0.0

    hop_order = ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]
    for hop in hop_order:
        hrs = by_hop.get(hop, [])
        if not hrs: continue
        hn    = len(hrs)
        em    = sum(r["metrics"]["em"]        for r in hrs) / hn
        f1    = sum(r["metrics"]["f1"]        for r in hrs) / hn
        prec  = sum(r["metrics"]["precision"] for r in hrs) / hn
        rec   = sum(r["metrics"]["recall"]    for r in hrs) / hn
        jacc  = sum(r["judge"]["accuracy"]     for r in hrs) / hn
        jcomp = sum(r["judge"]["completeness"] for r in hrs) / hn
        nf    = sum(1 for r in hrs if "NOT FOUND" in r["final_answer"].upper())
        total_em += em*hn; total_f1 += f1*hn
        total_prec += prec*hn; total_recall += rec*hn
        total_jacc += jacc*hn; total_jcomp += jcomp*hn
        add(f"  {hop:6s} | {hn:5d} | {em:6.3f} | {f1:6.3f} | "
            f"{prec:6.3f} | {rec:6.3f} | {jacc:6.1f} | {jcomp:6.1f} | "
            f"{nf:5d} ({nf/hn*100:.1f}%)")

    add("  " + "─" * 85)
    add(f"  {'TOTAL':6s} | {n:5d} | {total_em/n:6.3f} | {total_f1/n:6.3f} | "
        f"{total_prec/n:6.3f} | {total_recall/n:6.3f} | "
        f"{total_jacc/n:6.1f} | {total_jcomp/n:6.1f} |")
    add()

    # ── per-hop detailed breakdown ────────────────────────────────────────────
    add("── Per-Hop Detailed Breakdown ──")
    add()
    for hop in hop_order:
        hrs = by_hop.get(hop, [])
        if not hrs: continue
        hn        = len(hrs)
        em_sum    = sum(r["metrics"]["em"] for r in hrs)
        f1_avg    = sum(r["metrics"]["f1"] for r in hrs) / hn
        nf        = sum(1 for r in hrs if "NOT FOUND" in r["final_answer"].upper())
        wrong     = sum(1 for r in hrs
                        if r["metrics"]["em"] == 0
                        and "NOT FOUND" not in r["final_answer"].upper()
                        and r["final_answer"].strip())
        steps_all = [s for r in hrs for s in r["steps"]]
        ns        = len(steps_all)
        hit       = sum(1 for s in steps_all
                        if any(p["is_supporting"] for p in s["retrieved_paragraphs"]))
        inter_nf  = sum(1 for s in steps_all
                        if "NOT FOUND" in s["intermediate_answer"].upper())
        refined   = sum(1 for s in steps_all if s.get("refined", False))
        retry     = sum(1 for s in steps_all if s.get("attempts", 1) > 1)

        add(f"  [{hop}]  n={hn}  EM={em_sum/hn:.3f}  F1={f1_avg:.3f}  "
            f"correct={em_sum}  NOT_FOUND={nf}  wrong={wrong}")
        if ns:
            add(f"    Steps={ns}  supp_hit={hit}/{ns} ({hit/ns*100:.1f}%)  "
                f"inter_NF={inter_nf}/{ns} ({inter_nf/ns*100:.1f}%)  "
                f"retry={retry}/{ns} ({retry/ns*100:.1f}%)  "
                f"refined={refined}/{ns} ({refined/ns*100:.1f}%)")
        else:
            add("    (no steps)")
        add()

    # ── F1 distribution ───────────────────────────────────────────────────────
    add("── F1 Distribution ──")
    add()
    buckets = [
        ("0.00        (none)",    lambda f: f == 0.0),
        ("0.01–0.29   (low)",     lambda f: 0 < f < 0.3),
        ("0.30–0.59   (partial)", lambda f: 0.3 <= f < 0.6),
        ("0.60–0.99   (close)",   lambda f: 0.6 <= f < 1.0),
        ("1.00        (exact)",   lambda f: f == 1.0),
    ]
    for label, fn in buckets:
        cnt = sum(1 for r in results if fn(r["metrics"]["f1"]))
        bar = "█" * min(cnt // max(n // 200, 1), 50)
        add(f"  F1={label}: {cnt:6d} ({cnt/n*100:5.1f}%)  {bar}")
    add()

    # ── error analysis ────────────────────────────────────────────────────────
    add("── Error Analysis ──")
    add()
    bad         = [r for r in results if r["metrics"]["em"] == 0]
    nf_final    = [r for r in bad if "NOT FOUND" in r["final_answer"].upper()]
    wrong       = [r for r in bad if "NOT FOUND" not in r["final_answer"].upper()
                                  and r["final_answer"].strip()]
    total_steps = sum(len(r["steps"]) for r in results)
    inter_nf    = sum(1 for r in results for s in r["steps"]
                      if "NOT FOUND" in s["intermediate_answer"].upper())
    refined_steps = sum(1 for r in results for s in r["steps"] if s.get("refined", False))
    multi_attempt = sum(1 for r in results for s in r["steps"] if s.get("attempts", 1) > 1)

    add(f"  EM=1 correct              : {n-len(bad):6d} ({(n-len(bad))/n*100:.1f}%)")
    add(f"  EM=0 wrong                : {len(bad):6d} ({len(bad)/n*100:.1f}%)")
    add(f"    ↳ Final NOT FOUND       : {len(nf_final):6d} ({len(nf_final)/n*100:.1f}%)")
    add(f"    ↳ Wrong answer          : {len(wrong):6d} ({len(wrong)/n*100:.1f}%)")
    add(f"  Total reasoning steps     : {total_steps:6d}")
    add(f"  Intermediate NOT FOUND    : {inter_nf:6d}/{total_steps} ({inter_nf/total_steps*100:.1f}%)")
    add(f"  Steps with retry (>1 att) : {multi_attempt:6d}/{total_steps} ({multi_attempt/total_steps*100:.1f}%)")
    add(f"  Steps with refinement     : {refined_steps:6d}/{total_steps} ({refined_steps/total_steps*100:.1f}%)")
    add()

    # ── retrieval quality ─────────────────────────────────────────────────────
    add("── Retrieval Quality ──")
    add()
    hit = sum(1 for r in results for s in r["steps"]
              if any(p["is_supporting"] for p in s["retrieved_paragraphs"]))
    add(f"  Steps with ≥1 supporting para : {hit:6d}/{total_steps} ({hit/total_steps*100:.1f}%)")
    add(f"  Steps with NO supporting para  : {total_steps-hit:6d}/{total_steps} ({(total_steps-hit)/total_steps*100:.1f}%)")
    add()

    # ── accuracy by k ─────────────────────────────────────────────────────────
    add("── Accuracy by k (RAG Steps) ──")
    add()
    k_correct = defaultdict(int)
    k_total = defaultdict(int)
    for r in results:
        k_total[r["k"]] += 1
        if r["metrics"]["em"] == 1:
            k_correct[r["k"]] += 1
    add(f"  {'k':>3} | {'correct':>8} | {'total':>7} | {'EM%':>7}")
    add("  " + "─" * 35)
    for k in sorted(k_total):
        c = k_correct.get(k, 0)
        t = k_total[k]
        add(f"  {k:3d} | {c:8d} | {t:7d} | {c/t*100:6.1f}%")
    add()

    add("=" * 70)
    add(f"Total: {n} questions  |  Skipped: {total_skipped}  |  Batches merged: {len(batch_meta)}")
    add("=" * 70)

    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all Phase 3 batch result files")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).parent),
        help="Directory containing phase3_results_*.json files (default: script directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just list files that would be merged; don't write output",
    )
    args = parser.parse_args()
    merge(args.dir, dry_run=args.dry_run)