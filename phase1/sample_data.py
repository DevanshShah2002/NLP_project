"""
step1_sample_questions.py
--------------------------
Reads musique_ans_v1_0_train.jsonl, filters 2-hop questions,
randomly samples 100, and saves them to sampled_100_2hop.json

Usage:
    python step1_sample_questions.py

Input:  musique_ans_v1_0_train.jsonl  (must be in same folder)
Output: sampled_100_2hop.json
"""

import json
import random

INPUT_FILE  = "musique_ans_v1.0_train.jsonl"
OUTPUT_FILE = "sampled_100_2hop.json"
SAMPLE_SIZE = 100
RANDOM_SEED = 42          # fixed seed → same 100 questions every run

def main():
    # ── Load all 2-hop questions ──────────────────────────────────────────
    two_hop = []
    print("hi")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["id"].startswith("2hop") and entry["answerable"]:
                two_hop.append(entry)

    print(f"Total answerable 2-hop questions found: {len(two_hop)}")

    # ── Random sample ─────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    sampled = random.sample(two_hop, SAMPLE_SIZE)

    # ── Keep only the fields we need ──────────────────────────────────────
    output = []
    for entry in sampled:
        output.append({
            "id":                    entry["id"],
            "question":              entry["question"],
            "answer":                entry["answer"],
            "answer_aliases":        entry["answer_aliases"],
            # Gold decomposition — used later to evaluate LLM decomposition quality
            "gold_decomposition":    entry["question_decomposition"],
            # Paragraphs — used as the local retrieval corpus (20 paras per question)
            "paragraphs":            entry["paragraphs"],
        })

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output)} questions → {OUTPUT_FILE}")
    print(f"\nSample entry:")
    print(f"  id       : {output[0]['id']}")
    print(f"  question : {output[0]['question']}")
    print(f"  answer   : {output[0]['answer']}")
    print(f"  gold sub-questions:")
    for sq in output[0]['gold_decomposition']:
        print(f"    - {sq['question']}  →  {sq['answer']}")
    print(f"  paragraphs: {len(output[0]['paragraphs'])} (supporting: "
          f"{sum(1 for p in output[0]['paragraphs'] if p['is_supporting'])})")

if __name__ == "__main__":
    main()