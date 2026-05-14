"""
step2_llm_decomposer_900.py
----------------------------
Reads sampled_900_all.json, sends each question to Llama 3.3 70B
to decompose it into sub-questions. Does NOT answer anything.
Uses ThreadPoolExecutor for parallel API calls.

Usage:
    pip install openai python-dotenv
    # Connect to UTSA VPN first, then:
    python step2_llm_decomposer_900.py

Expects a .env file in the parent folder (nlp_final_project/.env):
    UTSA_API_KEY=gpustack...
    UTSA_BASE_URL=http://10.246.100.230/v1
    UTSA_MODEL=llama-3.3-70b-instruct-awq

Input:  sampled_900_all.json
Output: decomposed_900_all.json

Output schema per entry:
{
  "id":                "2hop__...",
  "hop_type":          "2hop",
  "question":          "the original question",
  "subquestion_count": 2,
  "subquestions":      ["sub-q 1", "sub-q 2"],
  "error":             null
}
"""

import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# ── Load .env from parent folder (nlp_final_project/.env) ─────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────
INPUT_FILE  = "sampled_900_all.json"
OUTPUT_FILE = "decomposed_900_all_prompt3.json"
MODEL       = os.getenv("UTSA_MODEL",    "llama-3.3-70b-instruct-awq")
API_BASE    = os.getenv("UTSA_BASE_URL", "http://10.246.100.230/v1")
MAX_WORKERS = 10    # parallel threads — reduce to 5 if server rate-limits
MAX_RETRIES = 2

# ── Hop type extraction ───────────────────────────────────────────────────
HOP_TYPES = ["4hop3", "4hop2", "4hop1", "3hop2", "3hop1", "2hop"]

def get_hop_type(qid: str) -> str:
    for hop in HOP_TYPES:
        if qid.startswith(hop):
            return hop
    return "unknown"

# ── Prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a question decomposition expert.

Break the given question into an ordered list of simple sub-questions that must be answered one by one to reach the final answer. Do NOT answer any of them.
It's adaptive rag so belive we have those data in paragraphs.
Rules:
- Each sub-question covers exactly one fact.
- Later sub-questions can reference a previous answer using [answer_1], [answer_2], etc.
- Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.

Step 1 — READ the full question carefully. Step 2 — IDENTIFY named entities already given (people, films, places, events). These are KNOWN. Do not ask about them. Step 3 — FIND the nested chain: questions hide a chain like "X of the Y of the Z of the W". Each "of the" usually = one unknown hop. Step 4 — COUNT the unknowns: how many separate database lookups are needed? That is your k. Step 5 — WRITE sub-questions from inside-out (resolve the deepest unknown first, work outward). Step 6 — CHECK: can any two adjacent steps be answered in a single lookup? If yes, merge them. If each needs a separate search, keep them separate. Step 7 — OUTPUT the JSON array. Do not revise or restart inside the array. 

Example:
Question: "In which country was the director of Titanic born?"
Output: ["Who directed Titanic?", "In which country was [answer_1] born?"]
Question: When was the astronomical clock in the city that Lucie Hradecká calls 
            sub-q1: Who is Lucie Hradecká?
            sub-q2: What city does [answer_1] call home?
            sub-q3: Is there an astronomical clock in [answer_2]?
            sub-q4: When was the astronomical clock in [answer_2] built?
    here sub-q1 and sub-q3 is no use **you dont need to check who is Lucie and Is there an astronomical clock**
    so in this case question will be only q2 and q4 must remember you dont need to check until that ask in question
** until its not ask in the question about Who is {any name}? dont add this question see in above example**
if thee is some name in question remember paragraph will have that context.
    """

def build_user_prompt(question: str) -> str:
    return f'Question: "{question}"\nOutput:'


# ── Parse LLM output into a clean list ───────────────────────────────────
def parse_subquestions(raw: str) -> list[str]:
    raw = raw.strip()

    # Best case: direct JSON array
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Find first [...] block in the text
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: grab anything in double quotes
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return quoted

    return []


# ── LLM call with retry ───────────────────────────────────────────────────
def call_llm(client: OpenAI, question: str) -> tuple[list[str], str | None]:
    """Returns (subquestions, error_message_or_None)"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(question)},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = response.choices[0].message.content or ""
            return parse_subquestions(raw), None

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return [], str(e)


from collections import defaultdict

# ── Expected k per hop type ───────────────────────────────────────────────
EXPECTED_K = {
    "2hop":  2,
    "3hop1": 3,
    "3hop2": 3,
    "4hop1": 4,
    "4hop2": 4,
    "4hop3": 4,
}

def print_summary(results: list[dict]) -> None:
    by_hop: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_hop[r["hop_type"]].append(r)

    hop_order = ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]

    print("\n" + "=" * 70)
    print("DECOMPOSITION SUMMARY — k distribution + composition analysis")
    print("=" * 70)

    grand_total    = 0
    grand_success  = 0
    grand_errors   = 0
    grand_exact    = 0
    grand_under    = 0
    grand_over1    = 0   # over by exactly 1
    grand_over1p   = 0   # over by more than 1

    for hop in hop_order:
        entries = by_hop.get(hop, [])
        if not entries:
            continue

        total     = len(entries)
        errors    = sum(1 for e in entries if e["error"])
        successes = total - errors
        expected  = EXPECTED_K.get(hop, None)

        k_counts: dict[int, int] = defaultdict(int)
        exact = under = over1 = over1p = 0

        for e in entries:
            if e["error"]:
                continue
            k = e["subquestion_count"]
            k_counts[k] += 1
            if expected is not None:
                if k == expected:
                    exact += 1
                elif k < expected:
                    under += 1
                elif k == expected + 1:
                    over1 += 1
                else:
                    over1p += 1

        grand_total   += total
        grand_success += successes
        grand_errors  += errors
        grand_exact   += exact
        grand_under   += under
        grand_over1   += over1
        grand_over1p  += over1p

        s = successes if successes else 1

        print(f"\n── {hop.upper()}  "
              f"(total: {total} | success: {successes} | errors: {errors} | "
              f"expected k={expected}) ──")

        for k in sorted(k_counts):
            bar = "█" * min(k_counts[k], 60)
            pct = k_counts[k] / s * 100
            if expected is not None:
                if k < expected:              tag = "  ↓ under"
                elif k == expected:           tag = "  ✓ exact"
                elif k == expected + 1:       tag = "  ↑ over by 1"
                else:                         tag = "  ↑ over by more than 1"
            else:
                tag = ""
            print(f"  k={k} : {k_counts[k]:4d} questions  {pct:5.1f}%  {bar}{tag}")

        if expected is not None and successes > 0:
            print(f"\n  Composition vs expected k={expected}:")
            print(f"    ✓ Exact            : {exact:4d}  ({exact/s*100:5.1f}%)")
            print(f"    ↓ Under            : {under:4d}  ({under/s*100:5.1f}%)")
            print(f"    ↑ Over by 1        : {over1:4d}  ({over1/s*100:5.1f}%)")
            print(f"    ↑ Over by more > 1 : {over1p:4d}  ({over1p/s*100:5.1f}%)")

    # ── Grand totals ──────────────────────────────────────────────────────
    gs = grand_success if grand_success else 1

    print(f"\n{'=' * 70}")
    print(f"OVERALL  (total: {grand_total} | success: {grand_success} | errors: {grand_errors})")
    print(f"{'=' * 70}")

    all_k: dict[int, int] = defaultdict(int)
    for r in results:
        if not r["error"]:
            all_k[r["subquestion_count"]] += 1
    for k in sorted(all_k):
        pct = all_k[k] / gs * 100
        print(f"  k={k} : {all_k[k]:4d} questions  {pct:5.1f}%")

    print(f"\n  Composition vs expected k per hop type:")
    print(f"    ✓ Exact            : {grand_exact:4d}  ({grand_exact/gs*100:5.1f}%)")
    print(f"    ↓ Under            : {grand_under:4d}  ({grand_under/gs*100:5.1f}%)")
    print(f"    ↑ Over by 1        : {grand_over1:4d}  ({grand_over1/gs*100:5.1f}%)")
    print(f"    ↑ Over by more > 1 : {grand_over1p:4d}  ({grand_over1p/gs*100:5.1f}%)")
    print("=" * 70)



# ── Main ──────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("UTSA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "UTSA_API_KEY not found.\n"
            "Add a .env file in nlp_final_project/ folder:\n"
            "  UTSA_API_KEY=gpustack...\n"
            "  UTSA_BASE_URL=http://10.246.100.230/v1\n"
            "  UTSA_MODEL=llama-3.3-70b-instruct-awq\n"
            "Also make sure you are connected to UTSA VPN."
        )

    client = OpenAI(api_key=api_key, base_url=API_BASE)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)
    print(f"Loaded   : {total} questions from {INPUT_FILE}")
    print(f"Model    : {MODEL}")
    print(f"Workers  : {MAX_WORKERS} parallel threads")
    print(f"Output   : {OUTPUT_FILE}\n")

    results  = []
    errors   = 0
    completed = 0

    # ── Worker function (runs in each thread) ─────────────────────────────
    def process_entry(entry: dict) -> dict:
        hop_type     = get_hop_type(entry["id"])
        subquestions, error = call_llm(client, entry["question"])
        return {
            "id":                entry["id"],
            "hop_type":          hop_type,
            "question":          entry["question"],
            "subquestion_count": len(subquestions),
            "subquestions":      subquestions,
            "error":             error,
        }

    # ── Run in parallel ───────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entry = {
            executor.submit(process_entry, entry): entry
            for entry in questions
        }

        for future in as_completed(future_to_entry):
            result = future.result()
            results.append(result)
            completed += 1

            hop_type = result["hop_type"]
            if result["error"]:
                print(f"[{completed:4d}/{total}] [{hop_type:6s}] ERROR: {result['error'][:70]}")
                errors += 1
            else:
                print(f"[{completed:4d}/{total}] [{hop_type:6s}] k={result['subquestion_count']}  |  {result['question'][:60]}")
                for j, sq in enumerate(result["subquestions"], 1):
                    print(f"              sub-q{j}: {sq}")
            print()

    # ── Restore original order (as_completed returns in completion order) ──
    original_order = {entry["id"]: i for i, entry in enumerate(questions)}
    results.sort(key=lambda r: original_order[r["id"]])

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {OUTPUT_FILE}")

    # ── Per-hop summary ───────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()