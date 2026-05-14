"""
step2_llm_decomposer.py
------------------------
Reads sampled_100_2hop.json, sends each question to Llama 3.3 70B
to decompose it into sub-questions. Does NOT answer anything.

Usage:
    pip install openai python-dotenv
    # Connect to UTSA VPN first, then:
    python step2_llm_decomposer.py

Expects a .env file in the same folder:
    UTSA_API_KEY=gpustack...
    UTSA_BASE_URL=http://10.246.100.230/v1
    UTSA_MODEL=llama-3.3-70b-instruct-awq

Input:  sampled_100_2hop.json
Output: decomposed_100_2hop.json

Output schema per entry:
{
  "id":                "2hop__...",
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
from openai import OpenAI
from dotenv import load_dotenv

# ── Load .env from same folder as this script ─────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────
INPUT_FILE  = "sampled_100_2hop.json"
OUTPUT_FILE = "decomposed_100_2hop.json"
MODEL       = os.getenv("UTSA_MODEL",    "llama-3.3-70b-instruct-awq")
API_BASE    = os.getenv("UTSA_BASE_URL", "http://10.246.100.230/v1")
DELAY_SEC   = 0.5
MAX_RETRIES = 2

# ── Prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a question decomposition expert.

Break the given question into an ordered list of simple sub-questions that must be answered one by one to reach the final answer. Do NOT answer any of them.
It's adaptive rag so belive we have those data in paragraphs.
Rules:
- Each sub-question covers exactly one fact.
- Later sub-questions can reference a previous answer using [answer_1], [answer_2], etc.
- Return ONLY a JSON array of strings. No explanation, no markdown, no extra text.
- Minimum 1 item, maximum 5 items.

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
                temperature=0.0,   # deterministic output
                max_tokens=200,
            )
            raw = response.choices[0].message.content or ""
            return parse_subquestions(raw), None

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return [], str(e)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("UTSA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "UTSA_API_KEY not found.\n"
            "Add a .env file in the phase1/ folder:\n"
            "  UTSA_API_KEY=gpustack...\n"
            "  UTSA_BASE_URL=http://10.246.100.230/v1\n"
            "  UTSA_MODEL=llama-3.3-70b-instruct-awq\n"
            "Also make sure you are connected to UTSA VPN."
        )

    client = OpenAI(api_key=api_key, base_url=API_BASE)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")
    print(f"Model : {MODEL}")
    print(f"Output: {OUTPUT_FILE}\n")

    results = []
    errors  = 0

    for i, entry in enumerate(questions, 1):
        subquestions, error = call_llm(client, entry["question"])

        result = {
            "id":                entry["id"],
            "question":          entry["question"],
            "subquestion_count": len(subquestions),
            "subquestions":      subquestions,
            "error":             error,
        }
        results.append(result)

        # ── Console progress ──────────────────────────────────────────────
        if error:
            print(f"[{i:3d}/100] ERROR: {error[:80]}")
            errors += 1
        else:
            print(f"[{i:3d}/100] k={len(subquestions)}  |  {entry['question'][:70]}")
            for j, sq in enumerate(subquestions, 1):
                print(f"            sub-q{j}: {sq}")
        print()

        time.sleep(DELAY_SEC)

    # ── Save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Saved → {OUTPUT_FILE}")
    print(f"Success : {len(results) - errors}/100")
    print(f"Errors  : {errors}/100")

    k_counts: dict[int, int] = {}
    for r in results:
        k = r["subquestion_count"]
        k_counts[k] = k_counts.get(k, 0) + 1
    print("\nk distribution:")
    for k in sorted(k_counts):
        print(f"  k={k} : {k_counts[k]} questions")


if __name__ == "__main__":
    main()