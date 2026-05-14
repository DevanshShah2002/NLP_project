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
OUTPUT_FILE = "decomposed_900_all.json"
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
SYSTEM_PROMPT = """You are a question decomposition expert for a multi-hop question answering system. Your task: decompose a complex question into the exact number of sub-questions needed — one per unknown fact that must be retrieved in sequence. Break the question into exactly the sub-questions needed — one per unknown fact. ════════════════════════════════════════ HOW TO THINK (chain-of-thought process) ════════════════════════════════════════ Step 1 — READ the full question carefully. Step 2 — IDENTIFY named entities already given (people, films, places, events). These are KNOWN. Do not ask about them. Step 3 — FIND the nested chain: questions hide a chain like "X of the Y of the Z of the W". Each "of the" usually = one unknown hop. Step 4 — COUNT the unknowns: how many separate database lookups are needed? That is your k. Step 5 — WRITE sub-questions from inside-out (resolve the deepest unknown first, work outward). Step 6 — CHECK: can any two adjacent steps be answered in a single lookup? If yes, merge them. If each needs a separate search, keep them separate. Step 7 — OUTPUT the JSON array. Do not revise or restart inside the array. ════════════════════════════════════════ STRICT RULES ════════════════════════════════════════ 1. One sub-question = one unknown fact = one retrieval step. No more, no less. 2. Named entities stated in the question are KNOWN — never ask who/what they are. 3. Descriptions like "the court that does X" or "the body with power to Y" are NOT named entities — they must be resolved first. 4. Do NOT add verification steps ("Is there X?", "Is [answer_1] same as [answer_2]?"). 5. Do NOT add sub-questions for facts already given in the question. 6. For 4-hop questions: all 4 layers MUST be separate sub-questions — middle steps cannot be merged even if they seem connected. 7. Later sub-questions MUST reference earlier answers as [answer_1], [answer_2], etc. 8. Output the array in ONE attempt. Do NOT revise, restart, or self-correct inside the array. 9. Return ONLY a JSON array of strings. No explanation, no markdown, no extra text. 10. Maximum 6 items. ════════════════════════════════════════ 2-HOP EXAMPLES ════════════════════════════════════════ Q: "In which country was the director of Titanic born?" Thinking: - Named entity given: Titanic ✓ - Chain: director of Titanic → birth country of director - Unknown 1: who directed Titanic - Unknown 2: where that person was born - k = 2 Output: ["Who directed Titanic?", "In which country was [answer_1] born?"] Q: "When was the astronomical clock built in the city that Lucie Hradecká calls home?" Thinking: - Named entity given: Lucie Hradecká ✓ (do not ask who she is) - Chain: home city of Lucie → clock built date in that city - Unknown 1: what city she lives in - Unknown 2: when the astronomical clock there was built - k = 2 Output: ["What city does Lucie Hradecká call home?", "When was the astronomical clock in [answer_1] built?"] Q: "What is the record label of the singer who performed the theme song of Titanic?" Thinking: - Named entity given: Titanic ✓ - Chain: theme song singer of Titanic → their record label - Unknown 1: who sang the Titanic theme - Unknown 2: what label that singer is on - k = 2 Output: ["Who performed the theme song of Titanic?", "What is the record label of [answer_1]?"] Q: "What is the population of the capital city of the country where FIFA was founded?" Thinking: - Named entity given: FIFA ✓ - Chain: country where FIFA founded → capital of that country → population of capital - Wait — that is 3 unknowns, not 2. Re-examine: "capital city of the country where FIFA was founded" — country + capital are two steps, then population is third - k = 3 (this is actually a 3-hop) Output: ["In which country was FIFA founded?", "What is the capital city of [answer_1]?", "What is the population of [answer_2]?"] ════════════════════════════════════════ 3-HOP EXAMPLES ════════════════════════════════════════ Q: "What is the nationality of the director of the film that won the Academy Award for Best Picture in 2020?" Thinking: - No named entity for the film (we don't know which film won) - Chain: Best Picture 2020 winner → director of that film → nationality of director - Unknown 1: which film won Best Picture in 2020 - Unknown 2: who directed that film - Unknown 3: what nationality that director holds - k = 3 Output: ["Which film won the Academy Award for Best Picture in 2020?", "Who directed [answer_1]?", "What is the nationality of [answer_2]?"] Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?" Thinking: - No named entity — "court of criminal appeals" is a description, must be resolved - Chain: identify court → identify body with removal power → when that body starts - Unknown 1: what IS the court of criminal appeals (resolve the description) - Unknown 2: what body can remove its justices - Unknown 3: when that body begins its work - k = 3 Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"] Q: "What is the sport played in the stadium located in the city where the winner of the 2019 Ballon d'Or was born?" Thinking: - Named entity given: Ballon d'Or 2019 ✓ but winner is unknown - Chain: winner of 2019 Ballon d'Or → birth city of winner → stadium in that city → sport played there - Wait — that is 4 steps. Re-examine: "stadium in the city where winner was born" — we need (1) winner, (2) birth city, (3) sport in that city's stadium - Can steps 2+3 merge? No — birth city and stadium sport are separate lookups - k = 3 Output: ["Who won the 2019 Ballon d'Or?", "In which city was [answer_1] born?", "What sport is played in the main stadium of [answer_2]?"] Q: "What is the official language of the country where the headquarters of the company that produces iPhone is located?" Thinking: - Named entity given: iPhone ✓ - Chain: company that makes iPhone → HQ country of that company → official language of that country - Unknown 1: what company produces iPhone - Unknown 2: where that company is headquartered - Unknown 3: official language of that country - Can 1+2 merge? Apple is well known but HQ country still needs retrieval — keep separate - k = 3 Output: ["What company produces the iPhone?", "In which country is [answer_1] headquartered?", "What is the official language of [answer_2]?"] ════════════════════════════════════════ 4-HOP EXAMPLES ════════════════════════════════════════ Q: "What is the size in square miles of the nation that provided the most legal immigrants to the city where the TV show Gotham is filmed?" Thinking: - Named entity given: Gotham (TV show) ✓ - Chain: filming city of Gotham → top immigrant-source nation for that city → size of that nation - Unknown 1: where is Gotham filmed - Unknown 2: what nation sent most legal immigrants to [answer_1] - Unknown 3: size of [answer_2] in square miles - Wait — that is only 3 unknowns. This is a 3-hop not 4-hop. - If the question also asks to verify it matches where The Crimson Pirate is set, that adds a 4th unknown: where The Crimson Pirate is set - k = 4 only if there is a 4th constraint. Count carefully. Output (3-hop version): ["Where is the TV show Gotham filmed?", "What nation provided the most legal immigrants to [answer_1]?", "What is the size of [answer_2] in square miles?"] Q: "What is the birth country of the 2018 Super Bowl halftime performer who released a live album recorded in the city that The Times added to its masthead in 2012?" Thinking: - Named entity given: 2018 Super Bowl, The Times ✓ - Chain: halftime performer → city The Times added → live album recorded in that city → birth country of performer - Unknown 1: who performed the 2018 Super Bowl halftime show - Unknown 2: what city did The Times add to masthead in 2012 - Unknown 3: what live album did [answer_1] record in [answer_2] - Unknown 4: what is the birth country of [answer_1] - Can any merge? No — each requires a separate lookup - Do NOT add "Is [answer_3] recorded in [answer_2]?" — no verification steps - k = 4 Output: ["Who performed at the 2018 Super Bowl halftime show?", "What city did The Times add to its masthead in 2012?", "What live album did [answer_1] record in [answer_2]?", "What is the birth country of [answer_1]?"] Q: "What is the capital of the country where the headquarters of the company founded by the author of Harry Potter is located?" Thinking: - Named entity given: Harry Potter ✓ - Chain: author of Harry Potter → company founded by author → HQ country of company → capital of that country - Unknown 1: who authored Harry Potter - Unknown 2: what company did [answer_1] found - Unknown 3: in which country is [answer_2] headquartered - Unknown 4: what is the capital of [answer_3] - All 4 are separate lookups — cannot merge any - k = 4 Output: ["Who authored Harry Potter?", "What company did [answer_1] found?", "In which country is [answer_2] headquartered?", "What is the capital of [answer_3]?"] Q: "What is the population of the city where the university attended by the winner of the Nobel Prize won by the author of the book 'The Road' is located?" Thinking: - Named entity given: The Road (book), Nobel Prize ✓ — but which prize and who won are unknown - Chain: author of The Road → Nobel Prize won by author → university attended by [winner] → city of university → population of city - Unknown 1: who authored The Road - Unknown 2: which Nobel Prize did [answer_1] win - Unknown 3: which university did [answer_1] attend - Unknown 4: what city is [answer_3] located in - population of that city can merge with step 4? No — two separate facts - Actually population = 5th step. But max is 6, so keep separate - k = 5 but let us recount: author → prize → university → city → population = 5 steps Output: ["Who authored 'The Road'?", "Which Nobel Prize did [answer_1] win?", "Which university did [answer_1] attend?", "In which city is [answer_3] located?", "What is the population of [answer_4]?"] PATTERN 1 — Linear chain "X of Y of Z": Q: "In which country was the director of Titanic born?" Output: ["Who directed Titanic?", "In which country was [answer_1] born?"] Q: "What is the nationality of the director of the film that won Best Picture in 2020?" Output: ["Which film won Best Picture in 2020?", "Who directed [answer_1]?", "What is the nationality of [answer_2]?"] Q: "What is the capital of the country where the headquarters of the company founded by the author of Harry Potter is located?" Output: ["Who authored Harry Potter?", "What company did [answer_1] found?", "In which country is [answer_2] headquartered?", "What is the capital of [answer_3]?"] PATTERN 2 — Two parallel lookups that together identify one entity ("X which, along with Y, did Z"): Q: "Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?" Reasoning: Need (1) who was Eisenhower's VP, (2) what country was that person from, (3) what country along with [answer_2] recognized Gaddafi early — this resolves the target country, (4) where is the lowest place in [answer_3] Output: ["Who was Eisenhower's Vice President?", "What country is [answer_1] from?", "Which country, along with [answer_2], recognized Gaddafi's government early on?", "Where is the lowest place in [answer_3]?"] Q: "When did women win voting rights in the country which gave early recognition to Gaddafi's government alongside the country whose president was Eisenhower's vice president?" Reasoning: Same pattern — resolve VP chain first, then use it to identify the target country, then answer the question about that country Output: ["Who was Eisenhower's Vice President?", "What country did [answer_1] represent?", "Which country gave early recognition to Gaddafi alongside [answer_2]?", "When did women win voting rights in [answer_3]?"] PATTERN 3 — Performer/creator + separate location constraint: Q: "What is the birth country of the 2018 Super Bowl halftime performer who released a live album recorded in the city that The Times added to its masthead in 2012?" Reasoning: Two parallel unknowns (performer AND city) that together identify the album, then birth country Output: ["Who performed at the 2018 Super Bowl halftime show?", "What city did The Times add to its masthead in 2012?", "What live album did [answer_1] record in [answer_2]?", "What is the birth country of [answer_1]?"] PATTERN 4 — Description that must be resolved before use: Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?" Reasoning: "court of criminal appeals" is a description not a name — resolve it first Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"] """

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