"""
phase3_adaptive_rag_full.py
-----------------------------
Complete Adaptive RAG pipeline for Phase 3 — full MuSiQue dataset.

This single file combines:
  1. LLM Question Decomposer  (step2 from phase2)
  2. Adaptive RAG Pipeline    (adaptive_Rag.py from phase2)

Flow:
  Step 1 — Load musique_ans_v1.0_train.jsonl (ALL answerable questions)
  Step 2 — Slice [START_IDX : END_IDX] (batch mode, no sampling)
  Step 3 — LLM decomposes each question into sub-questions (parallel)
  Step 4 — For each question run k-loop RAG:
              BGE-M3 cosine → reranker → answer with retry + refinement
  Step 5 — Judge LLM evaluates full reasoning trace
  Step 6 — Compute EM, F1, Precision, Recall, Count
  Step 7 — Save results JSON + summary text file (named with index range)

Usage:
    # Run full dataset in slices — submit multiple SLURM jobs:
    START_IDX=0    END_IDX=2000  python adaptive_rag_pipeline.py
    START_IDX=2000 END_IDX=4000  python adaptive_rag_pipeline.py
    START_IDX=4000 END_IDX=6000  python adaptive_rag_pipeline.py

    # Or pass as CLI args:
    python adaptive_rag_pipeline.py --start 0 --end 2000

.env (parent folder Final_project/.env):
    UTSA_API_KEY=gpustack...
    UTSA_BASE_URL=http://10.246.100.230/v1
    UTSA_MODEL=llama-3.3-70b-instruct-awq
    JUDGE_API_KEY=utsa-...
    JUDGE_BASE_URL=http://10.100.1.213:8888/v1
    JUDGE_MODEL=Qwen/Qwen3.5-27B
"""

import json
import os
import re
import time
import string
import random
import threading
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Load .env from parent folder ───────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

# ── Input ──────────────────────────────────────────────────────────────────
INPUT_JSONL = Path(__file__).parent.parent / "phase1" / "musique_ans_v1.0_train.jsonl"

# ── Batch index range (overridden by CLI args or env vars) ─────────────────
# Priority: CLI args > env vars > defaults (full dataset)
# Set START_IDX=0 END_IDX=2000 in SLURM script to process first 2000 questions.
# Use None for END_IDX to run to end of dataset.
_DEFAULT_START = int(os.getenv("START_IDX", "0"))
_DEFAULT_END   = os.getenv("END_IDX", None)
_DEFAULT_END   = int(_DEFAULT_END) if _DEFAULT_END is not None else None

# ── Output ─────────────────────────────────────────────────────────────────
# Output files are named with the index range, e.g.:
#   phase3_results_0-2000.json
#   phase3_summary_0-2000.txt
OUTPUT_DIR  = Path(__file__).parent

# ── LLM — decomposer + answerer ────────────────────────────────────────────
LLAMA_API_KEY  = os.getenv("UTSA_API_KEY")
LLAMA_BASE_URL = os.getenv("UTSA_BASE_URL", "http://10.246.100.230/v1")
LLAMA_MODEL    = os.getenv("UTSA_MODEL",    "llama-3.3-70b-instruct-awq")

# ── LLM — judge ────────────────────────────────────────────────────────────
JUDGE_API_KEY  = os.getenv("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://10.100.1.213:8888/v1")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL",    "Qwen/Qwen3.5-27B")

# ── Retrieval ──────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RECALL = 20
TOP_K_PARAS  = 2

# ── Threading ──────────────────────────────────────────────────────────────
MAX_WORKERS_DECOMPOSE = 10   # parallel decomposition calls
MAX_WORKERS_PIPELINE  = 16   # parallel RAG pipeline calls
MAX_RETRIES           = 2

# Thread-safe print
_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data loading and sampling
# ══════════════════════════════════════════════════════════════════════════

HOP_TYPES = ["4hop3", "4hop2", "4hop1", "3hop2", "3hop1", "2hop"]

def get_hop_type(qid: str) -> str:
    for hop in HOP_TYPES:
        if qid.startswith(hop):
            return hop
    return "unknown"


def load_and_slice(jsonl_path: str, start_idx: int, end_idx: int | None) -> list[dict]:
    """
    Load ALL answerable questions from MuSiQue JSONL,
    then slice [start_idx : end_idx].  No random sampling.
    Each entry gets a 'hop_type' field injected.
    """
    print(f"Loading {jsonl_path}...")
    all_answerable: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry.get("answerable", True):
                continue
            entry["hop_type"] = get_hop_type(entry["id"])
            all_answerable.append(entry)

    total_answerable = len(all_answerable)
    print(f"  Total answerable questions: {total_answerable}")

    # Count by hop type before slicing
    by_hop_counts: dict[str, int] = defaultdict(int)
    for e in all_answerable:
        by_hop_counts[e["hop_type"]] += 1
    for hop in HOP_TYPES:
        if by_hop_counts[hop]:
            print(f"  {hop:6s}: {by_hop_counts[hop]} total")

    # Slice the requested batch
    sliced = all_answerable[start_idx : end_idx]
    actual_end = end_idx if end_idx is not None else total_answerable
    print(f"\n  Batch slice [{start_idx} : {actual_end}]  →  {len(sliced)} questions")

    # Per-hop breakdown for this batch
    batch_by_hop: dict[str, int] = defaultdict(int)
    for e in sliced:
        batch_by_hop[e["hop_type"]] += 1
    for hop in HOP_TYPES:
        if batch_by_hop[hop]:
            print(f"  {hop:6s}: {batch_by_hop[hop]} in this batch")

    print(f"\nTotal in batch: {len(sliced)} questions\n")
    return sliced


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()

def compute_f1_precision_recall(prediction: str, ground_truths: list[str]) -> dict:
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    pred_tokens = Counter(get_tokens(prediction))
    for gt in ground_truths:
        gt_tokens = Counter(get_tokens(gt))
        common    = pred_tokens & gt_tokens
        num_same  = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / sum(pred_tokens.values())
        recall    = num_same / sum(gt_tokens.values())
        f1        = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": precision, "recall": recall}
    return best

def compute_em(prediction: str, ground_truths: list[str]) -> int:
    pred_norm = normalize_answer(prediction)
    return int(any(pred_norm == normalize_answer(gt) for gt in ground_truths))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLM decomposer
# ══════════════════════════════════════════════════════════════════════════

DECOMPOSE_SYSTEM = """You are a question decomposition expert for a multi-hop question answering system. Your task: decompose a complex question into the exact number of sub-questions needed — one per unknown fact that must be retrieved in sequence. Break the question into exactly the sub-questions needed — one per unknown fact. ════════════════════════════════════════ HOW TO THINK (chain-of-thought process) ════════════════════════════════════════ Step 1 — READ the full question carefully. Step 2 — IDENTIFY named entities already given (people, films, places, events). These are KNOWN. Do not ask about them. Step 3 — FIND the nested chain: questions hide a chain like "X of the Y of the Z of the W". Each "of the" usually = one unknown hop. Step 4 — COUNT the unknowns: how many separate database lookups are needed? That is your k. Step 5 — WRITE sub-questions from inside-out (resolve the deepest unknown first, work outward). Step 6 — CHECK: can any two adjacent steps be answered in a single lookup? If yes, merge them. If each needs a separate search, keep them separate. Step 7 — OUTPUT the JSON array. Do not revise or restart inside the array. ════════════════════════════════════════ STRICT RULES ════════════════════════════════════════ 1. One sub-question = one unknown fact = one retrieval step. No more, no less. 2. Named entities stated in the question are KNOWN — never ask who/what they are. 3. Descriptions like "the court that does X" or "the body with power to Y" are NOT named entities — they must be resolved first. 4. Do NOT add verification steps ("Is there X?", "Is [answer_1] same as [answer_2]?"). 5. Do NOT add sub-questions for facts already given in the question. 6. For 4-hop questions: all 4 layers MUST be separate sub-questions — middle steps cannot be merged even if they seem connected. 7. Later sub-questions MUST reference earlier answers as [answer_1], [answer_2], etc. 8. Output the array in ONE attempt. Do NOT revise, restart, or self-correct inside the array. 9. Return ONLY a JSON array of strings. No explanation, no markdown, no extra text. 10. Maximum 6 items. ════════════════════════════════════════ 2-HOP EXAMPLES ════════════════════════════════════════ Q: "In which country was the director of Titanic born?" Thinking: - Named entity given: Titanic ✓ - Chain: director of Titanic → birth country of director - Unknown 1: who directed Titanic - Unknown 2: where that person was born - k = 2 Output: ["Who directed Titanic?", "In which country was [answer_1] born?"] Q: "When was the astronomical clock built in the city that Lucie Hradecká calls home?" Thinking: - Named entity given: Lucie Hradecká ✓ (do not ask who she is) - Chain: home city of Lucie → clock built date in that city - Unknown 1: what city she lives in - Unknown 2: when the astronomical clock there was built - k = 2 Output: ["What city does Lucie Hradecká call home?", "When was the astronomical clock in [answer_1] built?"] Q: "What is the record label of the singer who performed the theme song of Titanic?" Thinking: - Named entity given: Titanic ✓ - Chain: theme song singer of Titanic → their record label - Unknown 1: who sang the Titanic theme - Unknown 2: what label that singer is on - k = 2 Output: ["Who performed the theme song of Titanic?", "What is the record label of [answer_1]?"] Q: "What is the population of the capital city of the country where FIFA was founded?" Thinking: - Named entity given: FIFA ✓ - Chain: country where FIFA founded → capital of that country → population of capital - Wait — that is 3 unknowns, not 2. Re-examine: "capital city of the country where FIFA was founded" — country + capital are two steps, then population is third - k = 3 (this is actually a 3-hop) Output: ["In which country was FIFA founded?", "What is the capital city of [answer_1]?", "What is the population of [answer_2]?"] ════════════════════════════════════════ 3-HOP EXAMPLES ════════════════════════════════════════ Q: "What is the nationality of the director of the film that won the Academy Award for Best Picture in 2020?" Thinking: - No named entity for the film (we don't know which film won) - Chain: Best Picture 2020 winner → director of that film → nationality of director - Unknown 1: which film won Best Picture in 2020 - Unknown 2: who directed that film - Unknown 3: what nationality that director holds - k = 3 Output: ["Which film won the Academy Award for Best Picture in 2020?", "Who directed [answer_1]?", "What is the nationality of [answer_2]?"] Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?" Thinking: - No named entity — "court of criminal appeals" is a description, must be resolved - Chain: identify court → identify body with removal power → when that body starts - Unknown 1: what IS the court of criminal appeals (resolve the description) - Unknown 2: what body can remove its justices - Unknown 3: when that body begins its work - k = 3 Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"] Q: "What is the sport played in the stadium located in the city where the winner of the 2019 Ballon d'Or was born?" Thinking: - Named entity given: Ballon d'Or 2019 ✓ but winner is unknown - Chain: winner of 2019 Ballon d'Or → birth city of winner → stadium in that city → sport played there - Wait — that is 4 steps. Re-examine: "stadium in the city where winner was born" — we need (1) winner, (2) birth city, (3) sport in that city's stadium - Can steps 2+3 merge? No — birth city and stadium sport are separate lookups - k = 3 Output: ["Who won the 2019 Ballon d'Or?", "In which city was [answer_1] born?", "What sport is played in the main stadium of [answer_2]?"] Q: "What is the official language of the country where the headquarters of the company that produces iPhone is located?" Thinking: - Named entity given: iPhone ✓ - Chain: company that makes iPhone → HQ country of that company → official language of that country - Unknown 1: what company produces iPhone - Unknown 2: where that company is headquartered - Unknown 3: official language of that country - Can 1+2 merge? Apple is well known but HQ country still needs retrieval — keep separate - k = 3 Output: ["What company produces the iPhone?", "In which country is [answer_1] headquartered?", "What is the official language of [answer_2]?"] ════════════════════════════════════════ 4-HOP EXAMPLES ════════════════════════════════════════ Q: "What is the size in square miles of the nation that provided the most legal immigrants to the city where the TV show Gotham is filmed?" Thinking: - Named entity given: Gotham (TV show) ✓ - Chain: filming city of Gotham → top immigrant-source nation for that city → size of that nation - Unknown 1: where is Gotham filmed - Unknown 2: what nation sent most legal immigrants to [answer_1] - Unknown 3: size of [answer_2] in square miles - Wait — that is only 3 unknowns. This is a 3-hop not 4-hop. - If the question also asks to verify it matches where The Crimson Pirate is set, that adds a 4th unknown: where The Crimson Pirate is set - k = 4 only if there is a 4th constraint. Count carefully. Output (3-hop version): ["Where is the TV show Gotham filmed?", "What nation provided the most legal immigrants to [answer_1]?", "What is the size of [answer_2] in square miles?"] Q: "What is the birth country of the 2018 Super Bowl halftime performer who released a live album recorded in the city that The Times added to its masthead in 2012?" Thinking: - Named entity given: 2018 Super Bowl, The Times ✓ - Chain: halftime performer → city The Times added → live album recorded in that city → birth country of performer - Unknown 1: who performed the 2018 Super Bowl halftime show - Unknown 2: what city did The Times add to masthead in 2012 - Unknown 3: what live album did [answer_1] record in [answer_2] - Unknown 4: what is the birth country of [answer_1] - Can any merge? No — each requires a separate lookup - Do NOT add "Is [answer_3] recorded in [answer_2]?" — no verification steps - k = 4 Output: ["Who performed at the 2018 Super Bowl halftime show?", "What city did The Times add to its masthead in 2012?", "What live album did [answer_1] record in [answer_2]?", "What is the birth country of [answer_1]?"] Q: "What is the capital of the country where the headquarters of the company founded by the author of Harry Potter is located?" Thinking: - Named entity given: Harry Potter ✓ - Chain: author of Harry Potter → company founded by author → HQ country of company → capital of that country - Unknown 1: who authored Harry Potter - Unknown 2: what company did [answer_1] found - Unknown 3: in which country is [answer_2] headquartered - Unknown 4: what is the capital of [answer_3] - All 4 are separate lookups — cannot merge any - k = 4 Output: ["Who authored Harry Potter?", "What company did [answer_1] found?", "In which country is [answer_2] headquartered?", "What is the capital of [answer_3]?"] Q: "What is the population of the city where the university attended by the winner of the Nobel Prize won by the author of the book 'The Road' is located?" Thinking: - Named entity given: The Road (book), Nobel Prize ✓ — but which prize and who won are unknown - Chain: author of The Road → Nobel Prize won by author → university attended by [winner] → city of university → population of city - Unknown 1: who authored The Road - Unknown 2: which Nobel Prize did [answer_1] win - Unknown 3: which university did [answer_1] attend - Unknown 4: what city is [answer_3] located in - population of that city can merge with step 4? No — two separate facts - Actually population = 5th step. But max is 6, so keep separate - k = 5 but let us recount: author → prize → university → city → population = 5 steps Output: ["Who authored 'The Road'?", "Which Nobel Prize did [answer_1] win?", "Which university did [answer_1] attend?", "In which city is [answer_3] located?", "What is the population of [answer_4]?"] PATTERN 1 — Linear chain "X of Y of Z": Q: "In which country was the director of Titanic born?" Output: ["Who directed Titanic?", "In which country was [answer_1] born?"] Q: "What is the nationality of the director of the film that won Best Picture in 2020?" Output: ["Which film won Best Picture in 2020?", "Who directed [answer_1]?", "What is the nationality of [answer_2]?"] Q: "What is the capital of the country where the headquarters of the company founded by the author of Harry Potter is located?" Output: ["Who authored Harry Potter?", "What company did [answer_1] found?", "In which country is [answer_2] headquartered?", "What is the capital of [answer_3]?"] PATTERN 2 — Two parallel lookups that together identify one entity ("X which, along with Y, did Z"): Q: "Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?" Reasoning: Need (1) who was Eisenhower's VP, (2) what country was that person from, (3) what country along with [answer_2] recognized Gaddafi early — this resolves the target country, (4) where is the lowest place in [answer_3] Output: ["Who was Eisenhower's Vice President?", "What country is [answer_1] from?", "Which country, along with [answer_2], recognized Gaddafi's government early on?", "Where is the lowest place in [answer_3]?"] PATTERN 3 — Performer/creator + separate location constraint: Q: "What is the birth country of the 2018 Super Bowl halftime performer who released a live album recorded in the city that The Times added to its masthead in 2012?" Output: ["Who performed at the 2018 Super Bowl halftime show?", "What city did The Times add to its masthead in 2012?", "What live album did [answer_1] record in [answer_2]?", "What is the birth country of [answer_1]?"] PATTERN 4 — Description that must be resolved before use: Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?" Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"]"""


def parse_subquestions(raw: str) -> list[str]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return quoted
    return []


def decompose_question(client: OpenAI, question: str) -> tuple[list[str], str | None]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM},
                    {"role": "user",   "content": f'Question: "{question}"\nOutput:'},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or ""
            return parse_subquestions(raw), None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return [], str(e)


def run_decomposition(questions: list[dict], llama: OpenAI) -> list[dict]:
    """
    Decompose all questions in parallel.
    Returns list with subquestions added to each entry.
    """
    total     = len(questions)
    results   = []
    completed = 0

    def process(entry: dict) -> dict:
        subquestions, error = decompose_question(llama, entry["question"])
        return {
            **entry,
            "subquestions":      subquestions,
            "subquestion_count": len(subquestions),
            "decompose_error":   error,
        }

    print(f"Decomposing {total} questions ({MAX_WORKERS_DECOMPOSE} workers)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DECOMPOSE) as executor:
        future_to_entry = {
            executor.submit(process, entry): entry
            for entry in questions
        }
        for future in as_completed(future_to_entry):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == total:
                tprint(f"  Decomposed {completed}/{total}")

    # Restore original order
    id_order = {e["id"]: i for i, e in enumerate(questions)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    errors = sum(1 for r in results if r["decompose_error"])
    print(f"Decomposition done. Errors: {errors}/{total}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Retrieval
# ══════════════════════════════════════════════════════════════════════════

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts, convert_to_numpy=True,
        show_progress_bar=False, normalize_embeddings=True,
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def retrieve_top_k(
    query: str,
    query_emb: np.ndarray,
    para_embs: np.ndarray,
    paragraphs: list[dict],
    reranker: CrossEncoder,
    k: int = TOP_K_PARAS,
) -> list[dict]:
    cos_scores = [cosine_similarity(query_emb, p_emb) for p_emb in para_embs]
    recall_idx = sorted(
        range(len(cos_scores)), key=lambda i: cos_scores[i], reverse=True
    )[:TOP_K_RECALL]

    candidates      = [paragraphs[i] for i in recall_idx]
    candidate_texts = [f"{p['title']}: {p['paragraph_text']}" for p in candidates]
    pairs           = [[query, text] for text in candidate_texts]
    rerank_scores   = reranker.predict(pairs).tolist()

    ranked = sorted(
        zip(recall_idx, candidates, rerank_scores),
        key=lambda x: x[2], reverse=True,
    )[:k]

    return [
        {
            "paragraph":    para,
            "rerank_score": round(rs, 4),
            "cosine_score": round(cos_scores[oi], 4),
            "cosine_rank":  recall_idx.index(oi),
        }
        for oi, para, rs in ranked
    ]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Answerer with retry + refinement
# ══════════════════════════════════════════════════════════════════════════

def llm_call(client: OpenAI, model: str, system: str, user: str,
             max_tokens: int = 150) -> tuple[str, str | None]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip(), None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return "", str(e)


ANSWER_SYSTEM = (
    "You are a precise extractive question-answering assistant. "
    "Your answer MUST be copied word-for-word from the context below. "
    "Do NOT use any outside knowledge or memory. "
    "Do NOT infer, calculate, or guess. "
    "If the answer is not explicitly stated in the context, reply with: NOT FOUND. "
    "Give only the answer — no explanation, no full sentences, no punctuation at the end."
)

ANSWER_SYSTEM_FALLBACK = (
    "You are a helpful question-answering assistant. "
    "The context below may contain a partial or nearby answer. "
    "Extract the closest relevant fact you can find — even if it is not perfectly precise. "
    "Give a short answer of a few words. "
    "Only reply NOT FOUND if the context has absolutely nothing related."
)


def answer_subquestion(llama: OpenAI, subquestion: str, context: str,
                       fallback: bool = False) -> tuple[str, str | None]:
    system = ANSWER_SYSTEM_FALLBACK if fallback else ANSWER_SYSTEM
    user   = f"Context:\n{context}\n\nQuestion: {subquestion}\n\nAnswer:"
    return llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)


def refine_question(llama: OpenAI, original_sq: str, partial_answer: str) -> str:
    system = (
        "You are a question refinement assistant. "
        "Rewrite the sub-question to be more specific, incorporating the partial answer, "
        "so that the next retrieval can find the exact fact needed. "
        "Return ONLY the rewritten question. No explanation."
    )
    user = (
        f"Original sub-question: {original_sq}\n"
        f"Partial answer found: {partial_answer}\n"
        f"Rewritten sub-question (more specific):"
    )
    refined, err = llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)
    if err or not refined.strip():
        return original_sq
    return refined.strip()


def answer_with_retry(
    llama: OpenAI,
    subquestion: str,
    context: str,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    paragraphs: list[dict],
    para_embs: np.ndarray,
) -> tuple[str, str, list[dict], int]:
    # Attempt 1: strict extraction
    ans, err = answer_subquestion(llama, subquestion, context, fallback=False)
    if err: ans = ""
    if ans.strip().upper() != "NOT FOUND" and ans.strip():
        return ans, subquestion, [], 1

    # Attempt 2: fallback prompt
    ans2, err2 = answer_subquestion(llama, subquestion, context, fallback=True)
    if err2: ans2 = ""

    if ans2.strip().upper() != "NOT FOUND" and ans2.strip():
        # Try refinement with partial answer
        refined_sq = refine_question(llama, subquestion, ans2)
        if refined_sq != subquestion:
            ref_emb       = embed_texts(embedder, [refined_sq])[0]
            refined_paras = retrieve_top_k(
                refined_sq, ref_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
            )
            refined_ctx = "\n\n".join(
                f"{p['paragraph']['title']}: {p['paragraph']['paragraph_text']}"
                for p in refined_paras
            )
            ans3, err3 = answer_subquestion(llama, refined_sq, refined_ctx, fallback=False)
            if err3: ans3 = ""
            if ans3.strip().upper() != "NOT FOUND" and ans3.strip():
                return ans3, refined_sq, refined_paras, 3
        return ans2, subquestion, [], 2

    return "NOT FOUND", subquestion, [], 2


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Judge
# ══════════════════════════════════════════════════════════════════════════

def judge_answer(
    qwen: OpenAI,
    original_question: str,
    subquestions: list[str],
    retrieved_paras: list[list[dict]],
    intermediate_answers: list[str],
    final_answer: str,
    ground_truth: str,
    ground_truth_aliases: list[str],
) -> dict:
    trace_lines = []
    for i, (sq, ans, paras) in enumerate(
        zip(subquestions, intermediate_answers, retrieved_paras), 1
    ):
        para_texts = "\n".join(
            f"  [{j+1}] {p['paragraph']['title']}: "
            f"{p['paragraph']['paragraph_text'][:200]}..."
            for j, p in enumerate(paras)
        )
        trace_lines.append(
            f"Step {i}:\n"
            f"  Sub-question: {sq}\n"
            f"  Retrieved:\n{para_texts}\n"
            f"  Answer: {ans}"
        )

    system = (
        "You are an expert judge evaluating a multi-step QA system. "
        "Respond ONLY with a JSON object. No markdown."
    )
    user = f"""Original Question: {original_question}

Reasoning Trace:
{chr(10).join(trace_lines)}

Final Answer: {final_answer}
Ground Truth: {ground_truth}
Aliases: {", ".join(ground_truth_aliases) if ground_truth_aliases else "none"}

Return ONLY this JSON:
{{
  "accuracy": <integer 0-100>,
  "completeness": <integer 0-100>,
  "reasoning": "<one sentence>"
}}"""

    raw, error = llm_call(qwen, JUDGE_MODEL, system, user, max_tokens=300)
    if error:
        return {"accuracy": 0, "completeness": 0, "reasoning": f"error: {error}"}

    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    clean = re.sub(r"```json|```", "", clean).strip()

    for pattern in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'\{[^{}]+\}', t, re.DOTALL).group()),
    ]:
        try:
            parsed = pattern(clean)
            return {
                "accuracy":     int(parsed.get("accuracy",     0)),
                "completeness": int(parsed.get("completeness", 0)),
                "reasoning":    str(parsed.get("reasoning",    "")),
            }
        except Exception:
            pass

    acc  = re.search(r'"?accuracy"?\s*:\s*(\d+)',     clean)
    comp = re.search(r'"?completeness"?\s*:\s*(\d+)', clean)
    reas = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', clean)
    return {
        "accuracy":     int(acc.group(1))  if acc  else 0,
        "completeness": int(comp.group(1)) if comp else 0,
        "reasoning":    reas.group(1)      if reas else clean[:150],
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Placeholder resolution
# ══════════════════════════════════════════════════════════════════════════

def resolve_placeholders(subquestion: str, answers_so_far: list[str]) -> str:
    for i, ans in enumerate(answers_so_far, 1):
        subquestion = subquestion.replace(f"[answer_{i}]", ans)
    return subquestion


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Per-question RAG worker
# ══════════════════════════════════════════════════════════════════════════

def process_question(
    idx: int,
    total: int,
    entry: dict,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    llama: OpenAI,
    qwen: OpenAI,
) -> dict | None:
    qid          = entry["id"]
    question     = entry["question"]
    subquestions = entry.get("subquestions", [])
    k            = entry.get("subquestion_count", 0)
    hop_type     = entry.get("hop_type", "unknown")
    paragraphs   = entry["paragraphs"]
    gold_answer  = entry["answer"]
    gold_aliases = entry.get("answer_aliases", [])
    all_gold     = [gold_answer] + gold_aliases

    if not subquestions:
        tprint(f"[{idx:4d}/{total}] SKIP {qid} — no subquestions")
        return None

    tprint(f"[{idx:4d}/{total}] [{hop_type}] k={k}  {question[:60]}")

    # Embed all paragraphs once
    para_texts = [f"{p['title']}: {p['paragraph_text']}" for p in paragraphs]
    para_embs  = embed_texts(embedder, para_texts)

    # Sequential k-loop
    intermediate_answers = []
    retrieved_per_step   = []
    step_meta            = []

    for step, raw_sq in enumerate(subquestions):
        resolved_sq = resolve_placeholders(raw_sq, intermediate_answers)

        sq_emb    = embed_texts(embedder, [resolved_sq])[0]
        top_paras = retrieve_top_k(
            resolved_sq, sq_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
        )
        context = "\n\n".join(
            f"{p['paragraph']['title']}: {p['paragraph']['paragraph_text']}"
            for p in top_paras
        )

        ans, used_sq, refined_paras, attempts = answer_with_retry(
            llama, resolved_sq, context,
            embedder, reranker, paragraphs, para_embs,
        )

        final_paras = refined_paras if refined_paras else top_paras
        tprint(f"         [{hop_type}] step {step+1} (att={attempts}): {used_sq[:50]}")
        tprint(f"                  → {ans[:70]}")

        intermediate_answers.append(ans)
        retrieved_per_step.append(final_paras)
        step_meta.append({
            "attempts":    attempts,
            "refined":     used_sq != resolved_sq,
            "original_sq": resolved_sq,
            "used_sq":     used_sq,
        })

    final_answer = intermediate_answers[-1] if intermediate_answers else ""

    em_score  = compute_em(final_answer, all_gold)
    pr_scores = compute_f1_precision_recall(final_answer, all_gold)

    tprint(f"         [{hop_type}] final='{final_answer[:50]}'  "
           f"gold='{gold_answer[:40]}'  EM={em_score}  F1={pr_scores['f1']:.3f}")

    judge_scores = judge_answer(
        qwen, question, subquestions, retrieved_per_step,
        intermediate_answers, final_answer, gold_answer, gold_aliases,
    )
    tprint(f"         [{hop_type}] judge acc={judge_scores['accuracy']} "
           f"comp={judge_scores['completeness']}")

    return {
        "id":           qid,
        "hop_type":     hop_type,
        "question":     question,
        "subquestions": subquestions,
        "k":            k,

        "steps": [
            {
                "step":              i + 1,
                "subquestion":       subquestions[i],
                "resolved_sq":       step_meta[i]["original_sq"],
                "used_sq":           step_meta[i]["used_sq"],
                "refined":           step_meta[i]["refined"],
                "attempts":          step_meta[i]["attempts"],
                "retrieved_paragraphs": [
                    {
                        "title":         p["paragraph"]["title"],
                        "text":          p["paragraph"]["paragraph_text"][:300],
                        "is_supporting": p["paragraph"]["is_supporting"],
                        "rerank_score":  p["rerank_score"],
                        "cosine_score":  p["cosine_score"],
                    }
                    for p in retrieved_per_step[i]
                ],
                "intermediate_answer": intermediate_answers[i],
            }
            for i in range(len(subquestions))
        ],

        "final_answer": final_answer,
        "gold_answer":  gold_answer,
        "gold_aliases": gold_aliases,

        "metrics": {
            "em":        em_score,
            "f1":        round(pr_scores["f1"],        4),
            "precision": round(pr_scores["precision"], 4),
            "recall":    round(pr_scores["recall"],    4),
            "count":     1,
        },

        "judge": {
            "accuracy":     judge_scores["accuracy"],
            "completeness": judge_scores["completeness"],
            "reasoning":    judge_scores["reasoning"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — Summary builder
# ══════════════════════════════════════════════════════════════════════════

def build_summary(results: list[dict], skipped: int, total: int,
                  start_idx: int, end_idx: int | None) -> str:
    """Build a detailed summary report. Hop types are always validated/repaired."""
    n     = len(results)
    lines = []
    def add(s=""): lines.append(s)

    # ── Hop-type repair (checkscript logic) ──────────────────────────────
    for r in results:
        if not r.get("hop_type") or r["hop_type"] == "unknown":
            r["hop_type"] = get_hop_type(r["id"])

    by_hop: dict[str, list] = defaultdict(list)
    for r in results:
        by_hop[r["hop_type"]].append(r)

    batch_label = f"{start_idx}-{end_idx if end_idx is not None else 'end'}"

    add("=" * 70)
    add("PHASE 3 — ADAPTIVE RAG FULL PIPELINE RESULTS")
    add(f"Batch: [{batch_label}]  |  Processed: {n}  |  Skipped: {skipped}")
    add("=" * 70)
    add()

    # ── Per-hop table ─────────────────────────────────────────────────────
    add(f"  {'Hop':6s} | {'N':>4} | {'EM':>6} | {'F1':>6} | "
        f"{'Prec':>6} | {'Recall':>6} | {'J.Acc':>6} | {'J.Comp':>6} | {'NOT_FOUND':>10}")
    add("  " + "-" * 78)

    total_em = total_f1 = total_prec = total_recall = 0.0
    total_jacc = total_jcomp = 0.0

    for hop in ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]:
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
        total_em    += em*hn;   total_f1    += f1*hn
        total_prec  += prec*hn; total_recall += rec*hn
        total_jacc  += jacc*hn; total_jcomp  += jcomp*hn
        add(f"  {hop:6s} | {hn:4d} | {em:6.3f} | {f1:6.3f} | "
            f"{prec:6.3f} | {rec:6.3f} | {jacc:6.1f} | {jcomp:6.1f} | "
            f"{nf:4d} ({nf/hn*100:.1f}%)")

    add(f"  {'─'*78}")
    add(f"  {'TOTAL':6s} | {n:4d} | {total_em/n:6.3f} | {total_f1/n:6.3f} | "
        f"{total_prec/n:6.3f} | {total_recall/n:6.3f} | "
        f"{total_jacc/n:6.1f} | {total_jcomp/n:6.1f} |")

    # ── Per-hop detailed breakdown ────────────────────────────────────────
    add(); add("── Per-Hop Detailed Breakdown ──"); add()
    for hop in ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]:
        hrs = by_hop.get(hop, [])
        if not hrs: continue
        hn        = len(hrs)
        em_sum    = sum(r["metrics"]["em"] for r in hrs)
        f1_avg    = sum(r["metrics"]["f1"] for r in hrs) / hn
        nf        = sum(1 for r in hrs if "NOT FOUND" in r["final_answer"].upper())
        wrong     = sum(1 for r in hrs if r["metrics"]["em"] == 0
                        and "NOT FOUND" not in r["final_answer"].upper()
                        and r["final_answer"].strip())
        steps_all = [s for r in hrs for s in r["steps"]]
        ns        = len(steps_all)
        hit       = sum(1 for s in steps_all
                        if any(p["is_supporting"] for p in s["retrieved_paragraphs"]))
        inter_nf  = sum(1 for s in steps_all
                        if "NOT FOUND" in s["intermediate_answer"].upper())
        add(f"  [{hop}]  n={hn}  EM={em_sum/hn:.3f}  F1={f1_avg:.3f}  "
            f"correct={em_sum}  NOT_FOUND={nf}  wrong={wrong}")
        add(f"    Steps={ns}  supp_hit={hit}/{ns} ({hit/ns*100:.1f}%)  "
            f"inter_NF={inter_nf}/{ns} ({inter_nf/ns*100:.1f}%)" if ns else "    (no steps)")
        add()

    # ── F1 distribution ───────────────────────────────────────────────────
    add("── F1 distribution ──"); add()
    for label, fn in [
        ("0.00        (none)",    lambda f: f == 0.0),
        ("0.01–0.29   (low)",     lambda f: 0 < f < 0.3),
        ("0.30–0.59   (partial)", lambda f: 0.3 <= f < 0.6),
        ("0.60–0.99   (close)",   lambda f: 0.6 <= f < 1.0),
        ("1.00        (exact)",   lambda f: f == 1.0),
    ]:
        cnt = sum(1 for r in results if fn(r["metrics"]["f1"]))
        bar = "█" * min(cnt // 5, 50)
        add(f"  F1={label}: {cnt:4d} ({cnt/n*100:5.1f}%)  {bar}")

    # ── Error analysis ────────────────────────────────────────────────────
    add(); add("── Error analysis ──"); add()
    bad         = [r for r in results if r["metrics"]["em"] == 0]
    nf_final    = [r for r in bad if "NOT FOUND" in r["final_answer"].upper()]
    wrong       = [r for r in bad if "NOT FOUND" not in r["final_answer"].upper()
                                  and r["final_answer"].strip()]
    total_steps = sum(len(r["steps"]) for r in results)
    inter_nf    = sum(
        1 for r in results for s in r["steps"]
        if "NOT FOUND" in s["intermediate_answer"].upper()
    )
    refined_steps = sum(
        1 for r in results for s in r["steps"] if s.get("refined", False)
    )
    multi_attempt = sum(
        1 for r in results for s in r["steps"] if s.get("attempts", 1) > 1
    )
    add(f"  EM=1 correct             : {n-len(bad):4d} ({(n-len(bad))/n*100:.1f}%)")
    add(f"  EM=0 wrong               : {len(bad):4d} ({len(bad)/n*100:.1f}%)")
    add(f"    ↳ Final NOT FOUND      : {len(nf_final):4d} ({len(nf_final)/n*100:.1f}%)")
    add(f"    ↳ Wrong answer         : {len(wrong):4d} ({len(wrong)/n*100:.1f}%)")
    add(f"  Intermediate NOT FOUND   : {inter_nf:4d}/{total_steps} ({inter_nf/total_steps*100:.1f}%)")
    add(f"  Steps with retry (>1 att): {multi_attempt:4d}/{total_steps} ({multi_attempt/total_steps*100:.1f}%)")
    add(f"  Steps with refinement    : {refined_steps:4d}/{total_steps} ({refined_steps/total_steps*100:.1f}%)")

    # ── Retrieval quality ─────────────────────────────────────────────────
    add(); add("── Retrieval quality ──"); add()
    hit = sum(
        1 for r in results for s in r["steps"]
        if any(p["is_supporting"] for p in s["retrieved_paragraphs"])
    )
    add(f"  Steps with ≥1 supporting para : {hit:4d}/{total_steps} ({hit/total_steps*100:.1f}%)")
    add(f"  Steps with NO supporting para  : {total_steps-hit:4d}/{total_steps} ({(total_steps-hit)/total_steps*100:.1f}%)")

    # ── Accuracy by k ─────────────────────────────────────────────────────
    add(); add("── Accuracy by k ──"); add()
    k_correct: dict[int,int] = defaultdict(int)
    k_total:   dict[int,int] = defaultdict(int)
    for r in results:
        k_total[r["k"]] += 1
        if r["metrics"]["em"] == 1:
            k_correct[r["k"]] += 1
    add(f"  {'k':>3} | {'correct':>8} | {'total':>6} | {'EM%':>7}")
    add(f"  {'─'*32}")
    for k in sorted(k_total):
        c = k_correct.get(k, 0)
        t = k_total[k]
        add(f"  {k:3d} | {c:8d} | {t:6d} | {c/t*100:6.1f}%")

    add()
    add(f"Batch [{batch_label}]  Processed: {n}/{total}  Skipped: {skipped}")
    add("=" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Parse CLI args (override env vars / defaults) ─────────────────────
    parser = argparse.ArgumentParser(description="Adaptive RAG Phase 3 — batch mode")
    parser.add_argument("--start", type=int, default=_DEFAULT_START,
                        help="Start index (inclusive) into the answerable dataset (default: 0)")
    parser.add_argument("--end",   type=int, default=_DEFAULT_END,
                        help="End index (exclusive). Omit or set to -1 for end of dataset.")
    args = parser.parse_args()

    start_idx = args.start
    end_idx   = args.end if (args.end is not None and args.end >= 0) else None

    batch_label = f"{start_idx}-{end_idx if end_idx is not None else 'end'}"
    OUTPUT_FILE = str(OUTPUT_DIR / f"phase3_results_{batch_label}.json")
    SUMMARY_FILE = str(OUTPUT_DIR / f"phase3_summary_{batch_label}.txt")

    print(f"Batch: [{batch_label}]")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}\n")

    missing = []
    if not LLAMA_API_KEY: missing.append("UTSA_API_KEY")
    if not JUDGE_API_KEY: missing.append("JUDGE_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing in .env: {', '.join(missing)}")

    llama = OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)
    qwen  = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    print("Loading BGE-M3 embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"  ready: {EMBED_MODEL}")
    print("Loading reranker model...")
    reranker = CrossEncoder(RERANK_MODEL)
    print(f"  ready: {RERANK_MODEL}\n")

    # ── Step 1: Load and slice ────────────────────────────────────────────
    questions = load_and_slice(INPUT_JSONL, start_idx, end_idx)
    total     = len(questions)

    print(f"Questions : {total}")
    print(f"Answerer  : {LLAMA_MODEL}")
    print(f"Judge     : {JUDGE_MODEL}\n")

    # ── Step 2: Decompose all questions ───────────────────────────────────
    questions = run_decomposition(questions, llama)

    # ── Step 3: Run RAG pipeline in parallel ──────────────────────────────
    results   = []
    skipped   = 0
    completed = 0

    print(f"Running RAG pipeline ({MAX_WORKERS_PIPELINE} workers)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_PIPELINE) as executor:
        future_to_idx = {
            executor.submit(
                process_question,
                i + 1, total, entry,
                embedder, reranker, llama, qwen
            ): i
            for i, entry in enumerate(questions)
        }

        for future in as_completed(future_to_idx):
            result = future.result()
            completed += 1
            if result is None:
                skipped += 1
            else:
                results.append(result)
            if completed % 50 == 0 or completed == total:
                tprint(f"  ── pipeline completed {completed}/{total} ──")

    # Restore original order
    id_order = {e["id"]: i for i, e in enumerate(questions)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    # ── Step 4: Repair hop types + aggregate metrics ──────────────────────
    for r in results:
        if not r.get("hop_type") or r["hop_type"] == "unknown":
            r["hop_type"] = get_hop_type(r["id"])

    by_hop: dict[str, list] = defaultdict(list)
    for r in results:
        by_hop[r.get("hop_type", "unknown")].append(r)

    per_hop_metrics = {}
    for hop, hrs in by_hop.items():
        hn = len(hrs)
        per_hop_metrics[hop] = {
            "count":                  hn,
            "em":                     round(sum(r["metrics"]["em"]        for r in hrs)/hn, 4),
            "f1":                     round(sum(r["metrics"]["f1"]        for r in hrs)/hn, 4),
            "precision":              round(sum(r["metrics"]["precision"] for r in hrs)/hn, 4),
            "recall":                 round(sum(r["metrics"]["recall"]    for r in hrs)/hn, 4),
            "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in hrs)/hn, 2),
            "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in hrs)/hn, 2),
        }

    n = len(results)
    aggregate = {
        "batch":     batch_label,
        "start_idx": start_idx,
        "end_idx":   end_idx,
        "em":        round(sum(r["metrics"]["em"]        for r in results)/n, 4) if n else 0,
        "f1":        round(sum(r["metrics"]["f1"]        for r in results)/n, 4) if n else 0,
        "precision": round(sum(r["metrics"]["precision"] for r in results)/n, 4) if n else 0,
        "recall":    round(sum(r["metrics"]["recall"]    for r in results)/n, 4) if n else 0,
        "count":     n,
        "skipped":   skipped,
        "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in results)/n, 2) if n else 0,
        "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in results)/n, 2) if n else 0,
    }

    # ── Step 5: Save ──────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate_metrics": aggregate,
            "per_hop_metrics":   per_hop_metrics,
            "results":           results,
        }, f, indent=2, ensure_ascii=False)

    summary = build_summary(results, skipped, total, start_idx, end_idx)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\nResults → {OUTPUT_FILE}")
    print(f"Summary → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()