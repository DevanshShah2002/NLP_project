"""
phase3_adaptive_rag_full.py
-----------------------------
Complete Adaptive RAG pipeline for Phase 3.
Runs on TWO datasets sequentially then combines results.

Datasets (in order):
  1. TriviaQA   - single-hop, tests retrieval under noisy evidence
  2. MuSiQue    - multi-hop (2-4 hop), tests decomposition + chaining

Folder structure expected:
  Final_project/
  |-- .env
  |-- phase3/
  |   |-- phase3_adaptive_rag_full.py
  |-- data/
      |-- musique_ans_v1.0_train.jsonl
      |-- trivia_qa-train.json

Usage:
    pip install openai python-dotenv sentence-transformers numpy
    python phase3_adaptive_rag_full.py
"""

import json
import os
import re
import time
import string
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Load .env from parent folder (Final_project/.env) ──────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "phase1"

MUSIQUE_JSONL  = DATA_DIR / "musique_ans_v1.0_train.jsonl"
# TriviaQA loaded directly from HuggingFace (no local file needed)
TRIVIAQA_HF_NAME   = "mandarjoshi/trivia_qa"
TRIVIAQA_HF_CONFIG = "rc"
TRIVIAQA_HF_SPLIT  = "train"

# Sampling
TRIVIAQA_SAMPLE = 300
MUSIQUE_SAMPLE_SIZES = {
    "2hop":  300,
    "3hop1": 150,
    "3hop2": 150,
    "4hop1": 100,
    "4hop2": 100,
    "4hop3": 100,
}
RANDOM_SEED = 42

# Output files
TRIVIAQA_OUTPUT = "phase3_triviaqa_results.json"
MUSIQUE_OUTPUT  = "phase3_musique_results.json"
COMBINED_OUTPUT = "phase3_combined_results.json"
SUMMARY_FILE    = "phase3_summary.txt"

# LLM - decomposer + answerer
LLAMA_API_KEY  = os.getenv("UTSA_API_KEY")
LLAMA_BASE_URL = os.getenv("UTSA_BASE_URL", "http://10.246.100.230/v1")
LLAMA_MODEL    = os.getenv("UTSA_MODEL",    "llama-3.3-70b-instruct-awq")

# LLM - judge
JUDGE_API_KEY  = os.getenv("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://10.100.1.213:8888/v1")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL",    "Qwen/Qwen3.5-27B")

# Retrieval
EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RECALL = 20
TOP_K_PARAS  = 2

# Threading
MAX_WORKERS_DECOMPOSE = 10
MAX_WORKERS_PIPELINE  = 20
MAX_RETRIES           = 2

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 - Data loaders
# ══════════════════════════════════════════════════════════════════════════

HOP_TYPES = ["4hop3", "4hop2", "4hop1", "3hop2", "3hop1", "2hop"]

def get_hop_type(qid: str) -> str:
    for hop in HOP_TYPES:
        if qid.startswith(hop):
            return hop
    return "unknown"


def load_musique(jsonl_path: Path) -> list:
    by_hop = defaultdict(list)
    print(f"Loading MuSiQue: {jsonl_path.name}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry.get("answerable", True):
                continue
            hop = get_hop_type(entry["id"])
            if hop in MUSIQUE_SAMPLE_SIZES:
                entry["hop_type"] = hop
                entry["dataset"]  = "musique"
                by_hop[hop].append(entry)

    for hop, entries in by_hop.items():
        print(f"  {hop}: {len(entries)} found")

    random.seed(RANDOM_SEED)
    sampled = []
    for hop, size in MUSIQUE_SAMPLE_SIZES.items():
        pool   = by_hop.get(hop, [])
        n      = min(size, len(pool))
        picked = random.sample(pool, n)
        sampled.extend(picked)
        print(f"  {hop}: sampled {n}")

    print(f"  Total MuSiQue: {len(sampled)}\n")
    return sampled


def load_triviaqa() -> list:
    """
    Load TriviaQA rc split directly from HuggingFace datasets.
    Uses entity_pages wiki_context as retrieval corpus.
    Chunks each WikiContext into ~150-word paragraphs (max 20 per question).
    """
    from datasets import load_dataset

    CHUNK_WORDS = 150
    MAX_PARAS   = 20

    print("Loading TriviaQA from HuggingFace (" + TRIVIAQA_HF_NAME + " / " + TRIVIAQA_HF_CONFIG + ")...")
    ds = load_dataset(TRIVIAQA_HF_NAME, TRIVIAQA_HF_CONFIG, split=TRIVIAQA_HF_SPLIT)
    print("  Total entries: " + str(len(ds)))

    random.seed(RANDOM_SEED + 1)
    indices = random.sample(range(len(ds)), min(TRIVIAQA_SAMPLE, len(ds)))

    result = []
    for idx in indices:
        row = ds[idx]

        qid      = row.get("question_id", "tqa_" + str(idx))
        question = row.get("question", "")
        answer   = row.get("answer", {})

        gold_value   = answer.get("value",   "")
        gold_aliases = answer.get("aliases", [])

        # entity_pages is a dict of lists in HF format:
        # {"title": [...], "wiki_context": [...]}
        entity_pages = row.get("entity_pages", {})
        titles    = entity_pages.get("title",        [])
        contexts  = entity_pages.get("wiki_context", [])

        paragraphs = []
        for title, context in zip(titles, contexts):
            if not context:
                continue
            words  = context.split()
            chunks = [
                " ".join(words[i:i + CHUNK_WORDS])
                for i in range(0, len(words), CHUNK_WORDS)
                if words[i:i + CHUNK_WORDS]
            ]
            for chunk in chunks:
                paragraphs.append({
                    "title":          title,
                    "paragraph_text": chunk,
                    "is_supporting":  True,
                })
                if len(paragraphs) >= MAX_PARAS:
                    break
            if len(paragraphs) >= MAX_PARAS:
                break

        if not paragraphs:
            continue

        result.append({
            "id":             "tqa__" + str(qid),
            "question":       question,
            "answer":         gold_value,
            "answer_aliases": gold_aliases,
            "paragraphs":     paragraphs,
            "hop_type":       "1hop",
            "dataset":        "triviaqa",
        })

    print("  Sampled TriviaQA: " + str(len(result)) + "\n")
    return result


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 - Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def get_tokens(s: str) -> list:
    return normalize_answer(s).split()

def compute_f1_precision_recall(prediction: str, ground_truths: list) -> dict:
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

def compute_em(prediction: str, ground_truths: list) -> int:
    pred_norm = normalize_answer(prediction)
    return int(any(pred_norm == normalize_answer(gt) for gt in ground_truths))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 - LLM decomposer
# ══════════════════════════════════════════════════════════════════════════

DECOMPOSE_SYSTEM = """You are a question decomposition expert for a multi-hop question answering system. Your task: decompose a complex question into the exact number of sub-questions needed, one per unknown fact that must be retrieved in sequence.

STRICT RULES:
1. One sub-question = one unknown fact = one retrieval step.
2. Named entities stated in the question are KNOWN. Their attributes are UNKNOWN.
3. Descriptions like "the court that does X" are NOT named entities - resolve them first.
4. Do NOT add verification steps.
5. Do NOT add sub-questions for facts already given in the question.
6. Later sub-questions MUST reference earlier answers as [answer_1], [answer_2], etc.
7. Output the array in ONE attempt. Do NOT revise or restart inside the array.
8. Return ONLY a JSON array of strings. No explanation, no markdown.
9. Maximum 6 items.

PATTERN 1 - Linear chain:
Q: "In which country was the director of Titanic born?"
Output: ["Who directed Titanic?", "In which country was [answer_1] born?"]

PATTERN 2 - Parallel lookups:
Q: "Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?"
Output: ["Who was Eisenhower's Vice President?", "What country is [answer_1] from?", "Which country, along with [answer_2], recognized Gaddafi's government early on?", "Where is the lowest place in [answer_3]?"]

PATTERN 3 - Description that must be resolved first:
Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?"
Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"]

PATTERN 4 - 4-hop linear:
Q: "What is the capital of the country where the headquarters of the company founded by the author of Harry Potter is located?"
Output: ["Who authored Harry Potter?", "What company did [answer_1] found?", "In which country is [answer_2] headquartered?", "What is the capital of [answer_3]?"]"""


def parse_subquestions(raw: str) -> list:
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


def decompose_question(client: OpenAI, question: str) -> tuple:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM},
                    {"role": "user",   "content": 'Question: "' + question + '"\nOutput:'},
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


def run_decomposition(questions: list, llama: OpenAI) -> list:
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

    id_order = {e["id"]: i for i, e in enumerate(questions)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    errors = sum(1 for r in results if r["decompose_error"])
    print(f"Decomposition done. Errors: {errors}/{total}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 - Retrieval
# ══════════════════════════════════════════════════════════════════════════

def embed_texts(model: SentenceTransformer, texts: list) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def retrieve_top_k(query, query_emb, para_embs, paragraphs, reranker, k=TOP_K_PARAS):
    cos_scores = [cosine_similarity(query_emb, p_emb) for p_emb in para_embs]
    recall_idx = sorted(
        range(len(cos_scores)), key=lambda i: cos_scores[i], reverse=True
    )[:TOP_K_RECALL]

    candidates      = [paragraphs[i] for i in recall_idx]
    candidate_texts = [p["title"] + ": " + p["paragraph_text"] for p in candidates]
    pairs           = [[query, text] for text in candidate_texts]
    rerank_scores   = reranker.predict(pairs).tolist()

    ranked = sorted(
        zip(recall_idx, candidates, rerank_scores),
        key=lambda x: x[2],
        reverse=True,
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
# SECTION 5 - Answerer with retry + refinement
# ══════════════════════════════════════════════════════════════════════════

def llm_call(client, model, system, user, max_tokens=150):
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
    "If the answer is not explicitly stated in the context, reply: NOT FOUND. "
    "Give only the answer, no explanation, no punctuation at the end."
)

ANSWER_SYSTEM_FALLBACK = (
    "You are a helpful question-answering assistant. "
    "Extract the closest relevant fact from the context. "
    "Give a short answer of a few words. "
    "Only reply NOT FOUND if the context has absolutely nothing related."
)


def answer_subquestion(llama, subquestion, context, fallback=False):
    system = ANSWER_SYSTEM_FALLBACK if fallback else ANSWER_SYSTEM
    user   = "Context:\n" + context + "\n\nQuestion: " + subquestion + "\n\nAnswer:"
    return llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)


def refine_question(llama, original_sq, partial_answer):
    system = (
        "You are a question refinement assistant. "
        "Rewrite the sub-question to be more specific using the partial answer. "
        "Return ONLY the rewritten question. No explanation."
    )
    user = (
        "Original sub-question: " + original_sq + "\n"
        "Partial answer found: " + partial_answer + "\n"
        "Rewritten sub-question (more specific):"
    )
    refined, err = llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)
    if err or not refined.strip():
        return original_sq
    return refined.strip()


def answer_with_retry(llama, subquestion, context, embedder, reranker, paragraphs, para_embs):
    # Attempt 1: strict
    ans, err = answer_subquestion(llama, subquestion, context, fallback=False)
    if err:
        ans = ""
    if ans.strip().upper() != "NOT FOUND" and ans.strip():
        return ans, subquestion, [], 1

    # Attempt 2: fallback
    ans2, err2 = answer_subquestion(llama, subquestion, context, fallback=True)
    if err2:
        ans2 = ""

    if ans2.strip().upper() != "NOT FOUND" and ans2.strip():
        refined_sq = refine_question(llama, subquestion, ans2)
        if refined_sq != subquestion:
            ref_emb       = embed_texts(embedder, [refined_sq])[0]
            refined_paras = retrieve_top_k(
                refined_sq, ref_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
            )
            refined_ctx = "\n\n".join(
                p["paragraph"]["title"] + ": " + p["paragraph"]["paragraph_text"]
                for p in refined_paras
            )
            ans3, err3 = answer_subquestion(llama, refined_sq, refined_ctx, fallback=False)
            if err3:
                ans3 = ""
            if ans3.strip().upper() != "NOT FOUND" and ans3.strip():
                return ans3, refined_sq, refined_paras, 3
        return ans2, subquestion, [], 2

    return "NOT FOUND", subquestion, [], 2


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 - Judge
# ══════════════════════════════════════════════════════════════════════════

def judge_answer(qwen, original_question, subquestions, retrieved_paras,
                 intermediate_answers, final_answer, ground_truth, ground_truth_aliases):
    trace_lines = []
    for i, (sq, ans, paras) in enumerate(
        zip(subquestions, intermediate_answers, retrieved_paras), 1
    ):
        para_texts = "\n".join(
            "  [" + str(j+1) + "] " + p["paragraph"]["title"] + ": " +
            p["paragraph"]["paragraph_text"][:200] + "..."
            for j, p in enumerate(paras)
        )
        trace_lines.append(
            "Step " + str(i) + ":\n"
            "  Sub-question: " + sq + "\n"
            "  Retrieved:\n" + para_texts + "\n"
            "  Answer: " + ans
        )

    system = (
        "You are an expert judge evaluating a multi-step QA system. "
        "Respond ONLY with a JSON object. No markdown."
    )
    user = (
        "Original Question: " + original_question + "\n\n"
        "Reasoning Trace:\n" + "\n".join(trace_lines) + "\n\n"
        "Final Answer: " + final_answer + "\n"
        "Ground Truth: " + ground_truth + "\n"
        "Aliases: " + (", ".join(ground_truth_aliases) if ground_truth_aliases else "none") + "\n\n"
        'Return ONLY this JSON:\n'
        '{\n'
        '  "accuracy": <integer 0-100>,\n'
        '  "completeness": <integer 0-100>,\n'
        '  "reasoning": "<one sentence>"\n'
        '}'
    )

    raw, error = llm_call(qwen, JUDGE_MODEL, system, user, max_tokens=300)
    if error:
        return {"accuracy": 0, "completeness": 0, "reasoning": "error: " + error}

    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    clean = re.sub(r"```json|```", "", clean).strip()

    for pattern_fn in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'\{[^{}]+\}', t, re.DOTALL).group()),
    ]:
        try:
            parsed = pattern_fn(clean)
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
# SECTION 7 - Placeholder resolution
# ══════════════════════════════════════════════════════════════════════════

def resolve_placeholders(subquestion: str, answers_so_far: list) -> str:
    for i, ans in enumerate(answers_so_far, 1):
        subquestion = subquestion.replace("[answer_" + str(i) + "]", ans)
    return subquestion


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 - Per-question worker
# ══════════════════════════════════════════════════════════════════════════

def process_question(idx, total, entry, embedder, reranker, llama, qwen):
    qid          = entry["id"]
    question     = entry["question"]
    subquestions = entry.get("subquestions", [])
    k            = entry.get("subquestion_count", 0)
    hop_type     = entry.get("hop_type") or get_hop_type(qid)
    dataset      = entry.get("dataset", "unknown")
    paragraphs   = entry["paragraphs"]
    gold_answer  = entry["answer"]
    gold_aliases = entry.get("answer_aliases", [])
    all_gold     = [gold_answer] + gold_aliases

    if not subquestions:
        tprint("[" + str(idx) + "/" + str(total) + "] SKIP " + qid + " - no subquestions")
        return None

    tprint("[" + str(idx) + "/" + str(total) + "] [" + hop_type + "] k=" + str(k) + "  " + question[:60])

    para_texts = [p["title"] + ": " + p["paragraph_text"] for p in paragraphs]
    para_embs  = embed_texts(embedder, para_texts)

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
            p["paragraph"]["title"] + ": " + p["paragraph"]["paragraph_text"]
            for p in top_paras
        )

        ans, used_sq, refined_paras, attempts = answer_with_retry(
            llama, resolved_sq, context, embedder, reranker, paragraphs, para_embs,
        )

        final_paras = refined_paras if refined_paras else top_paras
        tprint("         [" + hop_type + "] step " + str(step+1) + " (att=" + str(attempts) + "): " + used_sq[:50])
        tprint("                  -> " + ans[:70])

        intermediate_answers.append(ans)
        retrieved_per_step.append(final_paras)
        step_meta.append({
            "attempts":    attempts,
            "refined":     used_sq != resolved_sq,
            "original_sq": resolved_sq,
            "used_sq":     used_sq,
        })

    final_answer = intermediate_answers[-1] if intermediate_answers else ""
    em_score     = compute_em(final_answer, all_gold)
    pr_scores    = compute_f1_precision_recall(final_answer, all_gold)

    tprint("         [" + hop_type + "] final='" + final_answer[:50] + "'  gold='" + gold_answer[:40] + "'  EM=" + str(em_score) + "  F1=" + str(round(pr_scores["f1"], 3)))

    judge_scores = judge_answer(
        qwen, question, subquestions, retrieved_per_step,
        intermediate_answers, final_answer, gold_answer, gold_aliases,
    )
    tprint("         [" + hop_type + "] judge acc=" + str(judge_scores["accuracy"]) + " comp=" + str(judge_scores["completeness"]))

    return {
        "id":           qid,
        "dataset":      dataset,
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
# SECTION 9 - Dataset runner
# ══════════════════════════════════════════════════════════════════════════

def run_dataset(dataset_name, questions, embedder, reranker, llama, qwen, output_file):
    total     = len(questions)
    results   = []
    skipped   = 0
    completed = 0

    print("\n" + "="*60)
    print("Running " + dataset_name + " - " + str(total) + " questions (" + str(MAX_WORKERS_PIPELINE) + " workers)")
    print("="*60)

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
                tprint("  [" + dataset_name + "] completed " + str(completed) + "/" + str(total))

    id_order = {e["id"]: i for i, e in enumerate(questions)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    by_group = defaultdict(list)
    for r in results:
        by_group[r.get("hop_type", "unknown")].append(r)

    per_group_metrics = {}
    for grp, hrs in by_group.items():
        hn = len(hrs)
        per_group_metrics[grp] = {
            "count":                  hn,
            "em":                     round(sum(r["metrics"]["em"]         for r in hrs)/hn, 4),
            "f1":                     round(sum(r["metrics"]["f1"]         for r in hrs)/hn, 4),
            "precision":              round(sum(r["metrics"]["precision"]  for r in hrs)/hn, 4),
            "recall":                 round(sum(r["metrics"]["recall"]     for r in hrs)/hn, 4),
            "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in hrs)/hn, 2),
            "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in hrs)/hn, 2),
        }

    n = len(results)
    aggregate = {
        "dataset":   dataset_name,
        "em":        round(sum(r["metrics"]["em"]         for r in results)/n, 4) if n else 0,
        "f1":        round(sum(r["metrics"]["f1"]         for r in results)/n, 4) if n else 0,
        "precision": round(sum(r["metrics"]["precision"]  for r in results)/n, 4) if n else 0,
        "recall":    round(sum(r["metrics"]["recall"]     for r in results)/n, 4) if n else 0,
        "count":     n,
        "skipped":   skipped,
        "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in results)/n, 2) if n else 0,
        "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in results)/n, 2) if n else 0,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate_metrics": aggregate,
            "per_group_metrics": per_group_metrics,
            "results":           results,
        }, f, indent=2, ensure_ascii=False)

    print("  Saved " + dataset_name + " -> " + output_file)
    print("  EM=" + str(aggregate["em"]) + "  F1=" + str(aggregate["f1"]) + "  n=" + str(n))
    return results, skipped


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 - Summary builder
# ══════════════════════════════════════════════════════════════════════════

def build_summary(results_tqa, results_msq, skipped_tqa, skipped_msq):
    lines = []

    def add(s=""):
        lines.append(s)

    def metrics_table(results, hop_order, label):
        n = len(results)
        if n == 0:
            add("  No results.")
            return
        by_group = defaultdict(list)
        for r in results:
            by_group[r.get("hop_type", "unknown")].append(r)

        add("  " + "HopType".ljust(8) + " | " + "N".rjust(4) + " | " +
            "EM".rjust(6) + " | " + "F1".rjust(6) + " | " +
            "Prec".rjust(6) + " | " + "Recall".rjust(6) + " | " +
            "J.Acc".rjust(6) + " | " + "NOT_FOUND".rjust(10))
        add("  " + "-" * 72)

        total_em = total_f1 = total_prec = total_recall = 0.0
        for grp in hop_order:
            hrs = by_group.get(grp, [])
            if not hrs:
                continue
            hn    = len(hrs)
            em    = sum(r["metrics"]["em"]        for r in hrs) / hn
            f1    = sum(r["metrics"]["f1"]        for r in hrs) / hn
            prec  = sum(r["metrics"]["precision"] for r in hrs) / hn
            rec   = sum(r["metrics"]["recall"]    for r in hrs) / hn
            jacc  = sum(r["judge"]["accuracy"]    for r in hrs) / hn
            nf    = sum(1 for r in hrs if "NOT FOUND" in r["final_answer"].upper())
            total_em    += em   * hn
            total_f1    += f1   * hn
            total_prec  += prec * hn
            total_recall += rec * hn
            add("  " + grp.ljust(8) + " | " + str(hn).rjust(4) + " | " +
                str(round(em, 3)).rjust(6) + " | " + str(round(f1, 3)).rjust(6) + " | " +
                str(round(prec, 3)).rjust(6) + " | " + str(round(rec, 3)).rjust(6) + " | " +
                str(round(jacc, 1)).rjust(6) + " | " + (str(nf) + " (" + str(round(nf/hn*100,1)) + "%)").rjust(10))

        add("  " + "-" * 72)
        add("  " + "TOTAL".ljust(8) + " | " + str(n).rjust(4) + " | " +
            str(round(total_em/n, 3)).rjust(6) + " | " + str(round(total_f1/n, 3)).rjust(6) + " | " +
            str(round(total_prec/n, 3)).rjust(6) + " | " + str(round(total_recall/n, 3)).rjust(6) + " |")

    # TriviaQA block
    add("=" * 75)
    add("DATASET 1 - TriviaQA (single-hop, retrieval stress test)")
    add("=" * 75)
    metrics_table(results_tqa, ["1hop"], "hop_type")
    add("  Skipped: " + str(skipped_tqa))

    # MuSiQue block
    add()
    add("=" * 75)
    add("DATASET 2 - MuSiQue (multi-hop: 2-4 hops)")
    add("=" * 75)
    metrics_table(results_msq, ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"], "hop_type")
    add("  Skipped: " + str(skipped_msq))

    # Combined
    all_results = results_tqa + results_msq
    n_all = len(all_results)
    add()
    add("=" * 75)
    add("COMBINED - Both Datasets (n=" + str(n_all) + ")")
    add("=" * 75)

    if n_all > 0:
        em_all    = sum(r["metrics"]["em"]         for r in all_results) / n_all
        f1_all    = sum(r["metrics"]["f1"]         for r in all_results) / n_all
        prec_all  = sum(r["metrics"]["precision"]  for r in all_results) / n_all
        rec_all   = sum(r["metrics"]["recall"]     for r in all_results) / n_all
        jacc_all  = sum(r["judge"]["accuracy"]     for r in all_results) / n_all
        jcomp_all = sum(r["judge"]["completeness"] for r in all_results) / n_all
        nf_all    = sum(1 for r in all_results if "NOT FOUND" in r["final_answer"].upper())
        bad_all   = sum(1 for r in all_results if r["metrics"]["em"] == 0)

        add("  EM              : " + str(round(em_all, 4)) + "  (" + str(int(em_all*n_all)) + "/" + str(n_all) + ")")
        add("  F1              : " + str(round(f1_all, 4)))
        add("  Precision       : " + str(round(prec_all, 4)))
        add("  Recall          : " + str(round(rec_all, 4)))
        add("  Judge Accuracy  : " + str(round(jacc_all, 2)) + " / 100")
        add("  Judge Complete  : " + str(round(jcomp_all, 2)) + " / 100")
        add("  NOT FOUND final : " + str(nf_all) + " (" + str(round(nf_all/n_all*100, 1)) + "%)")
        add("  Wrong answers   : " + str(bad_all) + " (" + str(round(bad_all/n_all*100, 1)) + "%)")

    # Error analysis
    add()
    add("-- Error analysis (combined) --")
    add()
    total_steps = sum(len(r["steps"]) for r in all_results)
    inter_nf    = sum(1 for r in all_results for s in r["steps"]
                      if "NOT FOUND" in s["intermediate_answer"].upper())
    multi_att   = sum(1 for r in all_results for s in r["steps"]
                      if s.get("attempts", 1) > 1)
    refined     = sum(1 for r in all_results for s in r["steps"]
                      if s.get("refined", False))
    hit         = sum(1 for r in all_results for s in r["steps"]
                      if any(p["is_supporting"] for p in s["retrieved_paragraphs"]))
    if total_steps:
        add("  Intermediate NOT FOUND   : " + str(inter_nf) + "/" + str(total_steps) + " (" + str(round(inter_nf/total_steps*100, 1)) + "%)")
        add("  Steps with retry         : " + str(multi_att) + "/" + str(total_steps) + " (" + str(round(multi_att/total_steps*100, 1)) + "%)")
        add("  Steps with refinement    : " + str(refined) + "/" + str(total_steps) + " (" + str(round(refined/total_steps*100, 1)) + "%)")
        add("  Retrieval hit rate       : " + str(hit) + "/" + str(total_steps) + " (" + str(round(hit/total_steps*100, 1)) + "%)")

    # k accuracy
    add()
    add("-- Accuracy by k (combined) --")
    add()
    k_correct = defaultdict(int)
    k_total   = defaultdict(int)
    for r in all_results:
        k_total[r["k"]] += 1
        if r["metrics"]["em"] == 1:
            k_correct[r["k"]] += 1
    add("  " + "k".rjust(3) + " | " + "correct".rjust(8) + " | " + "total".rjust(6) + " | " + "EM%".rjust(7))
    add("  " + "-" * 32)
    for k in sorted(k_total):
        c = k_correct.get(k, 0)
        t = k_total[k]
        add("  " + str(k).rjust(3) + " | " + str(c).rjust(8) + " | " + str(t).rjust(6) + " | " + str(round(c/t*100, 1)).rjust(6) + "%")

    add()
    add("=" * 75)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 - Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    missing = []
    if not LLAMA_API_KEY:
        missing.append("UTSA_API_KEY")
    if not JUDGE_API_KEY:
        missing.append("JUDGE_API_KEY")
    if missing:
        raise EnvironmentError("Missing in .env: " + ", ".join(missing))

    llama = OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)
    qwen  = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    print("Loading BGE-M3 embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("  ready: " + EMBED_MODEL)
    print("Loading reranker model...")
    reranker = CrossEncoder(RERANK_MODEL)
    print("  ready: " + RERANK_MODEL)
    print("  Answerer : " + LLAMA_MODEL)
    print("  Judge    : " + JUDGE_MODEL + "\n")

    # Dataset 1: TriviaQA
    tqa_questions = load_triviaqa()
    tqa_questions = run_decomposition(tqa_questions, llama)
    results_tqa, skipped_tqa = run_dataset(
        "TriviaQA", tqa_questions,
        embedder, reranker, llama, qwen,
        TRIVIAQA_OUTPUT,
    )

    # Dataset 2: MuSiQue
    msq_questions = load_musique(MUSIQUE_JSONL)
    msq_questions = run_decomposition(msq_questions, llama)
    results_msq, skipped_msq = run_dataset(
        "MuSiQue", msq_questions,
        embedder, reranker, llama, qwen,
        MUSIQUE_OUTPUT,
    )

    # Combined save
    all_results = results_tqa + results_msq
    n_all       = len(all_results)

    combined_agg = {
        "em":        round(sum(r["metrics"]["em"]         for r in all_results)/n_all, 4) if n_all else 0,
        "f1":        round(sum(r["metrics"]["f1"]         for r in all_results)/n_all, 4) if n_all else 0,
        "precision": round(sum(r["metrics"]["precision"]  for r in all_results)/n_all, 4) if n_all else 0,
        "recall":    round(sum(r["metrics"]["recall"]     for r in all_results)/n_all, 4) if n_all else 0,
        "count":     n_all,
        "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in all_results)/n_all, 2) if n_all else 0,
        "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in all_results)/n_all, 2) if n_all else 0,
    }

    with open(COMBINED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "combined_aggregate": combined_agg,
            "triviaqa_count":     len(results_tqa),
            "musique_count":      len(results_msq),
            "results":            all_results,
        }, f, indent=2, ensure_ascii=False)
    print("\nCombined -> " + COMBINED_OUTPUT)

    summary = build_summary(results_tqa, results_msq, skipped_tqa, skipped_msq)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print("\nFiles: " + TRIVIAQA_OUTPUT + ", " + MUSIQUE_OUTPUT + ", " + COMBINED_OUTPUT)
    print("Summary: " + SUMMARY_FILE)


if __name__ == "__main__":
    main()
