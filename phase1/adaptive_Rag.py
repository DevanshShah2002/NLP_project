"""
step3_adaptive_rag_pipeline.py
-------------------------------
Full Adaptive RAG pipeline:
  1. Load decomposed_100_2hop.json  (questions + sub-questions from step2)
  2. Load sampled_100_2hop.json     (paragraphs + gold answers)
  3. For each question:
       a. Embed all 20 paragraphs with BGE-M3
       b. Loop k times (k = subquestion_count from step2):
            - Resolve [answer_N] placeholders in sub-question
            - Stage 1: BGE-M3 cosine → top-20 candidates
            - Stage 2: BERT CrossEncoder reranker → top-2 final
            - Call Llama 3.3-70B to answer ONLY that sub-question
              given the retrieved paragraph
       c. Use the last loop answer as the final answer
  4. Call Qwen3.5-27B (Judge LLM) with:
       - original question, all sub-questions, retrieved paragraphs,
         intermediate answers, final answer, ground truth
       → returns accuracy (0-100) and completeness (0-100)
  5. Compute paper's 5 metrics (EM, F1, Precision, Recall, Count)
     using the same token-overlap method as the original paper
  6. Save full results to pipeline_results.json

Usage:
    pip install openai python-dotenv sentence-transformers numpy
    # Connect to UTSA VPN first, then:
    python step3_adaptive_rag_pipeline.py

.env file (same folder):
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
from collections import Counter
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Load .env ──────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────
DECOMPOSED_FILE = "decomposed_100_2hop.json"   # from step2
SAMPLED_FILE    = "sampled_100_2hop.json"       # from step1 (has paragraphs + answers)
OUTPUT_FILE     = "pipeline_results_bge_reranker.json"

# Llama — answerer
LLAMA_API_KEY  = os.getenv("UTSA_API_KEY")
LLAMA_BASE_URL = os.getenv("UTSA_BASE_URL", "http://10.246.100.230/v1")
LLAMA_MODEL    = os.getenv("UTSA_MODEL",    "llama-3.3-70b-instruct-awq")

# Qwen — judge
JUDGE_API_KEY  = os.getenv("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://10.100.1.213:8888/v1")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL",    "Qwen/Qwen3.5-27B")

EMBED_MODEL    = "BAAI/bge-m3"                          # state-of-art embeddings
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2" # BERT-based reranker
TOP_K_RECALL   = 20   # stage 1: cosine keeps top-20 candidates
TOP_K_PARAS    = 2    # stage 2: reranker picks final top-2
DELAY_SEC      = 0.3
MAX_RETRIES    = 2


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Paper's evaluation metrics (same token-overlap logic)
# ══════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Lower, remove punctuation, articles, extra whitespace — same as paper."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_f1_precision_recall(prediction: str, ground_truths: list[str]) -> dict:
    """
    Compute F1, Precision, Recall against a list of acceptable answers.
    Take the max score across all ground truth answers (same as paper).
    """
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
    """Exact match — 1 if normalized prediction matches any ground truth."""
    pred_norm = normalize_answer(prediction)
    return int(any(pred_norm == normalize_answer(gt) for gt in ground_truths))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Two-stage retrieval: BGE-M3 cosine recall → BERT reranker
# ══════════════════════════════════════════════════════════════════════════
# Stage 1: BGE-M3 embeds query + all paragraphs → cosine similarity
#          → keeps top-20 candidates (full recall pass)
# Stage 2: CrossEncoder reranker scores each (query, paragraph) pair
#          directly → picks final top-k
# CrossEncoder is slower but much more accurate because it sees the query
# and paragraph together rather than as separate vectors.

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # BGE-M3 works best with normalized embeddings
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # With normalized embeddings this is just a dot product
    return float(np.dot(a, b))


def retrieve_top_k(
    query: str,
    query_emb: np.ndarray,
    para_embs: np.ndarray,
    paragraphs: list[dict],
    reranker: CrossEncoder,
    k: int = TOP_K_PARAS,
) -> list[dict]:
    """
    Stage 1 — cosine recall: score all paragraphs, keep top TOP_K_RECALL.
    Stage 2 — reranker: score each (query, para) pair, return top-k.
    """
    # ── Stage 1: cosine over all paragraphs ──────────────────────────────
    cos_scores = [cosine_similarity(query_emb, p_emb) for p_emb in para_embs]
    recall_idx = sorted(
        range(len(cos_scores)), key=lambda i: cos_scores[i], reverse=True
    )[:TOP_K_RECALL]

    # ── Stage 2: reranker on top-20 candidates ────────────────────────────
    candidates      = [paragraphs[i] for i in recall_idx]
    candidate_texts = [
        f"{p['title']}: {p['paragraph_text']}" for p in candidates
    ]
    pairs         = [[query, text] for text in candidate_texts]
    rerank_scores = reranker.predict(pairs).tolist()

    # Sort candidates by reranker score, take top-k
    ranked = sorted(
        zip(recall_idx, candidates, rerank_scores),
        key=lambda x: x[2],
        reverse=True,
    )[:k]

    return [
        {
            "paragraph":    para,
            "rerank_score": round(rerank_score, 4),
            "cosine_score": round(cos_scores[orig_idx], 4),
            "cosine_rank":  recall_idx.index(orig_idx),
        }
        for orig_idx, para, rerank_score in ranked
    ]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLM calls
# ══════════════════════════════════════════════════════════════════════════

def llm_call(client: OpenAI, model: str, system: str, user: str,
             max_tokens: int = 200) -> tuple[str, str | None]:
    """Generic LLM call. Returns (response_text, error_or_None)."""
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


def answer_subquestion(llama: OpenAI, subquestion: str, context: str) -> tuple[str, str | None]:
    system = (
        "You are a precise question-answering assistant. "
        "Answer the question using ONLY the provided context. "
        "Give a short, precise, direct answer — a few words or one sentence maximum. "
        "Do not explain or add extra information."
    )
    user = f"Context:\n{context}\n\nQuestion: {subquestion}\n\nAnswer:"
    return llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)


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
    """
    Judge LLM evaluates the full reasoning trace.
    Returns accuracy (0-100) and completeness (0-100).
    """
    # Build the reasoning trace for the judge
    trace_lines = []
    for i, (sq, ans, paras) in enumerate(
        zip(subquestions, intermediate_answers, retrieved_paras), 1
    ):
        para_texts = "\n".join(
            f"  [{j+1}] {p['paragraph']['title']}: {p['paragraph']['paragraph_text'][:200]}..."
            for j, p in enumerate(paras)
        )
        trace_lines.append(
            f"Step {i}:\n"
            f"  Sub-question: {sq}\n"
            f"  Retrieved paragraphs:\n{para_texts}\n"
            f"  Intermediate answer: {ans}"
        )

    all_gt = [ground_truth] + ground_truth_aliases

    system = (
        "You are an expert judge evaluating a multi-step question-answering system. "
        "You will be shown a reasoning trace and must evaluate it. "
        "Respond ONLY with a JSON object. No explanation, no markdown."
    )

    user = f"""Original Question: {original_question}

Reasoning Trace:
{chr(10).join(trace_lines)}

Final Answer: {final_answer}

Ground Truth Answer: {ground_truth}
Acceptable Aliases: {", ".join(ground_truth_aliases) if ground_truth_aliases else "none"}

Evaluate the system's performance and return ONLY this JSON:
{{
  "accuracy": <integer 0-100, how correct is the final answer vs ground truth>,
  "completeness": <integer 0-100, how completely did each step address its sub-question>,
  "reasoning": "<one sentence explaining your scores>"
}}"""

    raw, error = llm_call(qwen, JUDGE_MODEL, system, user, max_tokens=600)

    if error:
        print(error)
        return {"accuracy": 0, "completeness": 0, "reasoning": f"Judge error: {error}"}

    # Qwen3.5 thinking mode wraps output in <think>...</think> — strip it first
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    clean = re.sub(r"```json|```", "", clean).strip()
    print(clean)

    # Attempt 1: direct JSON parse
    try:
        parsed = json.loads(clean)
        return {
            "accuracy":     int(parsed.get("accuracy",     0)),
            "completeness": int(parsed.get("completeness", 0)),
            "reasoning":    str(parsed.get("reasoning",    "")),
        }
    except Exception:
        pass

    # Attempt 2: find first {...} block
    brace_match = re.search(r'\{[^{}]+\}', clean, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            return {
                "accuracy":     int(parsed.get("accuracy",     0)),
                "completeness": int(parsed.get("completeness", 0)),
                "reasoning":    str(parsed.get("reasoning",    "")),
            }
        except Exception:
            pass

    # Attempt 3: regex extract numbers directly
    acc  = re.search(r'"?accuracy"?\s*:\s*(\d+)',     clean)
    comp = re.search(r'"?completeness"?\s*:\s*(\d+)', clean)
    reas = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', clean)
    return {
        "accuracy":     int(acc.group(1))  if acc  else 0,
        "completeness": int(comp.group(1)) if comp else 0,
        "reasoning":    reas.group(1)      if reas else clean[:200],
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Placeholder resolution
# ══════════════════════════════════════════════════════════════════════════

def resolve_placeholders(subquestion: str, answers_so_far: list[str]) -> str:
    """Replace [answer_1], [answer_2], ... with actual answers from previous steps."""
    for i, ans in enumerate(answers_so_far, 1):
        subquestion = subquestion.replace(f"[answer_{i}]", ans)
    return subquestion


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main pipeline
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Validate credentials ──────────────────────────────────────────────
    missing = []
    if not LLAMA_API_KEY:  missing.append("UTSA_API_KEY")
    if not JUDGE_API_KEY:  missing.append("JUDGE_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing in .env: {', '.join(missing)}\n"
            "Make sure you are connected to UTSA VPN."
        )

    # ── Init clients ──────────────────────────────────────────────────────
    llama = OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)
    qwen  = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    print("Loading BGE-M3 embedding model (first time ~2.5GB download)...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"Embedding model ready: {EMBED_MODEL}")
    print("Loading reranker model (first time ~80MB download)...")
    reranker = CrossEncoder(RERANK_MODEL)
    print(f"Reranker ready: {RERANK_MODEL}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    with open(DECOMPOSED_FILE, encoding="utf-8") as f:
        decomposed = json.load(f)   # has subquestions per question id

    with open(SAMPLED_FILE, encoding="utf-8") as f:
        sampled = json.load(f)      # has paragraphs + gold answers

    # Build lookup: id → sampled entry
    sampled_by_id = {entry["id"]: entry for entry in sampled}

    print(f"Questions to process: {len(decomposed)}")
    print(f"Answerer : {LLAMA_MODEL}")
    print(f"Judge    : {JUDGE_MODEL}\n")

    results      = []
    total_em     = 0
    total_f1     = 0.0
    total_prec   = 0.0
    total_recall = 0.0
    skipped      = 0

    for idx, dec_entry in enumerate(decomposed, 1):
        qid          = dec_entry["id"]
        question     = dec_entry["question"]
        subquestions = dec_entry["subquestions"]
        k            = dec_entry["subquestion_count"]

        sampled_entry = sampled_by_id.get(qid)
        if not sampled_entry:
            print(f"[{idx:3d}/100] SKIP — no sampled entry for {qid}")
            skipped += 1
            continue

        paragraphs      = sampled_entry["paragraphs"]
        gold_answer     = sampled_entry["answer"]
        gold_aliases    = sampled_entry["answer_aliases"]
        all_gold        = [gold_answer] + gold_aliases

        print(f"[{idx:3d}/100] {question[:70]}")
        print(f"           k={k} sub-questions")

        # ── Embed all paragraphs once ─────────────────────────────────────
        para_texts = [
            f"{p['title']}: {p['paragraph_text']}" for p in paragraphs
        ]
        para_embs = embed_texts(embedder, para_texts)

        # ── RAG loop: k iterations ────────────────────────────────────────
        intermediate_answers = []
        retrieved_per_step   = []

        for step, raw_sq in enumerate(subquestions):
            # Resolve [answer_N] references
            resolved_sq = resolve_placeholders(raw_sq, intermediate_answers)

            # Two-stage retrieve: BGE-M3 cosine → reranker
            sq_emb    = embed_texts(embedder, [resolved_sq])[0]
            top_paras = retrieve_top_k(
                resolved_sq, sq_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
            )
            context   = "\n\n".join(
                f"{p['paragraph']['title']}: {p['paragraph']['paragraph_text']}"
                for p in top_paras
            )

            # Answer this sub-question
            ans, err = answer_subquestion(llama, resolved_sq, context)
            if err:
                ans = ""
                print(f"             step {step+1} LLM error: {err[:60]}")
            else:
                print(f"             step {step+1}: {resolved_sq[:60]}")
                print(f"                    → {ans[:80]}")

            intermediate_answers.append(ans)
            retrieved_per_step.append(top_paras)
            time.sleep(DELAY_SEC)

        # Final answer = last intermediate answer
        final_answer = intermediate_answers[-1] if intermediate_answers else ""

        # ── Paper's 5 metrics ─────────────────────────────────────────────
        em_score  = compute_em(final_answer, all_gold)
        pr_scores = compute_f1_precision_recall(final_answer, all_gold)

        total_em     += em_score
        total_f1     += pr_scores["f1"]
        total_prec   += pr_scores["precision"]
        total_recall += pr_scores["recall"]

        print(f"           Final answer : {final_answer}")
        print(f"           Gold answer  : {gold_answer}")
        print(f"           EM={em_score}  F1={pr_scores['f1']:.3f}")

        # ── Judge LLM evaluation ──────────────────────────────────────────
        print(f"           Calling Judge LLM...")
        judge_scores = judge_answer(
            qwen,
            question,
            subquestions,
            retrieved_per_step,
            intermediate_answers,
            final_answer,
            gold_answer,
            gold_aliases,
        )
        print(f"           Judge → accuracy={judge_scores['accuracy']}  "
              f"completeness={judge_scores['completeness']}")
        print()

        time.sleep(DELAY_SEC)

        # ── Store result ──────────────────────────────────────────────────
        results.append({
            # Question info
            "id":            qid,
            "question":      question,
            "subquestions":  subquestions,
            "k":             k,

            # RAG trace
            "steps": [
                {
                    "step":             i + 1,
                    "subquestion":      subquestions[i],
                    "resolved_subquestion": resolve_placeholders(
                        subquestions[i], intermediate_answers[:i]
                    ),
                    "retrieved_paragraphs": [
                        {
                            "title":        p["paragraph"]["title"],
                            "text":         p["paragraph"]["paragraph_text"][:300],
                            "is_supporting": p["paragraph"]["is_supporting"],
                            "rerank_score": p["rerank_score"],
                            "cosine_score": p["cosine_score"],
                            "cosine_rank":  p["cosine_rank"],
                        }
                        for p in retrieved_per_step[i]
                    ],
                    "intermediate_answer": intermediate_answers[i],
                }
                for i in range(len(subquestions))
            ],

            # Final answer
            "final_answer":  final_answer,
            "gold_answer":   gold_answer,
            "gold_aliases":  gold_aliases,

            # Paper's 5 metrics (per question)
            "metrics": {
                "em":        em_score,
                "f1":        round(pr_scores["f1"],        4),
                "precision": round(pr_scores["precision"], 4),
                "recall":    round(pr_scores["recall"],    4),
                "count":     1,
            },

            # Judge LLM scores
            "judge": {
                "accuracy":     judge_scores["accuracy"],
                "completeness": judge_scores["completeness"],
                "reasoning":    judge_scores["reasoning"],
            },
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────
    n = len(results)
    aggregate = {
        "em":        round(total_em / n, 4)     if n else 0,
        "f1":        round(total_f1 / n, 4)     if n else 0,
        "precision": round(total_prec / n, 4)   if n else 0,
        "recall":    round(total_recall / n, 4) if n else 0,
        "count":     n,
        "judge_accuracy_avg":     round(
            sum(r["judge"]["accuracy"]     for r in results) / n, 2) if n else 0,
        "judge_completeness_avg": round(
            sum(r["judge"]["completeness"] for r in results) / n, 2) if n else 0,
    }

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "aggregate_metrics": aggregate,
        "results": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Saved → {OUTPUT_FILE}")
    print(f"Processed: {n}/100  |  Skipped: {skipped}")
    print()
    print("── Paper metrics (aggregate) ──")
    print(f"  EM        : {aggregate['em']}")
    print(f"  F1        : {aggregate['f1']}")
    print(f"  Precision : {aggregate['precision']}")
    print(f"  Recall    : {aggregate['recall']}")
    print(f"  Count     : {aggregate['count']}")
    print()
    print("── Judge LLM (aggregate) ──")
    print(f"  Accuracy     avg : {aggregate['judge_accuracy_avg']} / 100")
    print(f"  Completeness avg : {aggregate['judge_completeness_avg']} / 100")


if __name__ == "__main__":
    main()