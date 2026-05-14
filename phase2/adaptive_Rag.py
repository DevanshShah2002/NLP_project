"""
step3_adaptive_rag_pipeline.py
-------------------------------
Full Adaptive RAG pipeline with:
  - Parallel question processing (ThreadPoolExecutor)
  - NOT FOUND retry loop with relaxed prompt
  - Inner refinement loop: if answer seems wrong type, retry with updated question
  - Per hop-type metrics
  - Summary saved to text file

Usage:
    pip install openai python-dotenv sentence-transformers numpy
    python step3_adaptive_rag_pipeline.py

.env (parent folder):
    UTSA_API_KEY=...
    UTSA_BASE_URL=http://10.246.100.230/v1
    UTSA_MODEL=llama-3.3-70b-instruct-awq
    JUDGE_API_KEY=...
    JUDGE_BASE_URL=http://10.100.1.213:8888/v1
    JUDGE_MODEL=Qwen/Qwen3.5-27B
"""

import json
import os
import re
import time
import string
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Load .env from parent folder ───────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────
DECOMPOSED_FILE = "decomposed_900_all.json"
SAMPLED_FILE    = "sampled_900_all.json"
OUTPUT_FILE     = "pipeline_results_v2.json"
SUMMARY_FILE    = "pipeline_summary_v2.txt"

LLAMA_API_KEY  = os.getenv("UTSA_API_KEY")
LLAMA_BASE_URL = os.getenv("UTSA_BASE_URL", "http://10.246.100.230/v1")
LLAMA_MODEL    = os.getenv("UTSA_MODEL",    "llama-3.3-70b-instruct-awq")

JUDGE_API_KEY  = os.getenv("JUDGE_API_KEY")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://10.100.1.213:8888/v1")
JUDGE_MODEL    = os.getenv("JUDGE_MODEL",    "Qwen/Qwen3.5-27B")

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RECALL = 20
TOP_K_PARAS  = 2
MAX_RETRIES  = 2
MAX_WORKERS  = 16    # parallel questions — raise to 10 on HPC

# Thread-safe print
_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Evaluation metrics
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
# SECTION 2 — Retrieval
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
# SECTION 3 — LLM calls
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


# ── Primary answer prompt (concise, not over-restrictive) ─────────────────
ANSWER_SYSTEM = (
    "You are a precise extractive question-answering assistant. "
    "Your answer MUST be copied word-for-word from the context below. "
    "Do NOT use any outside knowledge or memory. "
    "Do NOT infer, calculate, or guess. "
    "If the answer is not explicitly stated in the context, reply with: NOT FOUND. "
    "Give only the answer — no explanation, no full sentences, no punctuation at the end."
)

# ── Fallback prompt (more lenient — used on retry) ────────────────────────
ANSWER_SYSTEM_FALLBACK = (
    "You are a helpful question-answering assistant. "
    "The context below may contain a partial or nearby answer. "
    "Extract the closest relevant fact you can find — even if it is not perfectly precise. "
    "Give a short answer of a few words. "
    "Only reply NOT FOUND if the context has absolutely nothing related."
)


def answer_subquestion(
    llama: OpenAI,
    subquestion: str,
    context: str,
    fallback: bool = False,
) -> tuple[str, str | None]:
    system = ANSWER_SYSTEM_FALLBACK if fallback else ANSWER_SYSTEM
    user   = (
        f"Context:\n{context}\n\n"
        f"Question: {subquestion}\n\n"
        f"Answer:"
    )
    return llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)


def refine_question(
    llama: OpenAI,
    original_subquestion: str,
    partial_answer: str,
) -> str:
    """
    If we got a nearby but possibly wrong-type answer (e.g. got a state when
    we needed a county), ask the LLM to refine the sub-question using that
    partial answer to get the more specific fact.

    Example:
      original_subquestion: "Which county contains Ottawa?"
      partial_answer:        "Illinois"   ← got state, not county
      refined:               "Which county in Illinois contains Ottawa?"
    """
    system = (
        "You are a question refinement assistant. "
        "You will be given a sub-question and a partial answer that is close but "
        "may not be specific enough. "
        "Rewrite the sub-question to be more specific, incorporating the partial answer, "
        "so that the next retrieval can find the exact fact needed. "
        "Return ONLY the rewritten question. No explanation."
    )
    user = (
        f"Original sub-question: {original_subquestion}\n"
        f"Partial answer found: {partial_answer}\n"
        f"Rewritten sub-question (more specific):"
    )
    refined, err = llm_call(llama, LLAMA_MODEL, system, user, max_tokens=80)
    if err or not refined.strip():
        return original_subquestion  # fallback to original
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
    """
    Tries to answer a sub-question with up to 3 attempts:

    Attempt 1 — normal answer prompt
    Attempt 2 — if NOT FOUND: fallback prompt (more lenient)
    Attempt 3 — if still NOT FOUND or partial: refine sub-question using
                partial answer, re-retrieve, answer again

    Returns (final_answer, resolved_subquestion_used, top_paras, attempts_used)
    """
    # ── Attempt 1: normal ──────────────────────────────────────────────────
    ans, err = answer_subquestion(llama, subquestion, context, fallback=False)
    if err:
        ans = ""
    if ans.strip().upper() != "NOT FOUND" and ans.strip() != "":
        return ans, subquestion, [], 1

    # ── Attempt 2: fallback prompt, same context ───────────────────────────
    ans2, err2 = answer_subquestion(llama, subquestion, context, fallback=True)
    if err2:
        ans2 = ""

    if ans2.strip().upper() != "NOT FOUND" and ans2.strip() != "":
        # Got a partial/nearby answer — try inner refinement loop
        # Refine the question using the partial answer, re-retrieve, answer again
        refined_sq = refine_question(llama, subquestion, ans2)

        if refined_sq != subquestion:
            # Re-retrieve with refined question
            ref_emb      = embed_texts(embedder, [refined_sq])[0]
            refined_paras = retrieve_top_k(
                refined_sq, ref_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
            )
            refined_ctx = "\n\n".join(
                f"{p['paragraph']['title']}: {p['paragraph']['paragraph_text']}"
                for p in refined_paras
            )
            ans3, err3 = answer_subquestion(
                llama, refined_sq, refined_ctx, fallback=False
            )
            if err3:
                ans3 = ""
            if ans3.strip().upper() != "NOT FOUND" and ans3.strip() != "":
                return ans3, refined_sq, refined_paras, 3
        # Refined didn't help — use the partial answer from attempt 2
        return ans2, subquestion, [], 2

    # All attempts failed — return NOT FOUND
    return "NOT FOUND", subquestion, [], 2


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Judge
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
# SECTION 5 — Placeholder resolution
# ══════════════════════════════════════════════════════════════════════════

def resolve_placeholders(subquestion: str, answers_so_far: list[str]) -> str:
    for i, ans in enumerate(answers_so_far, 1):
        subquestion = subquestion.replace(f"[answer_{i}]", ans)
    return subquestion


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Per-question worker
# ══════════════════════════════════════════════════════════════════════════

def process_question(
    idx: int,
    total: int,
    dec_entry: dict,
    sampled_by_id: dict,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    llama: OpenAI,
    qwen: OpenAI,
) -> dict | None:
    qid          = dec_entry["id"]
    question     = dec_entry["question"]
    subquestions = dec_entry["subquestions"]
    k            = dec_entry["subquestion_count"]
    hop_type     = dec_entry.get("hop_type", "unknown")

    sampled_entry = sampled_by_id.get(qid)
    if not sampled_entry:
        tprint(f"[{idx:4d}/{total}] SKIP {qid}")
        return None

    paragraphs   = sampled_entry["paragraphs"]
    gold_answer  = sampled_entry["answer"]
    gold_aliases = sampled_entry["answer_aliases"]
    all_gold     = [gold_answer] + gold_aliases

    tprint(f"[{idx:4d}/{total}] [{hop_type}] k={k}  {question[:60]}")

    # Embed all paragraphs once
    para_texts = [f"{p['title']}: {p['paragraph_text']}" for p in paragraphs]
    para_embs  = embed_texts(embedder, para_texts)

    # ── Sequential k-loop ─────────────────────────────────────────────────
    intermediate_answers  = []
    retrieved_per_step    = []
    step_meta             = []   # tracks retry info per step

    for step, raw_sq in enumerate(subquestions):
        resolved_sq = resolve_placeholders(raw_sq, intermediate_answers)

        # Initial retrieval
        sq_emb    = embed_texts(embedder, [resolved_sq])[0]
        top_paras = retrieve_top_k(
            resolved_sq, sq_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
        )
        context = "\n\n".join(
            f"{p['paragraph']['title']}: {p['paragraph']['paragraph_text']}"
            for p in top_paras
        )

        # Answer with retry + inner refinement loop
        ans, used_sq, refined_paras, attempts = answer_with_retry(
            llama, resolved_sq, context,
            embedder, reranker, paragraphs, para_embs,
        )

        # If refinement found better paras, use those for the step record
        final_paras = refined_paras if refined_paras else top_paras

        tprint(f"         [{hop_type}] step {step+1} (attempts={attempts}): "
               f"{used_sq[:55]}")
        tprint(f"                  → {ans[:70]}")

        intermediate_answers.append(ans)
        retrieved_per_step.append(final_paras)
        step_meta.append({
            "attempts":         attempts,
            "refined":          used_sq != resolved_sq,
            "original_sq":      resolved_sq,
            "used_sq":          used_sq,
        })

    final_answer = intermediate_answers[-1] if intermediate_answers else ""

    # ── Metrics ───────────────────────────────────────────────────────────
    em_score  = compute_em(final_answer, all_gold)
    pr_scores = compute_f1_precision_recall(final_answer, all_gold)

    tprint(f"         [{hop_type}] final='{final_answer[:50]}'  "
           f"gold='{gold_answer[:40]}'  EM={em_score}  F1={pr_scores['f1']:.3f}")

    # ── Judge ─────────────────────────────────────────────────────────────
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
# SECTION 7 — Summary builder (also saves to text file)
# ══════════════════════════════════════════════════════════════════════════

def build_summary(results: list[dict], skipped: int, total: int) -> str:
    n      = len(results)
    lines  = []

    def add(s=""):
        lines.append(s)

    hop_order = ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"]
    by_hop: dict[str, list] = defaultdict(list)
    for r in results:
        by_hop[r.get("hop_type", "unknown")].append(r)

    add("=" * 70)
    add("PIPELINE RESULTS SUMMARY")
    add("=" * 70)

    # ── Per hop-type table ────────────────────────────────────────────────
    add()
    add("── Per hop-type metrics ──")
    add()
    add(f"  {'Hop':6s} | {'N':>4} | {'EM':>6} | {'F1':>6} | "
        f"{'Prec':>6} | {'Recall':>6} | {'J.Acc':>6} | {'J.Comp':>6} | {'NOT_FOUND':>10}")
    add("  " + "-" * 75)

    total_em = total_f1 = total_prec = total_recall = 0.0
    total_jacc = total_jcomp = 0.0

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
        total_em    += em * hn; total_f1    += f1 * hn
        total_prec  += prec*hn; total_recall+= rec * hn
        total_jacc  += jacc*hn; total_jcomp += jcomp*hn
        add(f"  {hop:6s} | {hn:4d} | {em:6.3f} | {f1:6.3f} | "
            f"{prec:6.3f} | {rec:6.3f} | {jacc:6.1f} | {jcomp:6.1f} | "
            f"{nf:4d} ({nf/hn*100:.1f}%)")

    add(f"  {'─'*75}")
    add(f"  {'TOTAL':6s} | {n:4d} | {total_em/n:6.3f} | {total_f1/n:6.3f} | "
        f"{total_prec/n:6.3f} | {total_recall/n:6.3f} | "
        f"{total_jacc/n:6.1f} | {total_jcomp/n:6.1f} |")

    # ── F1 distribution ───────────────────────────────────────────────────
    add(); add("── F1 distribution ──"); add()
    buckets = [
        ("0.00        (none)",    lambda f: f == 0.0),
        ("0.01–0.29   (low)",     lambda f: 0 < f < 0.3),
        ("0.30–0.59   (partial)", lambda f: 0.3 <= f < 0.6),
        ("0.60–0.99   (close)",   lambda f: 0.6 <= f < 1.0),
        ("1.00        (exact)",   lambda f: f == 1.0),
    ]
    for label, fn in buckets:
        cnt = sum(1 for r in results if fn(r["metrics"]["f1"]))
        bar = "█" * min(cnt // 5, 50)
        add(f"  F1={label}: {cnt:4d} ({cnt/n*100:5.1f}%)  {bar}")

    # ── Error analysis ────────────────────────────────────────────────────
    add(); add("── Error analysis ──"); add()
    bad       = [r for r in results if r["metrics"]["em"] == 0]
    nf_final  = [r for r in bad if "NOT FOUND" in r["final_answer"].upper()]
    wrong     = [r for r in bad if "NOT FOUND" not in r["final_answer"].upper()
                                and r["final_answer"].strip() != ""]
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

    # ── k accuracy ───────────────────────────────────────────────────────
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

    add(); add(f"Processed: {n}/{total}  Skipped: {skipped}")
    add("=" * 70)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    missing = []
    if not LLAMA_API_KEY: missing.append("UTSA_API_KEY")
    if not JUDGE_API_KEY: missing.append("JUDGE_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing in .env: {', '.join(missing)}")

    llama = OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)
    qwen  = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    print("Loading BGE-M3...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"  ready: {EMBED_MODEL}")
    print("Loading reranker...")
    reranker = CrossEncoder(RERANK_MODEL)
    print(f"  ready: {RERANK_MODEL}\n")

    with open(DECOMPOSED_FILE, encoding="utf-8") as f:
        decomposed = json.load(f)
    with open(SAMPLED_FILE, encoding="utf-8") as f:
        sampled = json.load(f)

    sampled_by_id = {e["id"]: e for e in sampled}
    total = len(decomposed)

    print(f"Questions : {total}")
    print(f"Workers   : {MAX_WORKERS}")
    print(f"Answerer  : {LLAMA_MODEL}")
    print(f"Judge     : {JUDGE_MODEL}\n")

    results   = []
    skipped   = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                process_question,
                i + 1, total, dec_entry, sampled_by_id,
                embedder, reranker, llama, qwen
            ): i
            for i, dec_entry in enumerate(decomposed)
        }

        for future in as_completed(future_to_idx):
            result = future.result()
            completed += 1
            if result is None:
                skipped += 1
            else:
                results.append(result)
            tprint(f"  ── completed {completed}/{total} ──")

    # Restore original order
    original_order = {e["id"]: i for i, e in enumerate(decomposed)}
    results.sort(key=lambda r: original_order.get(r["id"], 0))

    # Build per-hop aggregate for JSON
    by_hop: dict[str, list] = defaultdict(list)
    for r in results:
        by_hop[r.get("hop_type", "unknown")].append(r)

    per_hop_metrics = {}
    for hop, hrs in by_hop.items():
        hn = len(hrs)
        per_hop_metrics[hop] = {
            "count":                hn,
            "em":                   round(sum(r["metrics"]["em"]        for r in hrs)/hn, 4),
            "f1":                   round(sum(r["metrics"]["f1"]        for r in hrs)/hn, 4),
            "precision":            round(sum(r["metrics"]["precision"] for r in hrs)/hn, 4),
            "recall":               round(sum(r["metrics"]["recall"]    for r in hrs)/hn, 4),
            "judge_accuracy_avg":   round(sum(r["judge"]["accuracy"]     for r in hrs)/hn, 2),
            "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in hrs)/hn, 2),
        }

    n = len(results)
    aggregate = {
        "em":        round(sum(r["metrics"]["em"]        for r in results)/n, 4) if n else 0,
        "f1":        round(sum(r["metrics"]["f1"]        for r in results)/n, 4) if n else 0,
        "precision": round(sum(r["metrics"]["precision"] for r in results)/n, 4) if n else 0,
        "recall":    round(sum(r["metrics"]["recall"]    for r in results)/n, 4) if n else 0,
        "count":     n,
        "judge_accuracy_avg":     round(sum(r["judge"]["accuracy"]     for r in results)/n, 2) if n else 0,
        "judge_completeness_avg": round(sum(r["judge"]["completeness"] for r in results)/n, 2) if n else 0,
    }

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"aggregate_metrics": aggregate,
                   "per_hop_metrics":   per_hop_metrics,
                   "results":           results}, f, indent=2, ensure_ascii=False)

    # Build + save + print summary
    summary = build_summary(results, skipped, total)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\nResults → {OUTPUT_FILE}")
    print(f"Summary → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()