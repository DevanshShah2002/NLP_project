"""
Adaptive RAG Explorer — Streamlit UI
=====================================
• Questions loaded directly from musique_ans_v1.0_train.jsonl (raw dataset)
• Sidebar: paginated list of 20 questions at a time with search + filters
• Main screen: select a question → pipeline runs live, shows step-by-step trace
• Startup: animated progress bar + step indicators while BGE-M3 + CrossEncoder load

Usage:
    pip install streamlit sentence-transformers openai python-dotenv
    streamlit run app.py

.env (same folder or parent):
    UTSA_API_KEY=...
    UTSA_BASE_URL=http://10.246.100.230/v1
    UTSA_MODEL=llama-3.3-70b-instruct-awq
"""

import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── load env ──────────────────────────────────────────────────────────────────
for p in [Path(".env"), Path("../.env"), Path("../../.env")]:
    if p.exists():
        load_dotenv(p)
        break

LLAMA_API_KEY  = os.getenv("UTSA_API_KEY",   "dummy")
LLAMA_BASE_URL = os.getenv("UTSA_BASE_URL",  "http://10.246.100.230/v1")
LLAMA_MODEL    = os.getenv("UTSA_MODEL",     "llama-3.3-70b-instruct-awq")

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RECALL = 20
TOP_K_PARAS  = 2
MAX_RETRIES  = 2

# Raw MuSiQue dataset — questions come from here, NOT from results JSON
INPUT_JSONL = Path(__file__).parent.parent / "phase1" / "musique_ans_v1.0_train.jsonl"
PAGE_SIZE   = 20

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a3045; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* ── sidebar question card ── */
.q-card {
    background: #1e2535;
    border: 1px solid #2e3a55;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s;
}
.q-card:hover { border-color: #4a90e2; background: #253047; }
.q-card.active { border-color: #4a90e2; background: #1a3a6b; }
.q-card .qid   { font-size: 10px; color: #6b7a99; font-family: monospace; }
.q-card .qtxt  { font-size: 13px; color: #c9d1e0; margin: 4px 0; line-height: 1.4; }
.q-card .qmeta { font-size: 11px; color: #4a90e2; }

/* ── hop badges ── */
.hop-badge {
    display:inline-block; padding:2px 9px; border-radius:12px;
    font-size:11px; font-weight:700; letter-spacing:.5px;
}
.hop-2hop  { background:#1a4a2e; color:#4ade80; border:1px solid #2d7a4f; }
.hop-3hop1 { background:#1a3a6b; color:#60a5fa; border:1px solid #2d5a9e; }
.hop-3hop2 { background:#2d1a6b; color:#a78bfa; border:1px solid #4a2da8; }
.hop-4hop1 { background:#6b2d1a; color:#fb923c; border:1px solid #a84520; }
.hop-4hop2 { background:#6b1a3a; color:#f472b6; border:1px solid #a8205a; }
.hop-4hop3 { background:#4a3a1a; color:#fbbf24; border:1px solid #876a1a; }

/* ── main chat area ── */
.chat-bubble-q {
    background: linear-gradient(135deg, #1a3a6b, #0d2a52);
    border: 1px solid #2d5a9e;
    border-radius: 16px 16px 16px 4px;
    padding: 16px 20px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 16px;
    font-weight: 600;
    line-height: 1.5;
}

/* ── step card ── */
.step-card {
    background: #161b27;
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    position: relative;
}
.step-card.active-step { border-color: #4a90e2; box-shadow: 0 0 12px rgba(74,144,226,.2); }
.step-number {
    background: linear-gradient(135deg, #1a3a6b, #4a90e2);
    color: white;
    border-radius: 50%;
    width: 32px; height: 32px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    margin-right: 10px;
}
.step-label { font-size: 12px; color: #6b7a99; text-transform: uppercase; letter-spacing: .8px; }
.step-value { font-size: 14px; color: #c9d1e0; margin-top: 4px; line-height: 1.5; }

/* ── subquestion box ── */
.subq-box {
    background: #1e2535;
    border-left: 3px solid #4a90e2;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #93c5fd;
    font-style: italic;
}

/* ── paragraph card ── */
.para-card {
    background: #1a2535;
    border: 1px solid #2a3a55;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
}
.para-card.supporting {
    border-color: #2d7a4f;
    background: #1a2e22;
}
.para-title {
    font-size: 12px; font-weight: 700;
    color: #60a5fa; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: .5px;
}
.para-score { font-size: 11px; color: #6b7a99; float: right; }

/* ── answer bubble ── */
.answer-bubble {
    background: linear-gradient(135deg, #1a4a2e, #0d3320);
    border: 1px solid #2d7a4f;
    border-radius: 4px 16px 16px 16px;
    padding: 12px 18px;
    margin: 8px 0;
    font-size: 15px;
    font-weight: 600;
    color: #4ade80;
}
.answer-bubble.notfound {
    background: linear-gradient(135deg, #4a1a1a, #3a0d0d);
    border-color: #7a2d2d;
    color: #f87171;
}
.answer-bubble.fallback {
    background: linear-gradient(135deg, #4a3a1a, #3a2a0d);
    border-color: #876a1a;
    color: #fbbf24;
}

/* ── retry badge ── */
.retry-badge {
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-size:10px; font-weight:700;
    background:#2d1a1a; color:#f87171; border:1px solid #7a2d2d;
    margin-left:6px;
}
.refined-badge {
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-size:10px; font-weight:700;
    background:#1a2d4a; color:#60a5fa; border:1px solid #2d5a9e;
    margin-left:6px;
}

/* ── score cards ── */
.score-row { display:flex; gap:12px; margin:12px 0; flex-wrap:wrap; }
.score-card {
    flex:1; min-width:100px;
    background:#1e2535; border:1px solid #2e3a55;
    border-radius:10px; padding:12px 16px; text-align:center;
}
.score-card.pass  { border-color:#2d7a4f; background:#1a2e22; }
.score-card.fail  { border-color:#7a2d2d; background:#2e1a1a; }
.score-card.mid   { border-color:#876a1a; background:#2e2a1a; }
.score-num { font-size:26px; font-weight:800; color:#ffffff; }
.score-lbl { font-size:11px; color:#6b7a99; margin-top:2px; text-transform:uppercase; letter-spacing:.8px; }

/* ── k badge ── */
.k-badge {
    background:linear-gradient(135deg,#1a3a6b,#4a90e2);
    color:white; border-radius:8px; padding:6px 16px;
    font-size:22px; font-weight:800; display:inline-block;
}

/* ── section header ── */
.section-header {
    color:#6b7a99; font-size:11px; font-weight:700;
    text-transform:uppercase; letter-spacing:1.2px;
    border-bottom:1px solid #2a3045; padding-bottom:6px;
    margin:18px 0 10px 0;
}

/* ── loading screen ── */
.loading-screen {
    text-align:center; padding:60px 20px;
}
.loading-title { font-size:28px; font-weight:800; color:#4a90e2; margin-bottom:10px; }
.loading-sub   { font-size:14px; color:#6b7a99; margin-bottom:30px; }
.loader-step   { font-size:13px; color:#c9d1e0; padding:6px; }
.loader-step.done { color:#4ade80; }
.loader-step.active { color:#fbbf24; }

/* ── pipeline live ── */
.live-badge {
    background:linear-gradient(135deg,#7a1a1a,#c00);
    color:white; border-radius:6px; padding:3px 10px;
    font-size:11px; font-weight:700; letter-spacing:.5px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* ── scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0f1117; }
::-webkit-scrollbar-thumb { background:#2a3045; border-radius:3px; }

/* ── page nav ── */
.page-nav { display:flex; align-items:center; gap:8px; margin:8px 0; }
.page-info { font-size:12px; color:#6b7a99; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHING — models + data
# ══════════════════════════════════════════════════════════════════════════════

HOP_TYPES = ["4hop3","4hop2","4hop1","3hop2","3hop1","2hop"]

def get_hop_type(qid):
    for h in HOP_TYPES:
        if qid.startswith(h):
            return h
    return "unknown"

@st.cache_resource(show_spinner=False)
def load_models():
    """Load BGE-M3 + CrossEncoder. Runs once, cached across all sessions."""
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker  = CrossEncoder(RERANK_MODEL)
    return embedder, reranker

@st.cache_resource(show_spinner=False)
def get_llama_client():
    return OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)

@st.cache_data(show_spinner=False)
def load_questions():
    """
    Load all answerable questions from musique_ans_v1.0_train.jsonl.
    Returns a list of lightweight dicts with: id, question, hop_type,
    answer, answer_aliases, paragraphs (for pipeline use).
    """
    path = INPUT_JSONL
    if not path.exists():
        # fallback: try same folder as app.py
        path = Path(__file__).parent / "musique_ans_v1.0_train.jsonl"
    if not path.exists():
        return []

    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry.get("answerable", True):
                continue
            qid = entry["id"]
            questions.append({
                "id":            qid,
                "hop_type":      get_hop_type(qid),
                "question":      entry["question"],
                "answer":        entry.get("answer", ""),
                "answer_aliases":entry.get("answer_aliases", []),
                # keep full paragraphs for retrieval
                "paragraphs":    entry.get("paragraphs", []),
            })
    return questions


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS (same logic as adaptive_rag_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s):
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def compute_em(pred, golds):
    pn = normalize_answer(pred)
    return int(any(pn == normalize_answer(g) for g in golds))

def compute_f1(pred, golds):
    best = 0.0
    pt = Counter(normalize_answer(pred).split())
    for g in golds:
        gt = Counter(normalize_answer(g).split())
        common = pt & gt
        ns = sum(common.values())
        if ns == 0: continue
        pr = ns / sum(pt.values())
        rc = ns / sum(gt.values())
        f1 = 2*pr*rc/(pr+rc)
        best = max(best, f1)
    return best

def embed_texts(embedder, texts):
    return embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def retrieve_top_k(query, q_emb, para_embs, paragraphs, reranker, k):
    sims   = np.dot(para_embs, q_emb)
    top_n  = min(TOP_K_RECALL, len(paragraphs))
    idx    = np.argsort(sims)[::-1][:top_n]
    pairs  = [[query, paragraphs[i]["paragraph_text"]] for i in idx]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, idx), reverse=True)[:k]
    return [
        {
            "paragraph":    paragraphs[i],
            "rerank_score": float(s),
            "cosine_score": float(sims[i]),
        }
        for s, i in ranked
    ]

def llm_call(client, system, user, max_tokens=200):
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=LLAMA_MODEL,
                messages=[{"role":"system","content":system},
                          {"role":"user",  "content":user}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip(), None
        except Exception as e:
            if attempt < MAX_RETRIES: time.sleep(2**attempt)
            else: return "", str(e)

DECOMPOSE_SYSTEM = """You are a question decomposition expert for a multi-hop question answering system. Your task: decompose a complex question into the exact number of sub-questions needed — one per unknown fact that must be retrieved in sequence. Break the question into exactly the sub-questions needed — one per unknown fact. ════════════════════════════════════════ HOW TO THINK (chain-of-thought process) ════════════════════════════════════════ Step 1 — READ the full question carefully. Step 2 — IDENTIFY named entities already given (people, films, places, events). These are KNOWN. Do not ask about them. Step 3 — FIND the nested chain: questions hide a chain like "X of the Y of the Z of the W". Each "of the" usually = one unknown hop. Step 4 — COUNT the unknowns: how many separate database lookups are needed? That is your k. Step 5 — WRITE sub-questions from inside-out (resolve the deepest unknown first, work outward). Step 6 — CHECK: can any two adjacent steps be answered in a single lookup? If yes, merge them. If each needs a separate search, keep them separate. Step 7 — OUTPUT the JSON array. Do not revise or restart inside the array. ════════════════════════════════════════ STRICT RULES ════════════════════════════════════════ 1. One sub-question = one unknown fact = one retrieval step. No more, no less. 2. Named entities stated in the question are KNOWN — never ask who/what they are. 3. Descriptions like "the court that does X" or "the body with power to Y" are NOT named entities — they must be resolved first. 4. Do NOT add verification steps ("Is there X?", "Is [answer_1] same as [answer_2]?"). 5. Do NOT add sub-questions for facts already given in the question. 6. For 4-hop questions: all 4 layers MUST be separate sub-questions — middle steps cannot be merged even if they seem connected. 7. Later sub-questions MUST reference earlier answers as [answer_1], [answer_2], etc. 8. Output the array in ONE attempt. Do NOT revise, restart, or self-correct inside the array. 9. Return ONLY a JSON array of strings. No explanation, no markdown, no extra text. 10. Maximum 6 items. ════════════════════════════════════════ 2-HOP EXAMPLES ════════════════════════════════════════ Q: "In which country was the director of Titanic born?" Output: ["Who directed Titanic?", "In which country was [answer_1] born?"] Q: "When was the astronomical clock built in the city that Lucie Hradecká calls home?" Output: ["What city does Lucie Hradecká call home?", "When was the astronomical clock in [answer_1] built?"] Q: "What is the record label of the singer who performed the theme song of Titanic?" Output: ["Who performed the theme song of Titanic?", "What is the record label of [answer_1]?"] ════════════════════════════════════════ 3-HOP EXAMPLES ════════════════════════════════════════ Q: "What is the nationality of the director of the film that won the Academy Award for Best Picture in 2020?" Output: ["Which film won the Academy Award for Best Picture in 2020?", "Who directed [answer_1]?", "What is the nationality of [answer_2]?"] Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?" Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"] ════════════════════════════════════════ 4-HOP EXAMPLES ════════════════════════════════════════ Q: "What is the birth country of the 2018 Super Bowl halftime performer who released a live album recorded in the city that The Times added to its masthead in 2012?" Output: ["Who performed at the 2018 Super Bowl halftime show?", "What city did The Times add to its masthead in 2012?", "What live album did [answer_1] record in [answer_2]?", "What is the birth country of [answer_1]?"] Q: "What is the capital of the country where the headquarters of the company founded by the author of Harry Potter is located?" Output: ["Who authored Harry Potter?", "What company did [answer_1] found?", "In which country is [answer_2] headquartered?", "What is the capital of [answer_3]?"] PATTERN 1 — Linear chain: Q: "What is the nationality of the director of the film that won Best Picture in 2020?" Output: ["Which film won Best Picture in 2020?", "Who directed [answer_1]?", "What is the nationality of [answer_2]?"] PATTERN 2 — Two parallel lookups: Q: "Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?" Output: ["Who was Eisenhower's Vice President?", "What country is [answer_1] from?", "Which country, along with [answer_2], recognized Gaddafi's government early on?", "Where is the lowest place in [answer_3]?"] PATTERN 3 — Performer/creator + separate location constraint: Q: "What is the birth country of the 2018 Super Bowl halftime performer who released a live album recorded in the city that The Times added to its masthead in 2012?" Output: ["Who performed at the 2018 Super Bowl halftime show?", "What city did The Times add to its masthead in 2012?", "What live album did [answer_1] record in [answer_2]?", "What is the birth country of [answer_1]?"] PATTERN 4 — Description that must be resolved before use: Q: "When does the body with the power to remove a justice from the court of criminal appeals begin its work?" Output: ["What is the court of criminal appeals?", "What body has the power to remove a justice from [answer_1]?", "When does [answer_2] begin its work?"]"""


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


def decompose_question(client, question):
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
            return parse_subquestions(raw)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    return []

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

def answer_subq(client, sq, context, fallback=False):
    system = ANSWER_SYSTEM_FALLBACK if fallback else ANSWER_SYSTEM
    user   = f"Context:\n{context}\n\nQuestion: {sq}\n\nAnswer:"
    return llm_call(client, system, user, max_tokens=80)


def refine_question(client, original_sq, partial_answer):
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
    refined, err = llm_call(client, system, user, max_tokens=80)
    if err or not refined.strip():
        return original_sq
    return refined.strip()

def judge_live(client, question, steps_data, final_answer, gold, gold_aliases=None):
    gold_aliases = gold_aliases or []
    trace_lines = []
    for i, s in enumerate(steps_data, 1):
        paras = s.get("retrieved_paragraphs", [])
        para_texts = "\n".join(
            f"  [{j+1}] {p['title']}: {p['text'][:200]}..."
            for j, p in enumerate(paras)
        )
        trace_lines.append(
            f"Step {i}:\n"
            f"  Sub-question: {s['used_sq']}\n"
            f"  Retrieved:\n{para_texts}\n"
            f"  Answer: {s['intermediate_answer']}"
        )
    system = (
        "You are an expert judge evaluating a multi-step QA system. "
        "Respond ONLY with a JSON object. No markdown."
    )
    user = f"""Original Question: {question}

Reasoning Trace:
{chr(10).join(trace_lines)}

Final Answer: {final_answer}
Ground Truth: {gold}
Aliases: {", ".join(gold_aliases) if gold_aliases else "none"}

Return ONLY this JSON:
{{
  "accuracy": <integer 0-100>,
  "completeness": <integer 0-100>,
  "reasoning": "<one sentence>"
}}"""
    raw, err = llm_call(client, system, user, max_tokens=300)
    if err:
        return {"accuracy": 0, "completeness": 0, "reasoning": f"error: {err}"}
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
    a = re.search(r'"?accuracy"?\s*:\s*(\d+)',     clean)
    c = re.search(r'"?completeness"?\s*:\s*(\d+)', clean)
    r = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', clean)
    return {
        "accuracy":     int(a.group(1)) if a else 0,
        "completeness": int(c.group(1)) if c else 0,
        "reasoning":    r.group(1)      if r else clean[:150],
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

HOP_COLORS = {
    "2hop":"2hop", "3hop1":"3hop1", "3hop2":"3hop2",
    "4hop1":"4hop1", "4hop2":"4hop2", "4hop3":"4hop3"
}

def hop_badge(hop):
    cls = HOP_COLORS.get(hop, "2hop")
    return f'<span class="hop-badge hop-{cls}">{hop}</span>'

def score_color_class(val, lbl):
    if lbl == "EM":
        return "pass" if val == 1 else "fail"
    if lbl == "F1":
        if val >= 0.7: return "pass"
        if val >= 0.4: return "mid"
        return "fail"
    if lbl == "Steps":
        return "mid"   # always informational
    if lbl == "Retries":
        return "pass" if val == 0 else ("mid" if val <= 2 else "fail")
    # judge 0-100
    if val >= 70: return "pass"
    if val >= 40: return "mid"
    return "fail"

def render_score_cards(metrics, judge, steps=None):
    total_steps   = len(steps) if steps else 0
    total_retries = sum(s.get("attempts", 1) - 1 for s in steps) if steps else 0

    cards = [
        ("EM",         1 if metrics["em"] else 0, "EM"),
        ("F1",         round(metrics["f1"], 2),    "F1"),
        ("Steps",      total_steps,                "Steps"),
        ("Retries",    total_retries,              "Retries"),
        ("J.Accuracy", judge["accuracy"],          "J.Accuracy"),
        ("J.Complete", judge["completeness"],      "J.Complete"),
    ]
    html = '<div class="score-row">'
    for lbl, val, _ in cards:
        cls  = score_color_class(val, lbl)
        disp = "✓" if (lbl == "EM" and val == 1) else ("✗" if lbl == "EM" else str(val))
        html += f'''<div class="score-card {cls}">
            <div class="score-num">{disp}</div>
            <div class="score-lbl">{lbl}</div>
        </div>'''
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_step(step_data, step_num, total_steps):
    sq      = step_data.get("subquestion", "")
    rsq     = step_data.get("resolved_sq", sq)
    used_sq = step_data.get("used_sq", rsq)
    ans     = step_data.get("intermediate_answer", "")
    paras   = step_data.get("retrieved_paragraphs", [])
    attempts= step_data.get("attempts", 1)
    refined = step_data.get("refined", False)

    retry_html   = '<span class="retry-badge">RETRY</span>'   if attempts > 1 else ""
    refined_html = '<span class="refined-badge">REFINED</span>' if refined    else ""

    is_nf  = "NOT FOUND" in ans.upper()
    is_fb  = attempts >= 2 and not is_nf

    # ── sub-question html ────────────────────────────────────────────────────
    if sq != rsq:
        sq_html = (
            f'<div class="subq-box" style="color:#6b7a99">'
            f'<span style="font-size:10px;color:#4b5563">TEMPLATE:</span> {sq}</div>'
            f'<div class="subq-box">→ {rsq}</div>'
        )
    else:
        sq_html = f'<div class="subq-box">{sq}</div>'
    if refined and used_sq != rsq:
        sq_html += (
            f'<div class="subq-box" style="border-color:#4a90e2;color:#93c5fd">'
            f'🔁 Refined: {used_sq}</div>'
        )

    # ── paragraphs html ──────────────────────────────────────────────────────
    paras_html = ""
    for p in paras:
        supp  = p.get("is_supporting", False)
        title = p.get("title", "Unknown")
        text  = p.get("text", "")[:280]
        rsc   = p.get("rerank_score", 0.0)
        csc   = p.get("cosine_score", 0.0)
        pcls  = "para-card supporting" if supp else "para-card"
        sup_i = "✅ Supporting" if supp else "○ Distractor"
        paras_html += (
            f'<div class="{pcls}">'
            f'<div class="para-title">{title}'
            f'<span class="para-score">{sup_i} &nbsp;|&nbsp; rerank {rsc:.3f} &nbsp;|&nbsp; cos {csc:.3f}</span>'
            f'</div>{text}…</div>'
        )

    # ── answer html ──────────────────────────────────────────────────────────
    ans_cls = "notfound" if is_nf else ("fallback" if is_fb else "")
    icon    = "❌" if is_nf else ("⚠️" if is_fb else "✅")

    # ── single markdown call so all divs open and close in the same block ────
    st.markdown(f"""
<div class="step-card">
  <div style="display:flex;align-items:center;margin-bottom:10px">
    <span class="step-number">{step_num}</span>
    <span style="font-size:14px;font-weight:700;color:#c9d1e0">Reasoning Step</span>
    <span style="margin-left:auto;font-size:11px;color:#6b7a99">Step {step_num}/{total_steps}</span>
    {retry_html}{refined_html}
  </div>
  <div class="section-header">Sub-Question</div>
  {sq_html}
  <div class="section-header">Retrieved Paragraphs</div>
  {paras_html}
  <div class="section-header">Intermediate Answer</div>
  <div class="answer-bubble {ans_cls}">{icon} {ans}</div>
</div>
""", unsafe_allow_html=True)


def render_result(result):
    question   = result["question"]
    hop_type   = result.get("hop_type","?")
    k          = result.get("k", len(result.get("steps",[])))
    steps      = result.get("steps", [])
    final_ans  = result.get("final_answer","")
    gold       = result.get("gold_answer","")
    aliases    = result.get("gold_aliases",[])
    metrics    = result.get("metrics",{})
    judge      = result.get("judge",{})
    subqs      = result.get("subquestions",[])
    qid        = result.get("id","")

    # ── question bubble ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:6px">
      <div style="font-size:24px">🙋</div>
      <div>
        <div style="margin-bottom:6px">{hop_badge(hop_type)} &nbsp;
          <span style="font-size:11px;color:#6b7a99;font-family:monospace">{qid}</span>
        </div>
        <div class="chat-bubble-q">{question}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── k + decomposition ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:#161b27;border:1px solid #2a3045;border-radius:12px">
          <div style="font-size:11px;color:#6b7a99;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px">Hops (k)</div>
          <div class="k-badge">{k}</div>
          <div style="font-size:11px;color:#4a90e2;margin-top:8px">{hop_type}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-header">Decomposed Sub-Questions</div>', unsafe_allow_html=True)
        for i, sq in enumerate(subqs, 1):
            st.markdown(f"""
            <div class="subq-box">
              <span style="font-size:11px;color:#4b5563;font-weight:700">Q{i}:</span> {sq}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # ── step-by-step ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step-by-Step RAG Trace</div>', unsafe_allow_html=True)
    for step in steps:
        render_step(step, step["step"], len(steps))

    # ── final answer ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Final Answer</div>', unsafe_allow_html=True)
    is_nf = "NOT FOUND" in final_ans.upper()
    fa_cls = "notfound" if is_nf else ""
    fa_icon = "❌" if is_nf else "🤖"
    st.markdown(f"""
    <div style="display:flex;gap:12px;align-items:flex-start">
      <div style="font-size:20px">{fa_icon}</div>
      <div style="flex:1">
        <div class="answer-bubble {fa_cls}" style="font-size:16px">{final_ans}</div>
        <div style="margin-top:6px">
          <span style="font-size:12px;color:#6b7a99">Gold answer: </span>
          <span style="font-size:13px;color:#4ade80;font-weight:600">{gold}</span>
          {"<span style='font-size:11px;color:#6b7a99'> (aliases: " + ", ".join(aliases[:3]) + ")</span>" if aliases else ""}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── evaluation scores ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Evaluation Scores</div>', unsafe_allow_html=True)
    render_score_cards(metrics, judge, steps=steps)

    if judge.get("reasoning"):
        st.markdown(f"""
        <div style="background:#1e2535;border:1px solid #2e3a55;border-radius:8px;
                    padding:12px 16px;margin-top:8px;font-size:13px;color:#94a3b8">
          <span style="color:#6b7a99;font-size:11px;text-transform:uppercase;
                       letter-spacing:.8px">Judge Reasoning: </span>
          {judge.get("reasoning","")}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_live_pipeline(question, paragraphs, gold_answer, gold_aliases, embedder, reranker, llm):
    """Run the full pipeline live and yield step updates."""
    all_gold = [gold_answer] + gold_aliases

    # Step 1: Decompose
    yield {"type": "status", "msg": "🧠 Decomposing question..."}
    subqs = decompose_question(llm, question)
    if not subqs:
        yield {"type": "error", "msg": "Decomposition failed — no sub-questions returned."}
        return
    k = len(subqs)
    yield {"type": "decomposed", "subquestions": subqs, "k": k}

    # Step 2: Embed all paragraphs
    yield {"type": "status", "msg": "🔍 Embedding paragraphs..."}
    para_texts = [f"{p.get('title','')}: {p.get('paragraph_text','')}" for p in paragraphs]
    para_embs  = embed_texts(embedder, para_texts)

    # Step 3: k-loop
    intermediate_answers = []
    steps_data = []

    for step_i, raw_sq in enumerate(subqs):
        # Resolve placeholders
        rsq = raw_sq
        for j, a in enumerate(intermediate_answers, 1):
            rsq = rsq.replace(f"[answer_{j}]", a)

        yield {"type": "step_start", "step": step_i+1, "total": k, "sq": rsq}

        # Retrieve
        sq_emb    = embed_texts(embedder, [rsq])[0]
        top_paras = retrieve_top_k(rsq, sq_emb, para_embs, paragraphs, reranker, TOP_K_PARAS)
        context   = "\n\n".join(
            f"{p['paragraph'].get('title','')}: {p['paragraph'].get('paragraph_text','')}"
            for p in top_paras
        )
        yield {"type": "retrieved", "step": step_i+1, "paras": top_paras}

        # Answer — attempt 1
        ans, err = answer_subq(llm, rsq, context, fallback=False)
        attempts = 1
        used_sq  = rsq
        refined  = False
        refined_paras = []

        if err or not ans.strip() or ans.strip().upper() == "NOT FOUND":
            yield {"type": "retry", "step": step_i+1}
            # Attempt 2: fallback prompt
            ans2, err2 = answer_subq(llm, rsq, context, fallback=True)
            attempts = 2
            if not err2 and ans2.strip() and ans2.strip().upper() != "NOT FOUND":
                # Try refinement with partial answer
                refined_sq = refine_question(llm, rsq, ans2)
                if refined_sq != rsq:
                    refined = True
                    ref_emb = embed_texts(embedder, [refined_sq])[0]
                    refined_paras = retrieve_top_k(
                        refined_sq, ref_emb, para_embs, paragraphs, reranker, TOP_K_PARAS
                    )
                    ref_ctx = "\n\n".join(
                        f"{p['paragraph'].get('title','')}: {p['paragraph'].get('paragraph_text','')}"
                        for p in refined_paras
                    )
                    ans3, _ = answer_subq(llm, refined_sq, ref_ctx, fallback=False)
                    if ans3.strip() and ans3.strip().upper() != "NOT FOUND":
                        ans = ans3; used_sq = refined_sq; attempts = 3
                    else:
                        ans = ans2
                else:
                    ans = ans2
            else:
                ans = "NOT FOUND"

        final_paras = refined_paras if refined_paras else top_paras
        intermediate_answers.append(ans if ans else "NOT FOUND")

        step_record = {
            "step": step_i+1,
            "subquestion": raw_sq,
            "resolved_sq": rsq,
            "used_sq": used_sq,
            "refined": refined,
            "attempts": attempts,
            "retrieved_paragraphs": [
                {
                    "title":         p["paragraph"].get("title",""),
                    "text":          p["paragraph"].get("paragraph_text","")[:300],
                    "is_supporting": p["paragraph"].get("is_supporting", False),
                    "rerank_score":  p["rerank_score"],
                    "cosine_score":  p["cosine_score"],
                }
                for p in final_paras
            ],
            "intermediate_answer": intermediate_answers[-1],
        }
        steps_data.append(step_record)
        yield {"type": "step_done", "step": step_i+1, "data": step_record}

    # Final answer
    final_answer = intermediate_answers[-1] if intermediate_answers else "NOT FOUND"
    em  = compute_em(final_answer, all_gold)
    f1  = compute_f1(final_answer, all_gold)

    yield {"type": "status", "msg": "⚖️ Running judge evaluation..."}
    judge_scores = judge_live(llm, question, steps_data, final_answer, gold_answer, gold_aliases)

    result = {
        "id":           "LIVE",
        "hop_type":     "live",
        "question":     question,
        "subquestions": subqs,
        "k":            k,
        "steps":        steps_data,
        "final_answer": final_answer,
        "gold_answer":  gold_answer,
        "gold_aliases": gold_aliases,
        "metrics":      {"em": em, "f1": round(f1,4), "precision": 0.0, "recall": 0.0},
        "judge":        judge_scores,
    }
    yield {"type": "done", "result": result}


# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED LOADING SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def run_loading_screen():
    """
    Show a full animated loading screen with progress bar + step indicators
    while BGE-M3, CrossEncoder, Llama client, and dataset all load.
    Returns (embedder, reranker, llm_client, questions) when done.
    """

    # ── outer layout ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="height:60px"></div>
    <div style="text-align:center">
      <div style="font-size:52px;margin-bottom:12px">🔍</div>
      <div style="font-size:30px;font-weight:800;color:#4a90e2;margin-bottom:8px">
        Adaptive RAG Explorer
      </div>
      <div style="font-size:14px;color:#6b7a99;margin-bottom:32px">
        Initializing models &amp; dataset — this only happens once
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── progress bar ─────────────────────────────────────────────────────────
    prog_bar  = st.progress(0)
    prog_text = st.empty()

    # ── step checklist ────────────────────────────────────────────────────────
    steps_ph = st.empty()

    STEPS = [
        ("📦", "Loading BGE-M3 embedding model (BAAI/bge-m3)…"),
        ("⚡", "Loading CrossEncoder reranker…"),
        ("🤖", "Connecting to Llama 3.3 70B…"),
        ("📚", "Loading MuSiQue question dataset (19,938 Qs)…"),
    ]

    def render_steps(done_count, active_idx=None):
        html = '<div style="max-width:480px;margin:0 auto">'
        for i, (icon, label) in enumerate(STEPS):
            if i < done_count:
                color, indicator = "#4ade80", "✅"
            elif i == active_idx:
                color, indicator = "#fbbf24", "⏳"
            else:
                color, indicator = "#374151", "○"
            html += f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:10px 16px;margin:6px 0;
                        background:{'#1a2e22' if i<done_count else '#1e2535' if i==active_idx else '#161b27'};
                        border:1px solid {'#2d7a4f' if i<done_count else '#876a1a' if i==active_idx else '#2a3045'};
                        border-radius:10px;transition:all .3s">
              <span style="font-size:18px">{indicator}</span>
              <span style="font-size:13px;color:{color}">{icon} {label}</span>
            </div>"""
        html += "</div>"
        steps_ph.markdown(html, unsafe_allow_html=True)

    render_steps(0, active_idx=0)

    # ── Step 1: BGE-M3 ───────────────────────────────────────────────────────
    prog_text.markdown('<p style="text-align:center;color:#fbbf24;font-size:13px">Loading embedding model…</p>',
                       unsafe_allow_html=True)
    prog_bar.progress(5)
    embedder = SentenceTransformer(EMBED_MODEL)
    prog_bar.progress(35)
    render_steps(1, active_idx=1)

    # ── Step 2: CrossEncoder ─────────────────────────────────────────────────
    prog_text.markdown('<p style="text-align:center;color:#fbbf24;font-size:13px">Loading reranker…</p>',
                       unsafe_allow_html=True)
    reranker = CrossEncoder(RERANK_MODEL)
    prog_bar.progress(55)
    render_steps(2, active_idx=2)

    # ── Step 3: Llama client ─────────────────────────────────────────────────
    prog_text.markdown('<p style="text-align:center;color:#fbbf24;font-size:13px">Connecting to Llama 3.3 70B…</p>',
                       unsafe_allow_html=True)
    llm_client = OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)
    prog_bar.progress(65)
    render_steps(3, active_idx=3)

    # ── Step 4: Dataset ──────────────────────────────────────────────────────
    prog_text.markdown('<p style="text-align:center;color:#fbbf24;font-size:13px">Loading MuSiQue dataset…</p>',
                       unsafe_allow_html=True)
    questions = load_questions()
    prog_bar.progress(95)
    render_steps(4)

    # ── Done ─────────────────────────────────────────────────────────────────
    prog_bar.progress(100)
    prog_text.markdown(
        f'<p style="text-align:center;color:#4ade80;font-size:14px;font-weight:700">'
        f'✅ Ready — {len(questions):,} questions loaded</p>',
        unsafe_allow_html=True,
    )
    time.sleep(0.8)

    # cache the heavy objects in session so cache_resource works next rerun
    st.session_state._embedder   = embedder
    st.session_state._reranker   = reranker
    st.session_state._llm_client = llm_client
    st.session_state.models_loaded = True
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(questions):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 10px 0">
          <div style="font-size:22px;font-weight:800;color:#4a90e2">🔍 RAG Explorer</div>
          <div style="font-size:11px;color:#6b7a99;margin-top:4px">Phase 3 · MuSiQue · 19,938 Qs</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Search ───────────────────────────────────────────────────────────
        search = st.text_input("", placeholder="🔎 Search questions...",
                               key="search_box", label_visibility="collapsed")

        # ── Filter by hop type ───────────────────────────────────────────────
        hop_opts   = ["All"] + ["2hop","3hop1","3hop2","4hop1","4hop2","4hop3"]
        hop_filter = st.selectbox("Hop type", hop_opts, key="hop_filter",
                                  label_visibility="collapsed")

        # ── Apply filters ─────────────────────────────────────────────────────
        filtered = questions
        if hop_filter != "All":
            filtered = [q for q in filtered if q["hop_type"] == hop_filter]
        if search.strip():
            sq = search.strip().lower()
            filtered = [q for q in filtered
                        if sq in q["question"].lower() or sq in q["id"].lower()]

        total_filtered = len(filtered)

        # ── Pagination ────────────────────────────────────────────────────────
        total_pages = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
        if "page" not in st.session_state:
            st.session_state.page = 0

        page = max(0, min(st.session_state.page, total_pages - 1))
        st.session_state.page = page

        start      = page * PAGE_SIZE
        page_items = filtered[start : start + PAGE_SIZE]

        st.markdown(
            f'<div class="page-info">{total_filtered:,} questions &nbsp;·&nbsp; page {page+1}/{total_pages}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀", key="prev_btn", disabled=(page == 0), use_container_width=True):
                st.session_state.page = max(0, page - 1)
                st.rerun()
        with c2:
            new_page = st.number_input("", min_value=1, max_value=total_pages,
                                       value=page + 1, key="page_inp",
                                       label_visibility="collapsed")
            if new_page - 1 != page:
                st.session_state.page = new_page - 1
                st.rerun()
        with c3:
            if st.button("▶", key="next_btn", disabled=(page >= total_pages - 1),
                         use_container_width=True):
                st.session_state.page = min(total_pages - 1, page + 1)
                st.rerun()

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        # ── Question list ─────────────────────────────────────────────────────
        for q in page_items:
            qid    = q["id"]
            qtxt   = q["question"][:88]
            hop    = q["hop_type"]
            is_sel = (st.session_state.get("selected_id") == qid)
            icon   = "▶" if is_sel else "○"

            if st.button(
                f"{icon} [{hop}]  {qtxt}…",
                key=f"q_{qid}",
                use_container_width=True,
            ):
                st.session_state.selected_id = qid
                st.session_state.run_result  = None   # clear previous result
                st.rerun()

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── init session state ───────────────────────────────────────────────────
    defaults = {
        "models_loaded": False,
        "selected_id":   None,
        "run_result":    None,
        "page":          0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── LOADING SCREEN ───────────────────────────────────────────────────────
    # Show animated progress until all models + data are in session state
    if not st.session_state.models_loaded:
        run_loading_screen()
        return   # rerun() inside run_loading_screen() handles the transition

    # ── retrieve cached objects from session ─────────────────────────────────
    embedder   = st.session_state._embedder
    reranker   = st.session_state._reranker
    llm_client = st.session_state._llm_client
    questions  = load_questions()   # fast — already cached by @st.cache_data

    if not questions:
        st.error(f"⚠️ Dataset not found. Expected: `{INPUT_JSONL}`")
        st.info("Also tried: same folder as app.py — place `musique_ans_v1.0_train.jsonl` there.")
        return

    # ── sidebar ───────────────────────────────────────────────────────────────
    render_sidebar(questions)

    # ── main content ─────────────────────────────────────────────────────────
    sel_id = st.session_state.selected_id

    # ── WELCOME / overview ────────────────────────────────────────────────────
    if sel_id is None:
        hop_counts = {}
        for q in questions:
            h = q["hop_type"]
            hop_counts[h] = hop_counts.get(h, 0) + 1

        st.markdown("""
        <div style="text-align:center;padding:40px 20px 20px 20px">
          <div style="font-size:44px;margin-bottom:10px">🔍</div>
          <div style="font-size:28px;font-weight:800;color:#4a90e2;margin-bottom:8px">
            Adaptive RAG Explorer
          </div>
          <div style="font-size:14px;color:#6b7a99">
            Phase 3 &nbsp;·&nbsp; Full MuSiQue Dataset &nbsp;·&nbsp;
            Select a question from the sidebar to run the pipeline
          </div>
        </div>
        """, unsafe_allow_html=True)

        stats = [
            ("📊", f"{len(questions):,}", "Questions"),
            ("🤖", "Llama 3.3 70B", "Reader & Judge"),
            ("🔍", "BGE-M3", "Embedder"),
            ("⚡", "CrossEncoder", "Reranker"),
            ("🔗", "2 – 4", "Hop Range"),
        ]
        cols = st.columns(len(stats))
        for col, (icon, val, lbl) in zip(cols, stats):
            col.markdown(f"""
            <div style="text-align:center;background:#161b27;border:1px solid #2a3045;
                        border-radius:12px;padding:16px 8px">
              <div style="font-size:22px">{icon}</div>
              <div style="font-size:20px;font-weight:800;color:#4a90e2">{val}</div>
              <div style="font-size:11px;color:#6b7a99;text-transform:uppercase;
                          letter-spacing:.8px;margin-top:4px">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Hop-Type Distribution</div>',
                    unsafe_allow_html=True)
        hcols = st.columns(len(hop_counts))
        for col, (hop, cnt) in zip(hcols, sorted(hop_counts.items())):
            pct = cnt / len(questions) * 100
            col.markdown(f"""
            <div style="text-align:center;background:#161b27;border:1px solid #2a3045;
                        border-radius:10px;padding:12px 6px">
              {hop_badge(hop)}
              <div style="font-size:18px;font-weight:800;color:#c9d1e0;margin-top:8px">{cnt:,}</div>
              <div style="font-size:11px;color:#6b7a99">{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.info("👈 Click any question in the sidebar — the pipeline will run live and stream results here.")
        return

    # ── FIND SELECTED QUESTION ────────────────────────────────────────────────
    q_entry = next((q for q in questions if q["id"] == sel_id), None)
    if q_entry is None:
        st.error(f"Question `{sel_id}` not found in dataset.")
        return

    question   = q_entry["question"]
    hop_type   = q_entry["hop_type"]
    gold       = q_entry["answer"]
    aliases    = q_entry["answer_aliases"]
    paragraphs = q_entry["paragraphs"]

    # ── QUESTION BUBBLE (always shown) ────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:16px">
      <div style="font-size:28px">🙋</div>
      <div style="flex:1">
        <div style="margin-bottom:6px">
          {hop_badge(hop_type)}
          <span style="font-size:11px;color:#6b7a99;font-family:monospace;margin-left:8px">{sel_id}</span>
        </div>
        <div class="chat-bubble-q">{question}</div>
        <div style="margin-top:6px;font-size:12px;color:#6b7a99">
          Gold answer: <span style="color:#4ade80;font-weight:600">{gold}</span>
          {"&nbsp;·&nbsp;<span style='color:#6b7a99'>aliases: " + ", ".join(aliases[:3]) + "</span>" if aliases else ""}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ALREADY RAN THIS QUESTION → show cached result ────────────────────────
    if (st.session_state.run_result is not None and
            st.session_state.run_result.get("id") == sel_id):
        render_result(st.session_state.run_result)
        if st.button("🔄 Re-run Pipeline", key="rerun_btn"):
            st.session_state.run_result = None
            st.rerun()
        return

    # ── RUN PIPELINE ─────────────────────────────────────────────────────────
    run_col, _ = st.columns([2, 5])
    with run_col:
        run_clicked = st.button("▶ Run Adaptive RAG Pipeline",
                                key="run_pipeline_btn",
                                use_container_width=True)

    if not run_clicked:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#6b7a99">
          <div style="font-size:32px;margin-bottom:8px">⚡</div>
          <div style="font-size:14px">Click <b style="color:#4a90e2">▶ Run Adaptive RAG Pipeline</b> to process this question</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── STREAMING PIPELINE UI ─────────────────────────────────────────────────
    status_ph  = st.empty()
    prog_ph    = st.empty()
    steps_ph   = st.container()

    all_steps = []

    for event in run_live_pipeline(
        question, paragraphs, gold, aliases, embedder, reranker, llm_client
    ):
        etype = event["type"]

        if etype == "status":
            status_ph.markdown(
                f'<div style="color:#fbbf24;font-size:13px;padding:6px 0">{event["msg"]}</div>',
                unsafe_allow_html=True,
            )

        elif etype == "decomposed":
            k    = event["k"]
            subqs = event["subquestions"]
            prog_ph.progress(10)
            status_ph.markdown(
                f'<div style="color:#4ade80;font-size:13px;padding:6px 0">'
                f'✅ Decomposed into <b>{k}</b> sub-questions</div>',
                unsafe_allow_html=True,
            )
            with steps_ph:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:14px;margin:10px 0">
                  <div style="text-align:center;background:#161b27;border:1px solid #2a3045;
                              border-radius:10px;padding:10px 18px">
                    <div style="font-size:10px;color:#6b7a99;text-transform:uppercase;letter-spacing:.8px">Hops (k)</div>
                    <div class="k-badge" style="font-size:28px">{k}</div>
                  </div>
                  <div style="flex:1">
                    <div class="section-header">Decomposed Sub-Questions</div>
                    {"".join(f'<div class="subq-box"><span style="font-size:11px;color:#4b5563;font-weight:700">Q{i}:</span> {sq}</div>' for i, sq in enumerate(subqs,1))}
                  </div>
                </div>
                <div class="section-header" style="margin-top:16px">Step-by-Step RAG Trace</div>
                """, unsafe_allow_html=True)

        elif etype == "step_start":
            pct = 10 + int(70 * (event["step"] - 1) / max(event["total"], 1))
            prog_ph.progress(pct)
            status_ph.markdown(
                f'<div style="color:#fbbf24;font-size:13px;padding:6px 0">'
                f'🔄 Step {event["step"]}/{event["total"]}: {event["sq"][:70]}…</div>',
                unsafe_allow_html=True,
            )

        elif etype == "retry":
            status_ph.markdown(
                f'<div style="color:#f87171;font-size:13px;padding:6px 0">'
                f'⚠️ Step {event["step"]}: NOT FOUND — retrying with fallback…</div>',
                unsafe_allow_html=True,
            )

        elif etype == "step_done":
            all_steps.append(event["data"])
            pct = 10 + int(70 * event["step"] / max(event.get("total", event["step"]), 1))
            prog_ph.progress(pct)
            with steps_ph:
                render_step(event["data"], event["step"], event["data"]["step"])

        elif etype == "done":
            prog_ph.progress(100)
            status_ph.empty()
            res = event["result"]
            st.session_state.run_result = res

            with steps_ph:
                is_nf  = "NOT FOUND" in res["final_answer"].upper()
                fa_cls = "notfound" if is_nf else ""
                fa_icon = "❌" if is_nf else "🤖"
                st.markdown(f"""
                <div class="section-header">Final Answer</div>
                <div style="display:flex;gap:12px;align-items:flex-start">
                  <div style="font-size:22px">{fa_icon}</div>
                  <div class="answer-bubble {fa_cls}" style="font-size:16px;flex:1">
                    {res['final_answer']}
                  </div>
                </div>
                <div class="section-header" style="margin-top:16px">Evaluation Scores</div>
                """, unsafe_allow_html=True)
                render_score_cards(res["metrics"], res["judge"], steps=res["steps"])
                if res["judge"].get("reasoning"):
                    st.markdown(f"""
                    <div style="background:#1e2535;border:1px solid #2e3a55;border-radius:8px;
                                padding:12px 16px;margin-top:8px;font-size:13px;color:#94a3b8">
                      <span style="color:#6b7a99;font-size:11px;text-transform:uppercase;
                                   letter-spacing:.8px">Judge Reasoning: </span>
                      {res['judge']['reasoning']}
                    </div>""", unsafe_allow_html=True)

        elif etype == "error":
            st.error(event["msg"])


if __name__ == "__main__":
    main()