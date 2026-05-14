# Adaptive RAG — Multi-Hop Question Answering over MuSiQue

An end-to-end **Adaptive Retrieval-Augmented Generation** pipeline for multi-hop question answering on the [MuSiQue](https://arxiv.org/abs/2108.00573) dataset. The pipeline decomposes complex questions into sequential sub-questions, retrieves relevant paragraphs at each step, and synthesises a final answer using a Llama 3.3 70B reader model.

## Important Notes:

`phase1/musique_ans_v1.0_train.jsonl` is not included due to size (~241MB).  
Download it from the [MuSiQue dataset](musique_ans_v1.0_train) and place it in `phase1/`.

`phase3/phase3_results_all.json` (~89MB) is excluded from the repo.  

---

## Folder Structure

```
nlp_final_project_zip/
│
├── Adaptive_RAG_Final_Report.docx       # Final comprehensive project report
├── adaptiveRAG_arc.pdf                  # Pipeline architecture diagram
│
├── phase1/                              # 100 questions, 2-hop, retrieval config ablation
│   ├── adaptive_Rag.py                  # Main pipeline (Phase 1)
│   ├── llm_classifier.py               # LLM-based question decomposer
│   ├── sample_data.py                  # Data sampling script
│   ├── sampled_100_2hop.json           # 100 sampled 2-hop questions
│   ├── decomposed_100_2hop.json        # Decomposed sub-questions
│   ├── pipeline_results_all-MiniLM-L6-v2.json      # Results: Config 1
│   ├── pipeline_results_multi-qa-mpnet-base-dot-v1.json  # Results: Config 2
│   ├── pipeline_results_bge_reranker.json           # Results: Config 4 (best)
│   ├── musique_ans_v1.0_train.jsonl    # MuSiQue training data
│   └── Phase1_Report.docx             # Phase 1 report
│
├── phase2/                              # 900 questions, 2–4 hop, prompt & reader ablation
│   ├── adaptive_Rag.py                  # Main pipeline (Phase 2, Prompt 1 baseline)
│   ├── adaptive_Rag_CoN.py             # Chain-of-Note reader variant
│   ├── adaptive_Rag_copy.py            # Working copy
│   ├── adaptive_Rag_prompt3.py         # Prompt 3 (Chain-of-Thought) variant
│   ├── llm_classifier.py               # Decomposer (Prompt 1 & 2)
│   ├── llm_classifier_copy.py          # Decomposer working copy
│   ├── sample_data.py                  # Data sampling script (900 questions)
│   ├── summary.py                      # Results aggregation script
│   ├── run_clean.sh                    # SLURM job runner
│   ├── sampled_900_all.json            # 900 sampled questions (all hop types)
│   ├── decomposed_900_all.json         # Decomposed (Prompt 1)
│   ├── decomposed_900_all_prompt2.json # Decomposed (Prompt 2)
│   ├── decomposed_900_all_prompt3.json # Decomposed (Prompt 3)
│   ├── pipeline_results_v2.json        # Results: Prompt 1
│   ├── pipeline_results_v2_prompt2.json # Results: Prompt 2
│   ├── pipeline_results_v2_prompt3.json # Results: Prompt 3
│   ├── pipeline_results_CoN.json       # Results: Chain-of-Note
│   ├── pipeline_summary_v2.txt         # Summary: Prompt 1
│   ├── pipeline_summary_v2_prompt2.txt # Summary: Prompt 2
│   ├── pipeline_summary_v2_prompt3.txt # Summary: Prompt 3
│   ├── pipeline_summary_CoN.txt        # Summary: Chain-of-Note
│   └── Phase2_Report_v2.docx          # Phase 2 report
│
└── phase3/                              # Full MuSiQue dataset (19,938 questions) + UI
    ├── main_ui.py                       # ★ Streamlit interactive UI
    ├── adaptive_rag_pipeline.py         # Full pipeline (Phase 3)
    ├── adaptive_rag_full.py             # Alternate full pipeline script
    ├── merge_result.py                  # Merges SLURM batch results
    ├── analyze_steps.py                 # Step-level analysis
    ├── script.py                        # Utility script
    ├── run_pipeline.sh                  # SLURM batch job runner
    ├── phase3_results_all.json          # Full results (19,938 questions)
    ├── phase3_summary_all.txt           # Summary statistics
    └── Phase3_Report.pdf               # Phase 3 report
```

---

## Phase Summaries

### Phase 1 — Retrieval Configuration Ablation (100 Questions, 2-hop)
Establishes the baseline pipeline on 100 randomly sampled 2-hop questions from MuSiQue. The core architecture decomposes each question into ordered sub-questions using Llama 3.3 70B, retrieves the top-2 paragraphs per sub-question from a 20-paragraph local corpus, and synthesises a final answer step-by-step. Five retrieval configurations are benchmarked — from lightweight dense cosine similarity (all-MiniLM-L6-v2) to hybrid BM25+dense and cross-encoder reranking. **BGE-M3 + CrossEncoder** emerges as the best retriever and is carried forward to all subsequent phases.

### Phase 2 — Prompt & Reader Ablation (900 Questions, 2–4 hop)
Scales to 900 questions across all six MuSiQue compositional subtypes (2hop through 4hop3), locking in the Phase 1 best retriever. Four pipeline variants are evaluated: **Prompt 1** (Phase 1 baseline), **Prompt 2** (verbose decomposition with nested-entity guidance), **Prompt 3** (structured chain-of-thought with 7-step reasoning), and **Chain-of-Note** (reader writes a reading note before answering). A separate Qwen3.5-27B judge model replaces the self-judge setup from Phase 1 to eliminate self-consistency bias. The key addition is a **NOT FOUND retry loop** — the adaptive core — which makes up to three retrieval attempts before declaring a sub-question unanswerable. Prompt 1 (Baseline) achieves the best balance of EM/F1 and is selected for Phase 3.

### Phase 3 — Full Dataset Scale + Interactive UI (19,938 Questions)
Runs the Phase 2 best configuration (Prompt 1) over the complete answerable MuSiQue training set of 19,938 questions. A batch SLURM execution system divides the dataset into index-range slices and processes them in parallel to overcome the 24-hour HPC wall-time limit. Final results: **overall EM 0.450, F1 0.561** — above Phase 2 and the original Adaptive-RAG paper's MuSiQue benchmark. A **Streamlit UI** (`main_ui.py`) is added for interactive question answering using the full pipeline.

---

## Running the UI

```bash
cd phase3
streamlit run main_ui.py
```

---

## Environment Setup

Create a `.env` file in the project root with the following keys:

```env
UTSA_API_KEY=api_key
UTSA_BASE_URL=http://10.246.100.230/v1
UTSA_MODEL=llama-3.3-70b-instruct-awq

JUDGE_API_KEY=api_key
JUDGE_BASE_URL=http://10.246.100.230/v1
JUDGE_MODEL=llama-3.3-70b-instruct-awq
```

| Variable | Description |
|---|---|
| `UTSA_API_KEY` | API key for the UTSA reader/decomposer LLM |
| `UTSA_BASE_URL` | Base URL for the UTSA LLM endpoint |
| `UTSA_MODEL` | Model name for the reader and decomposer |
| `JUDGE_API_KEY` | API key for the judge LLM |
| `JUDGE_BASE_URL` | Base URL for the judge LLM endpoint |
| `JUDGE_MODEL` | Model name used for answer evaluation |

---

## Key Results Summary

| Phase | Questions | EM | F1 |
|---|---|---|---|
| Phase 3 (Full Dataset) | 19,938 (2–4 hop) | **0.450** | **0.561** |


