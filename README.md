# Research Report Agent

A tool-calling agent that answers questions over uploaded PDFs, built on a hybrid-retrieval RAG service.
Everything runs locally (Ollama + a 7B model). The project includes an evaluation harness that measures the agent
against a single-search baseline.

**Part 2 of a two-part project.** Part 1 ([Hybrid RAG + failure analysis](https://github.com/YannamsettiVenkataLakshmi/Hybrid_RAG)) built the retrieval pipeline and
found that the 3B generation model, not retrieval, was the bottleneck. This repo takes the retrieval stage, fixes and
extends it into a service (v2), and puts an agent on top of it.

## What changed from v1

| Area | v1 ([Hybrid_RAG](https://github.com/YannamsettiVenkataLakshmi/Hybrid_RAG)) | v2 (this repo) |
|---|---|---|
| Documents | one active document, saved as `data/current_document.pdf`; a new upload replaces it | many documents, each saved as `data/<doc_id>.pdf` and listed in a registry (`indexes/documents.json`: doc id, filename, path, file hash, session id, preview) |
| Isolation | none | `session_id` and `doc_id` on every chunk; retrieval is limited to the session, with optional `doc_ids` (checked against the session) |
| Upload | `POST /index` with a PDF | `POST /index` with a PDF and a `session_id`; a duplicate is detected by hash within the session |
| FAISS index | rebuilt for the uploaded PDF | new chunks are appended to the existing index |
| BM25 index | built for the current document | rebuilt over all chunks when a document is added |
| Configuration change (embedding model, chunk size or overlap) | rebuild the current document's index | rebuild the whole collection from the registry (implemented, not covered by tests) |
| Dense and BM25 search | one document, no filtering | filtered by session and, optionally, by document; BM25 drops zero-score chunks |
| Fusion (RRF) | rank fusion of BM25 and dense | **fixed:** the service's version ignored the dense results; chunks found by both retrievers are now merged |
| Results returned | top 4 | top 5 for one scope; for several explicit documents, 3 per document, interleaved |
| Endpoints | `/health`, `/index`, `/retrieve` | adds `GET /documents` (documents with previews) |
| Consumer | Streamlit app, SciFact claim verification (Qwen 2.5 3B) | tool-calling agent with an evaluation harness |

Retrieval quality was benchmarked in v1 on SciFact (Recall@5: BM25 0.60, dense 0.69, hybrid 0.72, hybrid + reranker 0.74).
The agent evaluation below is a different measurement, on question answering over three documents.
Sessions separate documents by metadata only. `session_id` is not authentication.

## Results

Task success on questions over three documents (a research paper, a course curriculum, a lecture deck), local `qwen2.5:7b`.

**Held-out set: 16 questions written after the agent was designed**

| Mode | Passed | Success | Searched the right documents | Tokens / task |
|---|---|---|---|---|
| baseline: one search, no strategy | 6/16 | 0.38 | 0.19 | ~1,200 |
| agentic_v1: tool loop + prompt rules | 14/16 | 0.88 | 0.88 | ~2,800 |
| agentic: stricter prompt | 13/16 | 0.81 | 0.94 | ~3,000 |
| guarded: stricter prompt + "search before answering" rule | 14/16 | 0.88 | 1.00 | ~3,200 |

**Development set: 31 tasks used while designing the agent (treat as optimistic)**

| Mode | Passed | Success | Searched the right documents | Tokens / task |
|---|---|---|---|---|
| baseline | 17/31 | 0.55 | 0.26 | ~1,400 |
| agentic_v1 | 21/31 | 0.68 | 0.74 | ~3,000 |
| agentic | 23/31 | 0.74 | 0.77 | ~3,200 |
| guarded | 28/31 | 0.90 | 0.94 | ~3,600 |

**What the results support**
- On both sets the agent clearly beats the baseline. The gain is almost entirely on single-document questions
  (held-out: 3/11 vs 9-10/11). It comes from choosing the right document and then searching it.
- The cost is about 2.5x the tokens of the baseline.
- The "search before answering" rule fixed the specific failure it targeted (the model answering from a document
  preview without searching). On the held-out set the modes are within one question of each other, so this README
  does **not** claim the guard or the stricter prompt improves accuracy.

**What they do not support**
- 16 and 31 questions are small samples, over three documents, with keyword-based scoring and one local model.
- The development set was used to tune prompts and the guard; the held-out set is the more honest number.
- The scoring rule for refusals ("the documents don't say") was widened after the first held-out run. It affects the
  two refusal tasks only.

## Known limitations

- **Numeric facts buried in dense paragraphs can be missed.** Example: "How many images were in each training set?"
  The query matched test-set text instead of the training-set sentence, and the answer was wrong. Fails in every mode.
- **Ambiguous questions** ("Who are the authors?") are answered for one document and the others are ignored. Fails in every mode.
- **The retry rule is rarely followed.** The prompt asks the model to rewrite and retry failed searches, but on
  single-document questions the 7B model averages exactly one search. The gains above come from document choice and
  forced searching, not from retrying.
- Scoring is a keyword check, not a human or model judge.
- The retrieval models run on the Apple GPU next to Ollama. A GPU out-of-memory warning appeared once in the API log during development, so in principle a search could be degraded without an error. Repeated runs under the same conditions gave identical results, but GPU memory pressure is an uncontrolled factor.
- The "rebuild everything when the index configuration changes" path is implemented but not tested or evaluated.
- One early held-out run (8/16) was made under an unrecorded system state and is excluded. Four later runs were identical.
  Runs where Ollama or the RAG API was down are detected by the harness and refused or skipped.

## Architecture

```
 question
    |
    v
 +----------------+   tool calls    +----------------------------------------+
 |  Agent loop    | --------------> |  RAG service (FastAPI)                 |
 |  qwen2.5:7b    | <-------------- |   GET  /documents   list + previews    |
 |  via Ollama    |    results      |   POST /retrieve    BM25 + BGE-M3      |
 +----------------+                 |                     -> RRF -> reranker |
    |  also: calculator              |   POST /index       upload a PDF       |
    v                                +----------------------------------------+
 answer + full trajectory log
```

**RAG service (`rag_service/`)**: hybrid retrieval (BM25 and BGE-M3 dense search fused with reciprocal rank fusion, then a
cross-encoder reranker). Documents are scoped by `session_id`; an optional `doc_ids` list narrows further. A request for
several documents retrieves per document and interleaves the results, so one document cannot crowd out the others.

**Agent (`agent/`)**: a hand-written loop (no framework) with three tools: `list_documents`, `retrieve`, `calculator`.
The agent refers to documents by number or filename. The tool layer adds the session and maps names to ids, so a small model never handles ids.
Errors go back to the model as text so it can correct itself. There is a step limit, repeated-call detection and a forced final answer.

| Mode | Description |
|---|---|
| `baseline` | one `retrieve` allowed, minimal prompt |
| `agentic_v1` | up to 3 searches, prompt rules for scoping, comparison and retry |
| `agentic` | stricter prompt ("always search before answering") |
| `guarded` | `agentic` prompt + a loop rule: if the model tries to answer before searching, it is sent back once |

## Evaluation harness

- Tasks are JSONL: question, expected keyword groups, expected tools, expected documents (`agent/tasks.jsonl`, `agent/tasks_heldout.jsonl`).
- **Success** needs the expected keywords **and** an actual search when the task needs one. **Tool accuracy** needs the expected tools and the expected documents.
- Every step of every run is logged (tool, arguments, status, retrieved passages, tokens).
- Every run records its provenance: git commit, uncommitted-changes flag, document-list hash, model, settings.
- `failures.py` splits failures into retrieval misses and generation misses; `variance.py` compares repeated runs; `probe.py` shows exactly what the retriever returns for a query.

The evaluation documents are not included (copyright). To reproduce, upload your own PDFs and write tasks in the same format.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r rag_service/requirements.txt
python -m spacy download en_core_web_sm
pip install -r agent/requirements.txt
ollama pull qwen2.5:7b                       # and keep Ollama running
```
```bash
# terminal 1: the RAG service (start it from rag_service/)
cd rag_service && uvicorn api:app --port 8000

# upload a PDF into a session
curl -F "file=@my.pdf" -F "session_id=chat_1" http://127.0.0.1:8000/index
```
```bash
# terminal 2: ask the agent
python agent/agent.py "Compare the curriculum and the lunar landing paper" --session chat_1 --mode guarded

# evaluate and compare
python agent/eval_agent.py --mode all --tasks agent/tasks_heldout.jsonl --out agent/results_heldout
python agent/compare_runs.py --dir results_heldout --tasks tasks_heldout.jsonl
pytest agent/tests                           # offline tests, no Ollama or API needed
```
`rag_service/` also keeps the Part 1 chat flow (`src/demo_chat.py`): query rewriter, a Gemini query router (it replaced a local Qwen router that chose "direct answer" for document questions), deterministic filename-based document matching (it replaced an LLM document selector), one search, then a Qwen answer. The agent does not use it and it is not part of the evaluation; the `baseline` mode is the closest equivalent (one search, then answer). It needs `GOOGLE_API_KEY` in a `.env` file.

## Design decisions worth knowing

- **Dropped the LLM document selector.** A small model choosing documents from filenames was unreliable. Scoping is now deterministic (session plus optional ids), and the agent chooses from `list_documents`, which shows a preview of each document.
- **The retrieval tool returns passages, not verdicts.** Part 1 showed the small model was the weak link in verdict generation.
- **Fixes found along the way:** the fusion step originally ignored dense results; BM25 padded results with zero-score chunks; empty `doc_ids` could search every document.

## Status and next steps
- Held-out comparison against a stronger planner model (to separate "7B model limit" from "instruction limit").
- Fixes for the two consistent failures (numeric-fact retrieval, ambiguity coverage), tested on fresh questions.
- Docker for the two services.