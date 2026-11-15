"""
Evaluation harness for the agent.

    python eval_agent.py --tasks tasks.jsonl --session chat_1 --mode both --model qwen2.5:7b

Per task it records the full trajectory, then scores:
  success        keyword check on the final answer (cheap and deterministic, but crude)
  tool_accuracy  called the expected tools AND searched the expected document(s)
  steps          number of LLM calls
  loop           hit the step limit or repeated an identical call

Task format (one JSON object per line in tasks.jsonl):
  id, type, question,
  must_contain_groups : [[kw, kw], [kw]]  every group needs at least one keyword in the answer
  expect_refusal      : true if the right behaviour is "I could not find it"
  expected_tools      : ["retrieve"]
  expected_docs       : ["lunar"]  filename substrings the agent should have searched
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import requests

from agent import Agent, OllamaClient
from tools import Tools

REFUSALS = ["could not find", "couldn't find", "cannot find", "can't find", "unable to find",
            "not contain", "no information", "not mention", "does not provide", "not provide",
            "not specify", "does not state", "not stated", "no mention", "not included",
            "not available", "not described", "does not describe", "not discuss", "does not say",
            "no relevant", "not recommend", "not give", "not list", "no specific", "no explicit",
            "no recommendation", "not explicitly", "not offer",
            "n't contain", "n't provide", "n't mention", "n't specify", "n't state", "n't recommend",
            "n't give", "n't say", "n't list", "n't describe", "n't discuss", "n't offer"]


HERE = os.path.dirname(os.path.abspath(__file__))


def collect_provenance(args, tasks_path):
    """Record the state of the system a run was made against, so differences between runs can be explained."""
    import hashlib
    import platform
    import subprocess

    def git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=HERE, stderr=subprocess.DEVNULL,
                                           text=True, timeout=5).strip()
        except Exception:
            return None

    status = git("status", "--porcelain")
    prov = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model, "num_ctx": args.num_ctx, "max_steps": args.max_steps,
        "tasks_file": os.path.basename(tasks_path),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
    }
    try:
        prov["tasks_sha1"] = hashlib.sha1(open(tasks_path, "rb").read()).hexdigest()[:10]
    except Exception:
        prov["tasks_sha1"] = None
    try:
        r = requests.get(f"{args.rag_url.rstrip('/')}/documents", params={"session_id": args.session}, timeout=10)
        docs = r.json()["documents"]
        prov["n_documents"] = len(docs)
        prov["docs_hash"] = hashlib.sha1(json.dumps(sorted((d["filename"], d.get("preview", "")) for d in docs)).encode()).hexdigest()[:10]
    except Exception:
        prov["n_documents"], prov["docs_hash"] = None, None
    return prov


def invalid_run(statuses):
    """A run where at least half the tasks hit a model connection error says nothing about the agent."""
    statuses = list(statuses)
    return bool(statuses) and sum(s == "llm_error" for s in statuses) * 2 >= len(statuses)


def rag_broken(trajs):
    """True if at least half of all retrieve calls failed because the RAG service itself errored."""
    calls = [tc for t in trajs for s in t["steps"] for tc in s["tool_calls"] if tc["name"] == "retrieve"]
    bad = [tc for tc in calls if tc["status"] == "error" and "Retrieval service error" in tc.get("result_preview", "")]
    return len(calls) >= 3 and len(bad) * 2 >= len(calls)


def rows_broken(rows):
    """Scores-only check: most tasks hit tool errors and never completed a single search."""
    return bool(rows) and sum(1 for r in rows if r.get("errors", 0) > 0 and r.get("retrievals", 0) == 0) * 2 >= len(rows)


def no_search_run(rows):
    """True if the agent almost never completed a search on tasks that need one: the signature of a broken setup."""
    needs = [r for r in rows if r.get("type") != "calculator"]
    return len(needs) >= 5 and sum(1 for r in needs if r.get("retrievals", 0) == 0) >= 0.8 * len(needs)


def preflight(args):
    """Stop BEFORE a long run if Ollama or the RAG API cannot be reached."""
    host = args.ollama_host.rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as exc:
        sys.exit(f"STOP: cannot reach Ollama at {host} ({type(exc).__name__}).\n"
                 "Start the Ollama app (or run `ollama serve`) and try again.")
    if args.model not in names and f"{args.model}:latest" not in names:
        sys.exit(f"STOP: model '{args.model}' is not installed in Ollama.\n"
                 f"Installed: {', '.join(names) or 'none'}\nRun: ollama pull {args.model}")
    try:
        r = requests.get(f"{args.rag_url.rstrip('/')}/documents", params={"session_id": args.session}, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        sys.exit(f"STOP: the RAG API is not answering at {args.rag_url} ({type(exc).__name__}).\n"
                 "Start it with:  cd rag_service && uvicorn api:app --port 8000")
    try:
        r = requests.post(f"{args.rag_url.rstrip('/')}/retrieve", json={"query": "test", "session_id": args.session}, timeout=120)
    except Exception as exc:
        sys.exit(f"STOP: a test search against the RAG API failed ({type(exc).__name__}).\n"
                 "Is it running and fully loaded?  cd rag_service && uvicorn api:app --port 8000")
    if r.status_code != 200:
        sys.exit(f"STOP: the RAG API lists documents but /retrieve returned HTTP {r.status_code}: {r.text[:200]}\n"
                 "Look at the terminal where uvicorn is running for the error message.")


def load_tasks(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def score_task(task, traj):
    answer = (traj.get("final_answer") or "").lower().replace("\u2019", "'")

    if task.get("expect_refusal"):
        success = any(m in answer for m in REFUSALS)
    else:
        groups = task.get("must_contain_groups") or []
        success = bool(answer) and all(any(k.lower() in answer for k in g) for g in groups)

    ok_calls = [tc for s in traj["steps"] for tc in s["tool_calls"] if tc["status"] == "ok"]
    called = {tc["name"] for tc in ok_calls}

    # An answer only counts if the agent actually searched when the task needs retrieval
    # (otherwise it may just be answering from memory and matching keywords by luck).
    if "retrieve" in task.get("expected_tools", []) and "retrieve" not in called:
        success = False
    tools_ok = all(t in called for t in task.get("expected_tools", []))

    docs_ok = True
    expected = [d.lower() for d in task.get("expected_docs", [])]
    if expected:
        searched = set()
        for tc in ok_calls:
            ds = tc["meta"].get("docs_searched")
            if tc["name"] == "retrieve" and isinstance(ds, list):
                searched.update(n.lower() for n in ds)
        covered = all(any(e in n for n in searched) for e in expected)
        extra = [n for n in searched if not any(e in n for e in expected)]
        docs_ok = covered and (len(expected) > 1 or not extra)

    t = traj["totals"]
    return {
        "id": task["id"], "type": task.get("type", "other"),
        "success": success, "tool_accuracy": tools_ok and docs_ok,
        "steps": t["steps"], "tool_calls": t["tool_calls"], "retrievals": t["retrievals"],
        "errors": t["errors"], "loop": traj["status"] == "forced_final" or t["repeated"] > 0,
        "tokens": t["prompt_tokens"] + t["completion_tokens"], "latency_s": t["latency_s"],
        "status": traj["status"],
    }


def summarize(rows):
    def agg(rs):
        n = len(rs)
        mean = lambda k: round(sum(r[k] for r in rs) / n, 2)
        return {"n": n, "success": mean("success"), "tool_acc": mean("tool_accuracy"),
                "avg_steps": mean("steps"), "avg_retrievals": mean("retrievals"),
                "loop_rate": mean("loop"), "avg_tokens": round(sum(r["tokens"] for r in rs) / n),
                "avg_latency_s": mean("latency_s")}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": agg(rows), "by_type": {k: agg(v) for k, v in sorted(by_type.items())}}


def print_summary(label, s):
    print(f"\n=== {label} ===")
    head = f"{'group':<14}{'n':>3}{'success':>9}{'tool_acc':>9}{'steps':>7}{'retr.':>7}{'loops':>7}{'tokens':>8}{'sec':>7}"
    print(head)
    for name, a in [("OVERALL", s["overall"])] + list(s["by_type"].items()):
        print(f"{name:<14}{a['n']:>3}{a['success']:>9}{a['tool_acc']:>9}{a['avg_steps']:>7}"
              f"{a['avg_retrievals']:>7}{a['loop_rate']:>7}{a['avg_tokens']:>8}{a['avg_latency_s']:>7}")


def run_mode(mode, tasks, args):
    rows, trajs = [], []
    for i, task in enumerate(tasks, 1):
        agent = Agent(Tools(args.rag_url, args.session), OllamaClient(args.model, host=args.ollama_host, num_ctx=args.num_ctx),
                      mode=mode, max_steps=args.max_steps)
        traj = agent.run(task["question"])
        traj["task_id"] = task["id"]
        row = score_task(task, traj)
        rows.append(row)
        trajs.append(traj)
        print(f"[{mode}] {i}/{len(tasks)} {task['id']:<8} success={row['success']!s:<5} "
              f"tools={row['tool_accuracy']!s:<5} steps={row['steps']} status={row['status']}")
        if traj["status"] == "llm_error":
            print(f"    LLM ERROR: {str(traj.get('error'))[:160]}")
        if i == 3 and rag_broken(trajs):
            sys.exit("STOP: the search calls to the RAG API are failing. Check the terminal where uvicorn is running. Nothing was saved.")
        if i == 3 and all(t["status"] == "llm_error" for t in trajs):
            sys.exit("STOP: the first 3 tasks all failed to reach the model. Check that Ollama is running. Nothing was saved.")
    return rows, trajs


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))   # always relative to this file, not to where you run it from
    ap.add_argument("--tasks", default=os.path.join(here, "tasks.jsonl"))
    ap.add_argument("--session", default="chat_1")
    ap.add_argument("--rag-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--mode", choices=["agentic", "agentic_v1", "baseline", "guarded", "both", "all"], default="both")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(here, "results"))
    ap.add_argument("--allow-no-search", action="store_true", help="save the run even if the agent almost never searched")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    args = ap.parse_args()

    preflight(args)
    tasks = load_tasks(args.tasks)
    prov = collect_provenance(args, args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    modes = {"both": ["baseline", "agentic"],
             "all": ["baseline", "agentic_v1", "agentic"]}.get(args.mode, [args.mode])

    summaries = {}
    for mode in modes:
        rows, trajs = run_mode(mode, tasks, args)
        if invalid_run(t["status"] for t in trajs) or rag_broken(trajs):
            sys.exit(f"STOP: mode '{mode}' ran into model or RAG API errors on at least half of its calls. Nothing was saved for it.")
        if not args.allow_no_search and no_search_run(rows):
            sys.exit(f"STOP: in mode '{mode}' the agent almost never completed a search. That usually means a setup problem "
                     "(Ollama still loading, RAG API errors). Nothing was saved. Use --allow-no-search to save it anyway.")
        summaries[mode] = summarize(rows)
        tag = f"{mode}_{args.model.replace(':', '-')}_{stamp}"
        with open(os.path.join(args.out, f"{tag}_trajectories.jsonl"), "w") as f:
            for t in trajs:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, f"{tag}_scores.json"), "w") as f:
            json.dump({"summary": summaries[mode], "rows": rows, "provenance": prov}, f, indent=2)
        print_summary(f"{mode.upper()} ({args.model})", summaries[mode])

    if len(modes) > 1:
        print("\n=== COMPARISON (overall) ===")
        print(f"{'mode':<12}{'success':>9}{'tool_acc':>10}{'avg_steps':>11}{'loops':>7}")
        for m in modes:
            o = summaries[m]["overall"]
            print(f"{m:<12}{o['success']:>9}{o['tool_acc']:>10}{o['avg_steps']:>11}{o['loop_rate']:>7}")


if __name__ == "__main__":
    main()
