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
import time
from collections import defaultdict

from agent import Agent, OllamaClient
from tools import Tools

REFUSALS = ["could not find", "couldn't find", "cannot find", "can't find", "unable to find",
            "not contain", "no information", "not mentioned", "does not provide", "not provide",
            "not specify", "does not state", "not stated", "no mention", "not included",
            "isn't mentioned", "not available", "not described", "does not describe",
            "not discuss", "does not say", "no relevant"]


def load_tasks(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def score_task(task, traj):
    answer = (traj.get("final_answer") or "").lower()

    if task.get("expect_refusal"):
        success = any(m in answer for m in REFUSALS)
    else:
        groups = task.get("must_contain_groups") or []
        success = bool(answer) and all(any(k.lower() in answer for k in g) for g in groups)

    ok_calls = [tc for s in traj["steps"] for tc in s["tool_calls"] if tc["status"] == "ok"]
    called = {tc["name"] for tc in ok_calls}
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
        agent = Agent(Tools(args.rag_url, args.session), OllamaClient(args.model, num_ctx=args.num_ctx),
                      mode=mode, max_steps=args.max_steps)
        traj = agent.run(task["question"])
        traj["task_id"] = task["id"]
        row = score_task(task, traj)
        rows.append(row)
        trajs.append(traj)
        print(f"[{mode}] {i}/{len(tasks)} {task['id']:<8} success={row['success']!s:<5} "
              f"tools={row['tool_accuracy']!s:<5} steps={row['steps']} status={row['status']}")
    return rows, trajs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--session", default="chat_1")
    ap.add_argument("--rag-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--mode", choices=["agentic", "baseline", "both"], default="both")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    modes = ["baseline", "agentic"] if args.mode == "both" else [args.mode]

    summaries = {}
    for mode in modes:
        rows, trajs = run_mode(mode, tasks, args)
        summaries[mode] = summarize(rows)
        tag = f"{mode}_{args.model.replace(':', '-')}_{stamp}"
        with open(os.path.join(args.out, f"{tag}_trajectories.jsonl"), "w") as f:
            for t in trajs:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, f"{tag}_scores.json"), "w") as f:
            json.dump({"summary": summaries[mode], "rows": rows}, f, indent=2)
        print_summary(f"{mode.upper()} ({args.model})", summaries[mode])

    if len(modes) == 2:
        b, a = summaries["baseline"]["overall"], summaries["agentic"]["overall"]
        print(f"\nBaseline -> Agentic: success {b['success']} -> {a['success']}, "
              f"tool accuracy {b['tool_acc']} -> {a['tool_acc']}, "
              f"avg steps {b['avg_steps']} -> {a['avg_steps']}")


if __name__ == "__main__":
    main()
