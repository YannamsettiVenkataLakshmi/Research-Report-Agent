"""
Explain WHY each task failed.

    python agent/failures.py --dir results_heldout --tasks tasks_heldout.jsonl --mode guarded
    python agent/failures.py                                   # the main 31 tasks, mode guarded

For each failed task it shows the queries the agent used, whether the expected words were in the
passages it retrieved, and the final answer. Then it counts:
    RETRIEVAL MISS   the expected words were in NO retrieved passage
    GENERATION MISS  the passages had the words, but the answer was still wrong
It re-scores with the CURRENT scoring rules, so no re-run is needed after a scoring fix.
(Needs trajectories recorded with full passages: run the eval once with this version.)
"""
import argparse
import glob
import json
import os
import re
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from eval_agent import score_task   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results")
ap.add_argument("--tasks", default="tasks.jsonl")
ap.add_argument("--mode", default="guarded")
ap.add_argument("--all", action="store_true", help="also show tasks that passed")
a = ap.parse_args()


def norm(s):
    s = s.lower()
    for x, y in {"\ufb01": "fi", "\ufb02": "fl", "\u2019": "'", "\u2013": "-", "\u2212": "-"}.items():
        s = s.replace(x, y)
    return re.sub(r"\s+", " ", s)


tasks = {}
for line in open(os.path.join(here, a.tasks)):
    if line.strip():
        t = json.loads(line)
        tasks[t["id"]] = t

files = []
for f in glob.glob(os.path.join(here, a.dir, f"{a.mode}_*_trajectories.jsonl")):
    name = os.path.basename(f)
    if a.mode == "agentic" and name.startswith("agentic_v1_"):
        continue
    files.append(f)
if not files:
    sys.exit(f"No '{a.mode}' trajectories in agent/{a.dir}. Run eval_agent.py first.")
path = max(files, key=os.path.getmtime)
trajs = [json.loads(l) for l in open(path) if l.strip()]
print(f"FILE: {os.path.basename(path)}  ({len(trajs)} tasks)\n")

counts = {"RETRIEVAL MISS": 0, "GENERATION MISS": 0, "OTHER": 0, "UNKNOWN": 0}
passed = 0
for traj in trajs:
    task = tasks.get(traj.get("task_id"))
    if not task:
        continue
    row = score_task(task, traj)
    if row["success"]:
        passed += 1
        if not a.all:
            continue

    calls = [tc for s in traj["steps"] for tc in s["tool_calls"] if tc["name"] == "retrieve" and tc["status"] == "ok"]
    has_full = any("result_full" in tc for tc in calls)
    text = norm(" ".join(tc.get("result_full", "") for tc in calls))
    groups = task.get("must_contain_groups") or []

    if row["success"]:
        verdict = "PASSED"
    elif task.get("expect_refusal"):
        verdict, counts["OTHER"] = "NOT A RECOGNISED REFUSAL (read the answer)", counts["OTHER"] + 1
    elif "calculator" in task.get("expected_tools", []):
        verdict, counts["OTHER"] = "CALCULATION TASK (read the answer)", counts["OTHER"] + 1
    elif not calls:
        verdict, counts["OTHER"] = "NEVER SEARCHED", counts["OTHER"] + 1
    elif not has_full:
        verdict, counts["UNKNOWN"] = "UNKNOWN (re-run the eval so full passages are saved)", counts["UNKNOWN"] + 1
    elif all(any(k.lower() in text for k in g) for g in groups):
        verdict, counts["GENERATION MISS"] = "GENERATION MISS (the passages contained the words)", counts["GENERATION MISS"] + 1
    else:
        verdict, counts["RETRIEVAL MISS"] = "RETRIEVAL MISS (expected words in NO retrieved passage)", counts["RETRIEVAL MISS"] + 1

    print("=" * 78)
    print(f"{task['id']} [{task['type']}]  {verdict}")
    print("Q:", task["question"])
    print("expected:", groups if groups else "(a refusal)")
    for tc in calls:
        print(f"  searched {tc['meta'].get('docs_searched')} with query: {tc['arguments'].get('query')!r}")
        if has_full and groups:
            found = [[k for k in g if k.lower() in norm(tc.get('result_full', ''))] for g in groups]
            print("    expected words present in this result:", found)
    print("A:", (traj.get("final_answer") or "").replace("\n", " ")[:300])

print("\n" + "=" * 78)
print(f"Passed (current scoring): {passed}/{len(trajs)}")
print("Failures:", {k: v for k, v in counts.items() if v})
