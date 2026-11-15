"""
How stable are the results? Lists every saved run of a mode and shows which tasks flip between runs.
All runs are re-scored with the CURRENT scoring rules, so they are comparable.

    python agent/variance.py --dir results_heldout --tasks tasks_heldout.jsonl --mode guarded
    python agent/variance.py                                   # main 31 tasks, mode guarded
"""
import argparse
import glob
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from eval_agent import score_task   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results")
ap.add_argument("--tasks", default="tasks.jsonl")
ap.add_argument("--mode", default="guarded")
a = ap.parse_args()

tasks = {}
for line in open(os.path.join(here, a.tasks)):
    if line.strip():
        t = json.loads(line)
        tasks[t["id"]] = t

files = []
for f in glob.glob(os.path.join(here, a.dir, f"{a.mode}_*_trajectories.jsonl")):
    if a.mode == "agentic" and os.path.basename(f).startswith("agentic_v1_"):
        continue
    files.append(f)
files.sort(key=os.path.getmtime)
if not files:
    sys.exit(f"No '{a.mode}' runs found in agent/{a.dir}")

runs = []                                   # list of {task_id: bool}
for f in files:
    res = {}
    for line in open(f):
        if not line.strip():
            continue
        tr = json.loads(line)
        task = tasks.get(tr.get("task_id"))
        if task:
            res[task["id"]] = bool(score_task(task, tr)["success"])
    runs.append(res)

print(f"{len(runs)} run(s) of mode '{a.mode}' in agent/{a.dir}\n")
rates = []
for i, (f, res) in enumerate(zip(files, runs), 1):
    n, p = len(res), sum(res.values())
    rates.append(p / n)
    print(f"run {i}: {p}/{n} = {p/n:.2f}   ({os.path.basename(f)})")
    sf = f.replace("_trajectories.jsonl", "_scores.json")
    if os.path.exists(sf):
        pv = json.load(open(sf)).get("provenance")
        if pv:
            print(f"        state: commit={pv.get('git_commit')} dirty={pv.get('git_dirty')} docs={pv.get('n_documents')}/{pv.get('docs_hash')} model={pv.get('model')}")
        else:
            print("        state: not recorded (older run)")
print(f"\nmean {sum(rates)/len(rates):.2f}   range {min(rates):.2f} - {max(rates):.2f}")

ids = sorted({i for r in runs for i in r})
unstable = [i for i in ids if len({r.get(i) for r in runs}) > 1]
always_fail = [i for i in ids if all(r.get(i) is False for r in runs)]
print(f"\nPASS(P)/FAIL(F) per run, one column per run:")
print(f"{'task':<6}{'runs':<{max(6, len(runs) + 2)}}question")
for i in ids:
    mark = "".join("P" if r.get(i) else "F" for r in runs)
    flag = "  <- UNSTABLE" if i in unstable else ("  <- fails every time" if i in always_fail else "")
    print(f"{i:<6}{mark:<{max(6, len(runs) + 2)}}{tasks[i]['question'][:60]}{flag}")
print(f"\nunstable tasks: {len(unstable)} | fail every run: {len(always_fail)} | pass every run: {len(ids) - len(unstable) - len(always_fail)}")
