"""
Compare the newest saved run of each mode, without re-running anything.

    python agent/compare_runs.py              # all models found
    python agent/compare_runs.py qwen2.5      # only files whose name contains "qwen2.5"
"""
import glob
import json
import os
import sys
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(here, "results")
model_filter = sys.argv[1] if len(sys.argv) > 1 else ""

MODES = ["baseline", "agentic_v1", "agentic", "guarded"]
MATCH_ORDER = sorted(MODES, key=len, reverse=True)      # so "agentic_v1_..." is not mistaken for "agentic_..."

questions = {}
tasks_path = os.path.join(here, "tasks.jsonl")
if os.path.exists(tasks_path):
    for line in open(tasks_path):
        if line.strip():
            t = json.loads(line)
            questions[t["id"]] = t["question"]

runs = {}
files = sorted(glob.glob(os.path.join(results_dir, "*_scores.json")), key=os.path.getmtime)
for f in files:
    name = os.path.basename(f)
    if model_filter and model_filter not in name:
        continue
    mode = next((m for m in MATCH_ORDER if name.startswith(m + "_")), None)
    if mode:
        runs[mode] = {r["id"]: r for r in json.load(open(f))["rows"]}      # newest file wins

if not runs:
    sys.exit(f"No *_scores.json files found in {results_dir}")

modes = [m for m in MODES if m in runs]
print("\nOVERALL")
print(f"{'mode':<12}{'n':>4}{'success':>9}{'tool_acc':>10}{'steps':>7}{'tokens':>8}")
for m in modes:
    rows = list(runs[m].values()); n = len(rows)
    print(f"{m:<12}{n:>4}{sum(r['success'] for r in rows)/n:>9.2f}{sum(r['tool_accuracy'] for r in rows)/n:>10.2f}"
          f"{sum(r['steps'] for r in rows)/n:>7.2f}{sum(r['tokens'] for r in rows)/n:>8.0f}")

print("\nSUCCESS BY TASK TYPE  (passed / total)")
types = sorted({r["type"] for m in modes for r in runs[m].values()})
print(f"{'type':<14}" + "".join(f"{m:>12}" for m in modes))
for ty in types:
    cells = []
    for m in modes:
        rs = [r for r in runs[m].values() if r["type"] == ty]
        cells.append(f"{sum(r['success'] for r in rs)}/{len(rs)}")
    print(f"{ty:<14}" + "".join(f"{c:>12}" for c in cells))

def changes(a, b):
    fixed = [i for i in runs[b] if i in runs[a] and not runs[a][i]["success"] and runs[b][i]["success"]]
    broke = [i for i in runs[b] if i in runs[a] and runs[a][i]["success"] and not runs[b][i]["success"]]
    print(f"\n{a}  ->  {b}:  {len(fixed)} fixed, {len(broke)} newly failing")
    for label, ids in (("fixed", fixed), ("NEWLY FAILING", broke)):
        for i in sorted(ids):
            print(f"  {label:<14}{i}  [{runs[b][i]['type']}]  {questions.get(i, '')[:70]}")

pairs = [("baseline", "agentic"), ("agentic", "guarded")]
for a, b in pairs:
    if a in runs and b in runs:
        changes(a, b)

still = [i for i in runs[modes[0]] if all(i in runs[m] and not runs[m][i]["success"] for m in modes)]
print(f"\nFAILED IN EVERY MODE ({len(still)}): " + ", ".join(sorted(still)))
