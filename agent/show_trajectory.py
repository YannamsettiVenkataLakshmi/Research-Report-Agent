"""
Print what the agent actually did on one task.

    python agent/show_trajectory.py q04 agentic
    python agent/show_trajectory.py q07 baseline

Reads the newest results/<mode>_*_trajectories.jsonl file.
"""
import glob
import json
import os
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
sub = "results"
if "--dir" in sys.argv:
    sub = sys.argv[sys.argv.index("--dir") + 1]
    args = [a for a in args if a != sub]
task_id = args[0] if len(args) > 0 else None
mode = args[1] if len(args) > 1 else "agentic"
if not task_id:
    sys.exit("usage: python agent/show_trajectory.py <task_id> [baseline|agentic_v1|agentic]")

here = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(here, sub, f"{mode}_*_trajectories.jsonl")), key=os.path.getmtime)
if not files:
    sys.exit(f"No results found for mode '{mode}'. Run eval_agent.py first.")

for line in open(files[-1]):
    t = json.loads(line)
    if t.get("task_id") != task_id:
        continue
    print(f"\nFILE:     {os.path.basename(files[-1])}")
    print(f"QUESTION: {t['question']}")
    print(f"STATUS:   {t['status']}   totals: {t['totals']}")
    for s in t["steps"]:
        print(f"\n--- step {s['step']} ---")
        if s["assistant_text"]:
            print("model said:", s["assistant_text"][:200])
        if not s["tool_calls"]:
            print("(no tool call: the model answered)")
        for c in s["tool_calls"]:
            print(f"tool: {c['name']}({json.dumps(c['arguments'])})  -> {c['status']}")
            if c.get("meta", {}).get("docs_searched") is not None:
                print("  searched:", c["meta"]["docs_searched"], "| passages:", c["meta"].get("n_results"))
            print("  result:", c["result_preview"][:160].replace("\n", " "))
    print("\nFINAL ANSWER:\n", t["final_answer"])
    break
else:
    print(f"Task {task_id} not found in {os.path.basename(files[-1])}")
