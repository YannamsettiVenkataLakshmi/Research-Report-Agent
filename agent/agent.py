"""
Hand-written agent loop (no framework).

    while steps remain:
        ask the model -> if it answers, stop
                      -> if it calls tools, run them, feed results back, repeat

Every step is recorded in a trajectory dict so eval_agent.py can score it.

Modes (for the with-retry vs without-retry experiment):
    agentic  - decomposes comparisons, rewrites + retries failed searches (up to 3 retrievals)
    baseline - one retrieval allowed, no strategy instructions (like the old pipeline)
"""
import argparse
import json
import re
import time

import requests

from tools import Tools, ToolError, TOOL_SCHEMAS

PROMPT_AGENTIC = """You are a research assistant answering questions about documents the user uploaded.

Tools: list_documents, retrieve, calculator.

Rules:
1. If the question refers to a particular document, or you are unsure which documents exist, call list_documents first.
2. For a question about ONE document, call retrieve restricted to that document.
3. To compare or combine SEVERAL documents, make a SEPARATE retrieve call for each document, each with a focused question about that document.
4. If the passages do not answer the question, rewrite the query using specific technical terms the document would actually use (not words like 'paper' or 'summary') and retrieve again. Make at most 3 retrieve calls per question.
5. Answer ONLY from retrieved passages and mention the filename you used. If nothing relevant was found after retrying, say you could not find it.
6. If the question is ambiguous about which document is meant, say which documents are available.
7. Use calculator for arithmetic."""

PROMPT_BASELINE = """You are a research assistant answering questions about documents the user uploaded.
Use the retrieve tool to search the documents, then answer using only the retrieved passages.
If the passages do not contain the answer, say you could not find it."""


def extract_tool_calls(msg):
    """Return (calls, used_fallback). Small models sometimes write the call as text."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": fn.get("name"), "arguments": args if isinstance(args, dict) else {}})
    if calls:
        return calls, False

    content = msg.get("content") or ""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S) \
        or re.match(r'^\s*(\{.*"name".*\})\s*$', content, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments") or obj.get("parameters") or {}
                return [{"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}], True
        except json.JSONDecodeError:
            pass
    return [], False


class OllamaClient:
    def __init__(self, model="qwen2.5:7b", host="http://localhost:11434",
                 temperature=0.0, num_ctx=8192, http=requests):
        self.model, self.host, self.http = model, host.rstrip("/"), http
        # num_ctx matters: Ollama's small default context silently truncates retrieved passages.
        self.options = {"temperature": temperature, "num_ctx": num_ctx}

    def chat(self, messages, tools=None):
        payload = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if tools:
            payload["tools"] = tools
        r = self.http.post(f"{self.host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()


class Agent:
    def __init__(self, tools, llm, mode="agentic", max_steps=6, max_retrievals=None):
        self.tools, self.llm, self.mode, self.max_steps = tools, llm, mode, max_steps
        self.max_retrievals = max_retrievals if max_retrievals is not None else (3 if mode == "agentic" else 1)
        self.system_prompt = PROMPT_AGENTIC if mode == "agentic" else PROMPT_BASELINE

    def _execute(self, call, state):
        name, args = call["name"], call["arguments"]
        key = (name, json.dumps(args, sort_keys=True, default=str))
        rec = {"name": name, "arguments": args, "status": "ok", "latency_s": 0.0, "meta": {}}
        t0 = time.time()

        if key in state["seen"]:
            rec["status"] = "repeated"
            text = "You already made this exact call. Use a different query or answer with the passages you have."
        elif name == "retrieve" and state["retrievals"] >= self.max_retrievals:
            rec["status"] = "capped"
            text = (f"Retrieval limit reached ({self.max_retrievals}). Answer now using the passages "
                    "you already have, or say you could not find it.")
        else:
            try:
                text, rec["meta"] = self.tools.run(name, args)
                if name == "retrieve":
                    state["retrievals"] += 1
            except ToolError as exc:
                rec["status"], text = "error", f"Error: {exc}"
            except Exception as exc:  # never let a tool crash the run
                rec["status"], text = "error", f"Error: {type(exc).__name__}: {exc}"
        state["seen"].add(key)
        rec["latency_s"] = round(time.time() - t0, 3)
        rec["result_preview"] = text[:300]
        return rec, text

    def run(self, question):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}]
        traj = {"question": question, "mode": self.mode, "steps": [], "final_answer": None,
                "status": "max_steps", "error": None}
        state = {"seen": set(), "retrievals": 0}
        started = time.time()

        for step_no in range(1, self.max_steps + 1):
            t0 = time.time()
            try:
                resp = self.llm.chat(messages, TOOL_SCHEMAS)
            except Exception as exc:
                traj["status"], traj["error"] = "llm_error", str(exc)
                break
            msg = resp.get("message", {})
            calls, fallback = extract_tool_calls(msg)
            step = {"step": step_no, "llm_latency_s": round(time.time() - t0, 3),
                    "prompt_tokens": resp.get("prompt_eval_count") or 0,
                    "completion_tokens": resp.get("eval_count") or 0,
                    "parse_fallback": fallback, "assistant_text": (msg.get("content") or "")[:500],
                    "tool_calls": []}
            traj["steps"].append(step)

            if not calls:
                traj["final_answer"] = (msg.get("content") or "").strip()
                traj["status"] = "answered"
                break

            messages.append(msg)
            for call in calls:
                rec, text = self._execute(call, state)
                step["tool_calls"].append(rec)
                messages.append({"role": "tool", "tool_name": call["name"], "content": text})

        if traj["status"] == "max_steps":
            # Ran out of steps: force one last answer without tools.
            messages.append({"role": "user", "content":
                             "Step limit reached. Give your best final answer now using only the passages "
                             "already retrieved. If they do not answer the question, say so."})
            try:
                resp = self.llm.chat(messages, None)
                traj["final_answer"] = (resp.get("message", {}).get("content") or "").strip()
                traj["status"] = "forced_final"
            except Exception as exc:
                traj["status"], traj["error"] = "llm_error", str(exc)

        calls_all = [tc for s in traj["steps"] for tc in s["tool_calls"]]
        traj["totals"] = {
            "steps": len(traj["steps"]),
            "tool_calls": len(calls_all),
            "retrievals": sum(1 for c in calls_all if c["name"] == "retrieve" and c["status"] == "ok"),
            "errors": sum(1 for c in calls_all if c["status"] == "error"),
            "repeated": sum(1 for c in calls_all if c["status"] == "repeated"),
            "capped": sum(1 for c in calls_all if c["status"] == "capped"),
            "prompt_tokens": sum(s["prompt_tokens"] for s in traj["steps"]),
            "completion_tokens": sum(s["completion_tokens"] for s in traj["steps"]),
            "latency_s": round(time.time() - started, 2),
        }
        return traj


def main():
    ap = argparse.ArgumentParser(description="Ask the agent one question.")
    ap.add_argument("question")
    ap.add_argument("--session", default="chat_1")
    ap.add_argument("--rag-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--mode", choices=["agentic", "baseline"], default="agentic")
    ap.add_argument("--max-steps", type=int, default=6)
    a = ap.parse_args()

    agent = Agent(Tools(a.rag_url, a.session), OllamaClient(a.model), a.mode, a.max_steps)
    traj = agent.run(a.question)
    for s in traj["steps"]:
        for tc in s["tool_calls"]:
            print(f"step {s['step']}: {tc['name']}({json.dumps(tc['arguments'])}) -> {tc['status']}")
    print("\nSTATUS:", traj["status"], "|", traj["totals"])
    print("\nANSWER:\n", traj["final_answer"])


if __name__ == "__main__":
    main()
