"""Offline tests: scripted fake LLM + fake RAG API. No Ollama, no server needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import Agent, extract_tool_calls          # noqa: E402
from eval_agent import score_task, summarize        # noqa: E402
from tools import Tools, ToolError, safe_eval        # noqa: E402

DOCS = [
    {"doc_id": "id-cur", "filename": "dl-curriculum.pdf", "preview": "CampusX Deep Learning Curriculum"},
    {"doc_id": "id-lun", "filename": "Lunar_Landing_Hazard_Detection.pdf", "preview": "IMAGE-BASED LUNAR LANDING"},
]


class FakeResp:
    def __init__(self, data): self._d = data
    def raise_for_status(self): pass
    def json(self): return self._d


class FakeHTTP:
    def __init__(self): self.posts = []
    def get(self, url, **kw): return FakeResp({"documents": DOCS})
    def post(self, url, json=None, **kw):
        self.posts.append(json)
        fname = "dl-curriculum.pdf" if json.get("doc_ids") == ["id-cur"] else "Lunar_Landing_Hazard_Detection.pdf"
        return FakeResp({"results": [{"content": "U-Net semantic segmentation", "source": {"source_filename": fname, "page": 0}}]})


class ScriptedLLM:
    """Returns the next scripted message each time chat() is called."""
    def __init__(self, script): self.script, self.calls = list(script), 0
    def chat(self, messages, tools=None):
        self.calls += 1
        if tools is None:      # the forced-final request is sent WITHOUT tools, so only text can come back
            return {"message": {"role": "assistant", "content": "best-effort final answer"}}
        msg = self.script.pop(0) if self.script else {"role": "assistant", "content": "final"}
        return {"message": msg, "prompt_eval_count": 10, "eval_count": 5}


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def final(text):
    return {"role": "assistant", "content": text}


def make(script, mode="agentic", **kw):
    http = FakeHTTP()
    return Agent(Tools("http://x", "s1", http=http), ScriptedLLM(script), mode=mode, **kw), http


def test_happy_path_resolves_document_and_injects_session():
    agent, http = make([call("list_documents"), call("retrieve", query="curriculum topics", documents=["1"]),
                        final("The curriculum covers neural networks.")])
    t = agent.run("what is in the curriculum?")
    assert t["status"] == "answered" and t["totals"]["steps"] == 3
    assert http.posts[0]["doc_ids"] == ["id-cur"] and http.posts[0]["session_id"] == "s1"
    assert t["steps"][1]["tool_calls"][0]["meta"]["docs_searched"] == ["dl-curriculum.pdf"]


def test_filename_reference_and_unknown_document_error_is_recoverable():
    agent, _ = make([call("retrieve", query="q", documents=["nonexistent.pdf"]),
                     call("retrieve", query="q", documents=["lunar"]), final("ok")])
    t = agent.run("q")
    assert t["steps"][0]["tool_calls"][0]["status"] == "error"
    assert "Valid documents" in t["steps"][0]["tool_calls"][0]["result_preview"]
    assert t["steps"][1]["tool_calls"][0]["status"] == "ok"      # model corrected itself


def test_repeated_identical_call_is_flagged_not_executed():
    agent, http = make([call("retrieve", query="same"), call("retrieve", query="same"), final("done")])
    t = agent.run("q")
    assert t["totals"]["repeated"] == 1 and len(http.posts) == 1


def test_step_limit_forces_a_final_answer():
    agent, _ = make([call("retrieve", query=f"q{i}") for i in range(10)], max_steps=3, max_retrievals=10)
    t = agent.run("q")
    assert t["status"] == "forced_final" and t["totals"]["steps"] == 3 and t["final_answer"]


def test_baseline_allows_only_one_retrieval():
    agent, http = make([call("retrieve", query="a"), call("retrieve", query="b"), final("done")], mode="baseline")
    t = agent.run("q")
    assert len(http.posts) == 1 and t["totals"]["capped"] == 1


def test_text_tool_call_fallback():
    calls, used = extract_tool_calls({"content": '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'})
    assert used and calls[0]["name"] == "calculator" and calls[0]["arguments"] == {"expression": "2+2"}
    calls, used = extract_tool_calls({"content": "plain answer"})
    assert calls == [] and not used


def test_calculator_is_safe():
    assert abs(safe_eval("17.5/100*2480") - 434) < 1e-9
    assert safe_eval("2^3") == 8
    for bad in ["__import__('os').system('ls')", "open('f')", "2**999999", "1/0"]:
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_scoring_comparison_task():
    task = {"id": "q03", "type": "comparison", "must_contain_groups": [["curriculum"], ["lunar"]],
            "expected_tools": ["retrieve"], "expected_docs": ["curriculum", "lunar"]}
    agent, _ = make([call("retrieve", query="a", documents=["1"]), call("retrieve", query="b", documents=["2"]),
                     final("The curriculum covers neural nets while the lunar paper uses U-Net.")])
    row = score_task(task, agent.run("compare"))
    assert row["success"] and row["tool_accuracy"] and row["retrievals"] == 2

    # searching only ONE of the two documents must fail tool accuracy
    agent, _ = make([call("retrieve", query="a", documents=["1"]), final("curriculum and lunar")])
    assert not score_task(task, agent.run("compare"))["tool_accuracy"]


def test_scoring_refusal_and_summary():
    task = {"id": "q05", "type": "not_in_docs", "expect_refusal": True}
    agent, _ = make([call("retrieve", query="refund"), final("I could not find a refund policy.")])
    row = score_task(task, agent.run("refund?"))
    assert row["success"]
    s = summarize([row])
    assert s["overall"]["n"] == 1 and "not_in_docs" in s["by_type"]


def test_answer_without_searching_does_not_count_as_success():
    # keywords are present, but the agent never called retrieve -> must NOT be a success
    task = {"id": "q02", "type": "vague", "must_contain_groups": [["neural"]],
            "expected_tools": ["retrieve"], "expected_docs": ["curriculum"]}
    agent, _ = make([final("The curriculum covers neural networks.")])
    row = score_task(task, agent.run("what is in the curriculum?"))
    assert row["success"] is False and row["retrievals"] == 0


def test_modes_use_different_prompts_and_retrieval_caps():
    from agent import PROMPT_AGENTIC, PROMPT_AGENTIC_V1, PROMPT_BASELINE
    a2, _ = make([], mode="agentic")
    a1, _ = make([], mode="agentic_v1")
    b, _ = make([], mode="baseline")
    assert a2.system_prompt == PROMPT_AGENTIC and a1.system_prompt == PROMPT_AGENTIC_V1 and b.system_prompt == PROMPT_BASELINE
    assert (a2.max_retrievals, a1.max_retrievals, b.max_retrievals) == (3, 3, 1)
    assert "ALWAYS call retrieve" in PROMPT_AGENTIC and "ALWAYS call retrieve" not in PROMPT_AGENTIC_V1


def test_guarded_mode_sends_the_model_back_once_if_it_answers_without_searching():
    agent, http = make([final("quick answer from memory"), call("retrieve", query="x"), final("grounded answer")],
                       mode="guarded")
    t = agent.run("q")
    assert t["status"] == "answered" and t["final_answer"] == "grounded answer"
    assert t["totals"]["nudges"] == 1 and t["totals"]["retrievals"] == 1 and len(http.posts) == 1


def test_guard_nudges_only_once_and_not_in_other_modes():
    agent, _ = make([final("answer one"), final("answer two")], mode="guarded")
    t = agent.run("q")
    assert t["totals"]["nudges"] == 1 and t["final_answer"] == "answer two"      # second answer is accepted
    agent, _ = make([final("answer one")], mode="agentic")
    t = agent.run("q")
    assert t["totals"]["nudges"] == 0 and t["final_answer"] == "answer one"


def test_guard_allows_calculator_only_answers():
    agent, _ = make([call("calculator", expression="2+2"), final("4")], mode="guarded")
    t = agent.run("what is 2+2?")
    assert t["totals"]["nudges"] == 0 and t["final_answer"] == "4"
