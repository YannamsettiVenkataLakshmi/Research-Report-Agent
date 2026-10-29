"""
Tools the agent can call. Each tool returns (text_for_the_model, metadata_for_the_log).

Design choices for small models:
  * The agent never sees session_id or UUIDs. Documents are referred to by number
    or filename; this module maps them to doc_ids and injects the session_id.
  * Errors are returned to the model as text (ToolError) so it can correct itself,
    instead of crashing the run.
"""
import ast
import operator
import re

import requests


class ToolError(Exception):
    """A problem the model can understand and recover from."""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List the documents available in this conversation, numbered, "
                "each with a short preview. Call this when you need to know which "
                "documents exist or which one the user means."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search the uploaded documents and return the most relevant passages "
                "(with filename and page). Optionally restrict the search to specific "
                "documents by number or filename."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query using specific technical terms.",
                    },
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Optional. Document numbers or filenames, e.g. ["1"] or ["dl-curriculum.pdf"].',
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression such as '17.5/100*2480'. Supports + - * / ** % and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str):
    """Arithmetic only. No names, no calls, no attribute access."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("Exponent too large.")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ToolError("Unsupported expression. Use only numbers and + - * / ** % ( ).")

    try:
        tree = ast.parse(expression.strip().replace("^", "**").replace(",", ""), mode="eval")
    except SyntaxError:
        raise ToolError("Could not parse the expression.")
    try:
        return ev(tree)
    except ZeroDivisionError:
        raise ToolError("Division by zero.")


class Tools:
    def __init__(self, rag_url, session_id, http=requests, timeout=60):
        self.rag_url = rag_url.rstrip("/")
        self.session_id = session_id
        self.http = http
        self.timeout = timeout
        self._docs = None

    # ---- helpers -------------------------------------------------------
    def _load_docs(self):
        if self._docs is None:
            try:
                r = self.http.get(f"{self.rag_url}/documents",
                                  params={"session_id": self.session_id},
                                  timeout=self.timeout)
                r.raise_for_status()
            except Exception as exc:
                raise ToolError(f"Could not list documents: {exc}")
            self._docs = r.json().get("documents", [])
        return self._docs

    def _valid_options(self):
        return "; ".join(f"{i}. {d['filename']}" for i, d in enumerate(self._load_docs(), 1))

    def _resolve(self, refs):
        docs = self._load_docs()
        if isinstance(refs, str):
            refs = [refs]
        ids, names = [], []
        for ref in refs:
            key = str(ref).strip().strip(".").lower()
            match = None
            if key.isdigit() and 1 <= int(key) <= len(docs):
                match = docs[int(key) - 1]
            elif key:
                for d in docs:
                    if key == d["doc_id"].lower() or key in d["filename"].lower():
                        match = d
                        break
            if match is None:
                raise ToolError(f"Unknown document {ref!r}. Valid documents: {self._valid_options()}")
            if match["doc_id"] not in ids:
                ids.append(match["doc_id"])
                names.append(match["filename"])
        return ids, names

    # ---- tools ---------------------------------------------------------
    def list_documents(self):
        docs = self._load_docs()
        if not docs:
            return "No documents are available.", {"n_documents": 0}
        lines = [f"{i}. {d['filename']} - {d.get('preview', '')}" for i, d in enumerate(docs, 1)]
        return "\n".join(lines), {"n_documents": len(docs)}

    def retrieve(self, query, documents=None):
        if not query or not str(query).strip():
            raise ToolError("query is required.")
        payload = {"query": str(query), "session_id": self.session_id}
        names = None
        if documents:
            ids, names = self._resolve(documents)
            payload["doc_ids"] = ids
        try:
            r = self.http.post(f"{self.rag_url}/retrieve", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except Exception as exc:
            raise ToolError(f"Retrieval service error: {exc}")
        results = r.json().get("results", [])
        meta = {
            "docs_searched": names if names else "ALL",
            "n_results": len(results),
            "sources": [(x["source"].get("source_filename"), x["source"].get("page")) for x in results],
        }
        if not results:
            return "No passages found.", meta
        lines = []
        for i, x in enumerate(results, 1):
            src = x["source"]
            text = " ".join(x["content"].split())[:700]
            lines.append(f"[{i}] {src.get('source_filename')}, p.{(src.get('page') or 0) + 1}: {text}")
        return "\n\n".join(lines), meta

    def calculator(self, expression):
        value = safe_eval(str(expression))
        return str(value), {"value": value}

    def run(self, name, args):
        if name == "list_documents":
            return self.list_documents()
        if name == "retrieve":
            return self.retrieve(args.get("query"), args.get("documents"))
        if name == "calculator":
            return self.calculator(args.get("expression", ""))
        raise ToolError(f"Unknown tool {name!r}. Available tools: list_documents, retrieve, calculator.")
