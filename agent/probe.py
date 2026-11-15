"""
See exactly which passages the retriever returns for a query (full text, not a preview).

    python agent/probe.py "performance change Sun inclination angle" --doc lunar --expect overexposed
    python agent/probe.py "ritikaverma@iitbhilai.ac.in" --doc aspirational --expect ritikaverma

--doc     part of a filename (optional; leave out to search every document)
--expect  a word you expect in the RIGHT passage; the script tells you if it was found
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import Tools, ToolError   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--doc", default=None)
ap.add_argument("--expect", default=None)
ap.add_argument("--session", default="chat_1")
ap.add_argument("--rag-url", default="http://127.0.0.1:8000")
a = ap.parse_args()

try:
    text, meta = Tools(a.rag_url, a.session).retrieve(a.query, [a.doc] if a.doc else None)
except ToolError as exc:
    sys.exit(f"Error: {exc}")

print(f"searched: {meta['docs_searched']} | passages returned: {meta['n_results']}\n")
print(text)
if a.expect:
    ranks = [i + 1 for i, block in enumerate(text.split("\n\n")) if a.expect.lower() in block.lower()]
    print("\n" + "-" * 60)
    print(f"'{a.expect}' FOUND in passage(s) {ranks}" if ranks else f"'{a.expect}' NOT FOUND in any of the returned passages")
