import re

TECH_PATTERNS = [

    # API endpoints
    re.compile(r"/[A-Za-z0-9._/-]+"),

    # GPT-4, Llama-3.1
    re.compile(r"\b[A-Za-z]+-\d+(?:\.\d+)*\b"),

    # CUDA-12.8
    re.compile(r"\b[A-Za-z]+-\d+(?:\.\d+)*\b"),

    # HTTP/2
    re.compile(r"\b[A-Za-z]+/\d+(?:\.\d+)*\b"),

    # Error codes
    re.compile(r"\b[A-Z]+(?:_[A-Z0-9]+)+\b"),

    # URLs
    re.compile(r"https?://\S+"),

    # Linux paths
    re.compile(r"(?:/[A-Za-z0-9._-]+)+"),

    # Windows paths
    re.compile(r"[A-Za-z]:\\(?:[^\\\n]+\\?)*")
]