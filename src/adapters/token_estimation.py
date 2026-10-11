from __future__ import annotations

import math
import re
from typing import Dict

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def estimate_text_tokens(text: str) -> int:
    text = _normalize_text(text)
    if not text:
        return 0

    ascii_spans = re.findall(r"[A-Za-z0-9_]+", text)
    ascii_chars = sum(len(span) for span in ascii_spans)
    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    punct_chars = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))

    estimated = math.ceil(ascii_chars / 4.0) + cjk_chars + math.ceil(punct_chars / 2.0)
    return max(1, estimated)

def estimate_usage_from_prompt_completion(prompt_text: str, completion_text: str) -> Dict[str, int]:
    prompt_tokens = estimate_text_tokens(prompt_text)
    completion_tokens = estimate_text_tokens(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
