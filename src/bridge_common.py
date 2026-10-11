
from __future__ import annotations
import os, re, json, time, hashlib
from typing import Any, Dict, List, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class OpenAICompatibleLLM:
    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if OpenAI else None
        self.model = model

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if self.client is None:
            return {"content": "[NO_LLM_CLIENT]", "usage": {}}
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )
            content = response.choices[0].message.content or ""
            return {"content": content, "usage": response.usage.model_dump() if response.usage else {}}
        except Exception as e:
            return {"content": f"[LLM_ERROR: {e}]", "usage": {}}

def as_bullets(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)

def ensure_think_answer_tags(text: str) -> str:
    if "<answer>" not in text:
        text = f"<answer>{text}</answer>"
    return text

def extract_labels(text: str, labels: Optional[List[str]] = None) -> List[str]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    answer = match.group(1).strip() if match else text.strip()
    return [answer]
