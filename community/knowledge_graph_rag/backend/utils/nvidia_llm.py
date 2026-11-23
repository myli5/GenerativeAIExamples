# Added for the 2026 re-run of this example (not in upstream).
#
# 1. ChatNVIDIA with reasoning switched off. The 2024 models this example was written for (Mixtral, Llama 3) are
#    retired; today's hosted Nemotron models "think" first by default. With langchain-nvidia-ai-endpoints 0.1.2
#    (max_tokens=1024) the thinking used up the whole budget and `content` came back empty on 20/20 test chunks,
#    so ast.literal_eval() failed. chat_template_kwargs={"enable_thinking": False} turns the thinking off.
# 2. A model list for the UI's "Choose an LLM" box. ChatNVIDIA.get_available_models() in 0.1.2 returns the
#    package's own 2024 model table (none of those chat models is still served); we list live chat models instead.
import os
import requests
from langchain_nvidia_ai_endpoints import ChatNVIDIA as _ChatNVIDIA

DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"   # chosen by measurement: 20/20 chunks parsed, ~0.95 s per chunk
CHAT_MODELS = [
    DEFAULT_MODEL,
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "mistralai/mistral-large-4-1t-a52b",
    "minimaxai/minimax-m3.1-flash",
]


class ChatNVIDIA(_ChatNVIDIA):
    """ChatNVIDIA that asks the model not to emit reasoning ("thinking") tokens."""

    def _get_payload(self, inputs, **kwargs):
        payload = super()._get_payload(inputs, **kwargs)
        payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return payload


def live_chat_models():
    """CHAT_MODELS that the API catalog still lists (DEFAULT_MODEL first)."""
    try:
        r = requests.get("https://integrate.api.nvidia.com/v1/models",
                         headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}, timeout=30)
        live = {m["id"] for m in r.json()["data"]}
        return [m for m in CHAT_MODELS if m in live] or CHAT_MODELS
    except Exception:
        return CHAT_MODELS


def strip_fences(text):
    """Remove a Markdown code fence (```json ... ``` / ```python ... ```) around a model's answer.
    Today's models often wrap JSON or Python lists in fences; json.loads / ast.literal_eval then fail."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
