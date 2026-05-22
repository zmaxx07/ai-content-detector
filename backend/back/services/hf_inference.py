import os
import aiohttp
from typing import Dict, Any

HF_API_KEY = os.getenv("HF_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

async def query_hf_model(model_id: str, payload: Dict[str, Any]) -> Any:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=HEADERS, json=payload) as response:
            return await response.json()

# Example Usage
# await query_hf_model("roberta-base-openai-detector", {"inputs": "Some text to check"})
