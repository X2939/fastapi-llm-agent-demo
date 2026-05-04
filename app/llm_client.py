import os 

import httpx
from dotenv import load_dotenv

load_dotenv()


LLM_BASE_URL = os.getenv("LLM_BASE_URL","http://127.0.0.1:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY","token-abc123")
LLM_MODEL = os.getenv("LLM_MODEL","Qwen/Qwen2.5-1.5B-Instruct")

def chat_completion(messages:list[dict],temperature:float = 0.2)->str:
    """
    Call an OpenAI-compatible chat completions API.
    Works with local vLLM or OpenAI-compatible providers.
    """
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization":f"Bearer {LLM_API_KEY}",
        "Content-type":"application/json",

    }
    #发送给 AI 的内容
    payload = {
        "model":LLM_MODEL,
        "messages":messages,
        "temperature":temperature,
    }
    try:
        response = httpx.post(url,headers=headers,json=payload,timeout=60)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
   
    except httpx.HTTPError as e:
        return f"LLM request failed: {str(e)}"
