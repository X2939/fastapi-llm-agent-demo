import os 

import httpx
from dotenv import load_dotenv

load_dotenv()# 加载 .env 文件里的配置

# 从环境变量读取，没有就用默认值
LLM_BASE_URL = os.getenv("LLM_BASE_URL","http://127.0.0.1:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY","token-abc123")
LLM_MODEL = os.getenv("LLM_MODEL","Qwen/Qwen2.5-1.5B-Instruct")

#核心函数：调用 AI
def chat_completion(messages:list[dict],temperature:float = 0.2)->str:
    """
    Call an OpenAI-compatible chat completions API.
    Works with local vLLM or OpenAI-compatible providers.
    """
    url = f"{LLM_BASE_URL}/chat/completions"#OpenAI 格式固定地址：/chat/completions

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
        response = httpx.post(url,headers=headers,json=payload,timeout=60)#向 AI 模型服务器发送一个 POST 请求，把问题发过去，等它回答。
        response.raise_for_status()#如果请求失败（400/401/404/500 等），直接抛出错误，不让程序继续往下跑！

        data = response.json()
        return data["choices"][0]["message"]["content"]
    #OpenAI 格式固定结构：choices[0].message.content 就是 AI 说的话。
    except httpx.HTTPError as e:
        return f"LLM request failed: {str(e)}"