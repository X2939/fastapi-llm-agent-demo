import json

import httpx

BASE_URL = "http://127.0.0.1:9000"

def call_agent(message:str)->dict:
    response =httpx.post(
        f"{BASE_URL}/agent/chat",
        json={"message":message},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

def main():
    test_messages=[
        "请帮我计算 20 + 4 * (-3)",
        "请总结这个文件",
        "请查询价格大于100的商品",
        "请帮我计算 abc",
        "RAG 是什么？",
        "PagedAttention 解决了什么问题？",
        "这个项目有没有接入 Redis？",
    ]

    for message in test_messages:
        print("=" * 80)
        print(f"User: {message}")

        result = call_agent(message)
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()