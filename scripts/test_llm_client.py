from app.llm_client import chat_completion


def main():
    print("Start testing LLM client...")

    answer = chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请用一句话解释什么是 FastAPI。"},
        ]
    )

    print("LLM answer:")
    print(answer)


if __name__ == "__main__":
    main()
