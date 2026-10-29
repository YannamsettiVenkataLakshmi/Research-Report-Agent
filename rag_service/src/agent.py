from langchain_ollama import ChatOllama


def create_agent():

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0
    )

    return llm