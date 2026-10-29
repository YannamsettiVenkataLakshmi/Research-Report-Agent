from langchain_ollama import ChatOllama


def create_rewriter():

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0
    )

    return llm


def rewrite_query(llm, conversation_history, query):

    
    prompt = f"""
    You are a query rewriting assistant for a research QA system.

    Your task is to convert the user's current question into a standalone,
    clear query when necessary.

    Rules:
    1. Use conversation history only to resolve references to earlier
    topics, such as "it", "that method", or "what about its advantages".
    2. Preserve the subject and intent of the current question.
    3. If the current question clearly introduces a new topic, treat it
    as a new question. Do not carry over the previous topic.
    4. Do not add assumptions, subjects, or constraints that the user
    did not express or imply.
    5. If the current question is already clear and standalone, keep it
    unchanged.
    6. Return only the rewritten question, without explanations.

    Conversation history:
    {conversation_history}

    Current question:
    {query}

    Standalone query:
    """

    response = llm.invoke(prompt)

    return response.content.strip()