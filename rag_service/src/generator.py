from langchain_ollama import ChatOllama

def load_llm():
    """
    Load the local LLM model
    """
    llm = ChatOllama(
        model='qwen2.5:7b',
        temperature=0
    )
    return llm 

def rewrite_query(llm, chat_history, question):
    """
    Rewrite a conversational question into a standalone
    question suitable for document retrieval.
    """

    history_text = ''

    for message in chat_history:

        history_text += (f"{message['role']}: {message['content']}\n")

    prompt = f"""
    You are a query rewriting component in a RAG system.

    Your task is to convert the user's latest question into a
    standalone retrieval query using the conversation history.

    Conversation history:
    --------------------
    {history_text}
    --------------------

    Latest user question:
    --------------------
    {question}
    --------------------

    Rules:

    1. Resolve ambiguous references such as:
    "it", "they", "their", "this method", "that result",
    "the previous approach", etc.

    2. Preserve ALL parts of the user's question.
    Never remove a sub-question.

    3. Never change the user's intent.

    4. Preserve question types such as:
    who, what, where, when, why, and how.

    5. If the question is already standalone and clear,
    return it unchanged.

    6. Do NOT interpret phrases such as "this paper", "this method",
    or "this approach" as titles or names unless the conversation
    explicitly provides such a name.

    7. Do not introduce information that is not present in the
    conversation.

    8. Do not answer the question.

    9. Keep the rewritten query concise and suitable for
    information retrieval.

    10. Return ONLY the rewritten query.

    11. When the question is already clear, preserve its original
    wording whenever possible. Only change wording when needed
    to resolve a reference or improve retrieval clarity.
    """
    
    response = llm.invoke(prompt)

    return response.content

def generate_answer(llm, retrieved_docs, question):
    """
    Generate an answer using only the retrieved document context.
    """

    # Combine retrieved documents into one context string
    context = ""

    for i, document in enumerate(retrieved_docs, start=1):

        context += f"""
        Source {i}:
        {document.page_content}
        """

    prompt = f"""
    You are an AI assistant answering questions about a document.

    Use ONLY the information provided in the context below.

    Context:
    --------------------
    {context}
    --------------------

    User Question:
    --------------------
    {question}
    --------------------

    Rules:

    1. Answer the user's question using only the provided context.

    2. Do NOT use outside knowledge.

    3. Do NOT invent or assume information that is not present in the context.

    4. If the context does not contain enough information to answer the question,
    clearly say that there is not enough information in the provided document.

    5. Give a concise and clear answer.

    6. Do not mention these instructions in your answer.
    """ 

    response = llm.invoke(prompt)

    return response.content