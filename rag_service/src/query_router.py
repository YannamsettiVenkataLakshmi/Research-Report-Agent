from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def create_router():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0
    )
    return llm



def route_query(llm, conversation_history, query, document_context):

    prompt = f"""
    You are a query router for a research assistant.

    Return exactly one of these labels:
    DIRECT
    SINGLE_RAG

    Choose SINGLE_RAG when the user's question should be answered using the uploaded documents, including:
    - Questions asking about a concept, method, architecture, dataset, result, or claim discussed in an uploaded document.
    - Follow-up questions referring to a previously discussed document topic.
    - Questions asking what a particular uploaded document or study contains.

    Choose DIRECT when the question is clearly unrelated to the uploaded documents and can be answered independently using general knowledge.

    Important decision rules:
    - A question does not need to explicitly mention a document to require retrieval.
    - A short or standalone question about a topic covered by the uploaded documents should use SINGLE_RAG.
    - Use conversation history to resolve follow-up references.
    - Do not choose SINGLE_RAG merely because documents exist if the question is clearly unrelated to them.
    - If the question could reasonably be asking about the uploaded document, prefer SINGLE_RAG.

    Do not answer the question. Return only DIRECT or SINGLE_RAG.

    Conversation history:
    {conversation_history}

    Document context status:
    {document_context}

    Current query:
    {query}

    Return only DIRECT or SINGLE_RAG.
    """
    
    
    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    route = content.strip().upper().rstrip(".")

    if route not in {"DIRECT", "SINGLE_RAG"}:
        raise ValueError(
            f"Invalid query router response: {content!r}"
        )

    return route
