from src.agent import create_agent
from src.rag_pipeline import process_query


session_id = "chat_1"

document_context = True
current_doc_ids = []


system_prompt = """
You are a research assistant answering questions using retrieved document passages.

Instructions:
1. Read every retrieved passage carefully.
2. Identify any passage that directly or partially answers the question.
3. Combine relevant information across passages when appropriate.
4. For questions asking to list or summarize topics, include all relevant topics supported by the retrieved passages.
5. Do not reject the evidence merely because a passage covers only part of the question.
6. Do not invent topics or use outside knowledge.
7. If the passages provide only partial information, summarize what they support and clearly state that the list may be incomplete.
8. If none of the passages contains relevant evidence, say that the available passages do not provide the answer.

Answer the exact question and avoid unrelated details.
"""


messages = [
    ("system", system_prompt)
]


llm = create_agent()


questions = [
    "What is the summary of the paper?"]


for question in questions:

    print("\n" + "=" * 70)

    print("\nUSER:", question)

    # Convert existing conversation into text
    conversation_history = str(messages)

    print("\n--- CONVERSATION HISTORY SENT TO REWRITER ---")
    print(conversation_history)
    print("---------------------------------------------")

    result = process_query(
        conversation_history,
        question,
        session_id,
        document_context,
        doc_ids=current_doc_ids
    )

    print("\nRoute:", result["route"])

    if result["route"] == "SINGLE_RAG":
         document_context = True

    # Add user question
    messages.append(
        ("user", question)
    )

    # Direct question
    if result["route"] == "DIRECT":

        response = llm.invoke(messages)

        messages.append(response)

        print("\nFINAL ANSWER:")
        print(response.content)

        continue

    # RAG question
    context = "\n\n".join(
        f"DOCUMENT {item['rank']}:\n{item['content']}"
        for item in result["results"]
    )

    prompt = f"""
    You are a research assistant answering questions about
    an uploaded document.

    Answer the user's question using ONLY the retrieved
    document information below.

    Important rules:

    1. Carefully read ALL retrieved document passages.
    2. Use information from multiple passages when they
    complement each other.
    3. Do not say that information is missing if the retrieved
    passages contain the answer.
    4. Do not use outside knowledge to fill missing information.
    5. If the retrieved passages truly do not contain enough
    information, say so clearly.
    6. Answer the exact question asked.
    7. Do not introduce unrelated models, architectures,
    datasets, or concepts that are not relevant to the question.

    Retrieved document information:

    {context}

    User question:

    {result['query']}
    """

    print("\n========== AGENT CONTEXT ==========")
    print(context)
    print("===================================")

    response = llm.invoke(prompt)

    messages.append(response)

    print("\nFINAL ANSWER:")
    print(response.content)