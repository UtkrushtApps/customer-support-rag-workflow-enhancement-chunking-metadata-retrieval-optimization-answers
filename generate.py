from vector_store import ChromaVectorStore
from typing import List
import os
import openai

# For demonstration, we'll use OpenAI's GPT-3.5 as the LLM
def generate_answer(query: str, retrieved_chunks: List[dict], openai_api_key: str, max_context: int = 2000) -> str:
    # Concatenate retrieved chunk texts, respecting max_context characters
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    if len(context) > max_context:
        context = context[:max_context]
    prompt = f"""
You are a helpful customer support assistant. Given the following support context and question, answer as accurately as possible grounded in the support documentation context.
---
SUPPORT CONTEXT:
{context}
---
QUESTION: {query}
"""
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.2,
    )
    return response['choices'][0]['message']['content'].strip()

def answer_query(
    query: str,
    persist_directory: str,
    collection_name: str,
    openai_api_key: str
):
    vs = ChromaVectorStore(persist_directory, collection_name)
    retrieved = vs.retrieve(query, top_k=5)
    answer = generate_answer(query, retrieved, openai_api_key)
    return answer, retrieved
