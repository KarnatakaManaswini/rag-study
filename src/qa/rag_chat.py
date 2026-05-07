from openai import OpenAI
import os


def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, context_chunks: list[str]) -> str:
    client = get_client()

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a study assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
