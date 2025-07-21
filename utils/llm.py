import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(question, contexts):
    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the provided context. Be concise and direct.

Context:
{" ".join(contexts)}

Question: {question}
Answer:
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()
