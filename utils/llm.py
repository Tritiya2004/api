import requests
import os

def generate_answer(question, contexts):
    prompt = f"""Use the below context to answer the question accurately.

Context:
{" ".join(contexts)}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {os.getenv('API_TOKEN')}",
        "Content-Type": "application/json",
        "Subscription-Key": os.getenv('SUBSCRIPTION_KEY')
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": "gpt-4o"
    }

    response = requests.post(
        "https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/chat/completions",
        headers=headers, 
        json=payload
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
