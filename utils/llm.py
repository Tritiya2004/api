import requests

GPT_API_URL = "https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/chat/completions"
API_TOKEN = "sk-spgw-api01-b39ff15132692a6834835a552e6f65b3"

def generate_answer(question, contexts):
    prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question.

Context:
{" ".join(contexts)}

Question: {question}
Answer:"""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": "gpt-4o"
    }
    resp = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
