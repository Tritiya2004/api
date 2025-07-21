import requests
import os
from dotenv import load_dotenv

load_dotenv()

GPT_API_URL = os.getenv("GPT_API_URL", "https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/chat/completions")
API_TOKEN = os.getenv("API_TOKEN", "sk-spgw-api01-key")
SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY", "your_subscription_key_here")

def generate_answer(question: str, contexts: list):
    """Generate answer using external GPT API"""
    
    if not contexts:
        return "No relevant information found to answer the question."
    
    # Build context from chunks
    context_text = "\n".join(contexts)
    
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context. Be concise and accurate.

Context:
{context_text}

Question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "Subscription-Key": SUBSCRIPTION_KEY
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
            {"role": "user", "content": prompt}
        ],
        "model": "gpt-4o"
    }

    try:
        response = requests.post(
            GPT_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error calling LLM API: {str(e)}"
    except KeyError as e:
        return f"Error parsing LLM response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
