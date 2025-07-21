Generated File
import requests
import os
import time
from typing import List

# External LLM API configuration
GPT_API_URL = "https://dev-api.healthrx.co.in/sp-gw/api/openai/v1/chat/completions"
API_TOKEN = os.getenv("API_TOKEN", "sk-spgw-api01-key")
SUBSCRIPTION_KEY = os.getenv("SUBSCRIPTION_KEY", "")

def generate_answer(question: str, contexts: List[str], max_retries: int = 3) -> str:
    """
    Generate answer using external LLM API with context from document chunks
    """
    # Prepare the context
    context_text = "\n\n".join(contexts[:3])  # Use top 3 most relevant chunks
    
    # Create the prompt
    prompt = f"""You are a helpful assistant that answers questions based only on the provided context from insurance policy documents.

Context:
{context_text}

Question: {question}

Instructions:
- Answer only based on the information provided in the context above
- If the context doesn't contain relevant information, say "The provided document doesn't contain information about this topic"
- Be concise and direct in your response
- Quote specific policy clauses or conditions when relevant

Answer:"""

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Add subscription key if available
    if SUBSCRIPTION_KEY:
        headers["Subscription-Key"] = SUBSCRIPTION_KEY

    payload = {
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant specialized in analyzing insurance policy documents. Always base your answers strictly on the provided context."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "model": "gpt-4o",
        "temperature": 0.1,  # Low temperature for consistent, factual responses
        "max_tokens": 500    # Reasonable limit for answers
    }

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            print(f"Making LLM API call (attempt {attempt + 1})...")
            
            response = requests.post(
                GPT_API_URL,
                headers=headers,
                json=payload,
                timeout=45  # Longer timeout for LLM calls
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the answer
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"Generated answer: {answer[:100]}...")
                return answer
            else:
                raise ValueError("No choices in API response")
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return f"Unable to generate answer due to timeout. Question: {question}"
            time.sleep(2)  # Wait before retry
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return f"Unable to generate answer due to API error. Question: {question}"
            time.sleep(2)  # Wait before retry
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return generate_fallback_answer(question, contexts)
            time.sleep(2)  # Wait before retry

def generate_fallback_answer(question: str, contexts: List[str]) -> str:
    """
    Generate a simple fallback answer when LLM API is unavailable
    """
    if not contexts:
        return "No relevant information found in the document to answer this question."
    
    # Simple keyword-based fallback
    question_lower = question.lower()
    relevant_context = ""
    
    for context in contexts:
        if any(keyword in context.lower() for keyword in question_lower.split() if len(keyword) > 3):
            relevant_context = context[:300] + "..." if len(context) > 300 else context
            break
    
    if relevant_context:
        return f"Based on the document content: {relevant_context}"
    else:
        return "The document contains information but it may not directly answer your specific question."

def test_llm_connection() -> bool:
    """
    Test connection to LLM API
    """
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        if SUBSCRIPTION_KEY:
            headers["Subscription-Key"] = SUBSCRIPTION_KEY

        payload = {
            "messages": [
                {"role": "user", "content": "Say 'API connection test successful'"}
            ],
            "model": "gpt-4o",
            "max_tokens": 10
        }

        response = requests.post(
            GPT_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"LLM API connection test failed: {e}")
        return False
