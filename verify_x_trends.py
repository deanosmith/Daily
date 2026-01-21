import logging
import sys
import os
import requests
import json
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

def fetch_x_trending_test():
    """Isolated test function for X trending."""
    logger.info("Testing X trending fetch...")
    
    if not XAI_API_KEY:
        logger.warning("No API key found.")
        return []

    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        system_prompt = (
            "You are a helpful news assistant. List 3 top trending topics on X right now. "
            "Focus on technology, science, world events, and business. "
            "STRICTLY AVOID gossip, celebrity drama, or shallow viral trends. "
            "Return a JSON object with a single key 'trends' which is an array of objects. "
            "Each object must have keys: 'headline', 'summary', 'link'. "
            "For 'link', make a search URL like 'https://x.com/search?q=Topic+Name'."
        )
        
        payload = {
            "model": "grok-4-1-fast-reasoning", # Or try "grok-beta" if this times out?
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is trending on X right now? Return JSON."}
            ],
            "response_format": {"type": "json_object"}
        }
        
        logger.info("Sending request...")
        # Increased timeout for test
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        logger.info(f"Response Status: {response.status_code}")
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"Raw Content: {content}")
        
        data = json.loads(content)
        
        # Robust parsing
        if isinstance(data, dict):
            if "trends" in data and isinstance(data["trends"], list):
                return data["trends"]
            # Fallback for single object or other keys
            for key in ["topics", "items", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Check if data itself is the single item
            if "headline" in data:
                return [data]
                
        if isinstance(data, list):
            return data
            
        return []
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return []

if __name__ == "__main__":
    results = fetch_x_trending_test()
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))
