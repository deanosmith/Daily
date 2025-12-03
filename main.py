import os
import sys
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get keys from .env
XAI_API_KEY = os.getenv("XAI_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


prompt = """
Using a combination of today's day and date as a deterministic selection technique to ensure randomness, select 1 random quote from Jesus in the NKJV Bible.
Ensure the quote covers the entire quote if it spans multiple verses.

Provide a brief translation of the original key words.

Use the following format:
*{Book} : {Verse}*
\n*{Verse Text}*
\n{Contextual information if applicable}
\n_Translation: {Original word and modern translation/simlarity}_
"""

# print("Sending Request")
# xAI API request
url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "grok-4-fast-reasoning",  # Adjust if exact name differs; check x.ai/api docs
    "messages": [{"role": "user", "content": prompt}]
}

response = requests.post(url, headers=headers, json=data)
response.raise_for_status()  # Raise error if request fails

# print("Request Received")

content = response.json()["choices"][0]["message"]["content"]

# print(content)

# Forward to Slack
slack_data = {"text": content}
slack_response = requests.post(SLACK_WEBHOOK_URL, json=slack_data)
slack_response.raise_for_status()

# print("Sent")