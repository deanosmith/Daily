import os
import json
import requests
from requests_oauthlib import OAuth1Session
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# OAuth credentials (use your actual values; hardcoded for testing if needed)
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

# Debug: Check if credentials are set
if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
    print("Error: Missing OAuth credentials.")
    exit(1)

oauth = OAuth1Session(
    consumer_key,
    client_secret=consumer_secret,
    resource_owner_key=access_token,
    resource_owner_secret=access_token_secret
)

url = 'https://api.x.com/2/users/personalized_trends'

response = oauth.get(url)

if response.status_code == 200:
    raw_data = response.json().get('data', [])
    print(f"Debug: API returned {len(raw_data)} items total.") # <--- Add this
    trends_data = raw_data[:20]
    # trends_data = response.json().get('data', [])[:10]  # Limit to top 10

    # Format trends
    formatted_trends = []
    for t in trends_data:
        trend_name = t.get('trend_name', 'N/A')
        formatted_trends.append({
            'name': trend_name,
            'post_count': t.get('post_count') or t.get('tweet_count', 'N/A'),
            'category': t.get('category', 'N/A'),
            'trending_since': t.get('trending_since', 'N/A'),
            'link': f"https://x.com/search?q={quote(trend_name)}"
        })

    # JSON output
    # print(json.dumps(formatted_trends, indent=2, ensure_ascii=False))

    # Human-readable output
    print("\nTop Trends (Personalized):")
    if not formatted_trends:
        print("No trends available.")
    else:
        for i, trend in enumerate(formatted_trends, 1):
            print(f"{i}. {trend['name']}")
            print(f"   Posts: {trend['post_count']}")
            print(f"   Category: {trend['category']}")
            print(f"   Trending Since: {trend['trending_since']}")
            print(f"   Search Link: {trend['link']}")
else:
    print(f"Error {response.status_code}: {response.text}")