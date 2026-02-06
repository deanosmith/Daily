import os
import json
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# 1. Use BEARER_TOKEN as verified in your working snippet
bearer_token = os.getenv("BEARER_TOKEN")

if not bearer_token:
    print("Error: Missing BEARER_TOKEN in environment variables.")
    exit(1)

# 2. The working endpoint for US Trends (WOEID: 23424977)
url = "https://api.x.com/2/trends/by/woeid/23424977"

# 3. App-Only Authentication (Bearer Token)
headers = {
    "Authorization": f"Bearer {bearer_token}"
}

try:
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse JSON
        response_json = response.json()
        
        # Handle v2 structure (usually wrapped in 'data' or a list)
        # Based on v2 conventions, it typically returns a 'data' key
        raw_data = response_json.get('data', [])
        
        # If raw_data is empty, check if the response is a direct list (v1.1 compatibility)
        if not raw_data and isinstance(response_json, list):
             # Some endpoints wrap it as [{"trends": [...]}]
             raw_data = response_json[0].get('trends', [])

        print(f"Debug: API returned {len(raw_data)} trends.")
        
        # Limit to top 20
        trends_data = raw_data[:10]

        formatted_trends = []
        for t in trends_data:
            # v2 'trend_name' vs v1.1 'name'
            trend_name = t.get('trend_name') or t.get('name', 'N/A')
            
            # v2 'tweet_count' vs v1.1 'tweet_volume'
            count = t.get('tweet_count') or t.get('tweet_volume')
            
            # Formatting the count nicely
            if count:
                post_count = f"{count:,}"
            else:
                post_count = "N/A"

            formatted_trends.append({
                'name': trend_name,
                'post_count': post_count,
                'link': f"https://x.com/search?q={quote(trend_name)}"
            })

        # --- Human-readable output ---
        print("\nTop Trends:")
        if not formatted_trends:
            print("No trends available.")
        else:
            for i, trend in enumerate(formatted_trends, 1):
                print(f"{i}. {trend['name']}")
                print(f"   Volume: {trend['post_count']}")
                print(f"   Link:   {trend['link']}")
    
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"System Error: {e}")