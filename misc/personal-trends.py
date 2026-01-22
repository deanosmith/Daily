import json
import os
from requests_oauthlib import OAuth1Session
from dotenv import load_dotenv

load_dotenv()

# Replace with your keys and user tokens
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

# consumer_key = 'EUCslzDaI2RstsBXNkc7MwDky'
# consumer_secret = 'ZErY5IMmqHYQtYiOHqTZNZjUaRT4cptiaFmMmZkqgOeH7fnDq3'
# access_token = '329605343-naBJBwtvpKwsBcPgFlqJsfMG3qqjYQ2yFuJbAlMb'
# access_token_secret = 'BDdpsrMgLvdtBBmwIHJjBgLNgWwCzhkiAVAq18J75YHYt'

oauth = OAuth1Session(consumer_key,
                      client_secret=consumer_secret,
                      resource_owner_key=access_token,
                      resource_owner_secret=access_token_secret)

url = 'https://api.x.com/2/users/personalized_trends'
response = oauth.get(url)

if response.status_code == 200:
    raw_data = response.json().get('data', [])
    print(f"Debug: API returned {len(raw_data)} items total.") # <--- Add this
    trends_data = raw_data[:10]

    # Cleaned-up and extended keyword groups (lowercase only)
    categories = {
        'Science': ['science', 'physics', 'astronomy', 'nasa', 'space', 'discovery'],
        'Tech': ['ai', 'artificial intelligence', 'crypto', 'blockchain', 'tesla', 'spacex', 'elon', 'musk'],
        'World News': ['breaking', 'war', 'nuclear'], 
        'World Politics': ['politics', 'trump', 'far-right', 'immigration', 'davos'],
        'Copenhagen / Denmark': ['copenhagen', 'cph', 'denmark'],
        'Misc': ['northern lights', 'south africa', 'aurora']
    }

    filtered = {cat: [] for cat in categories}

    for cat, keywords in categories.items():
        matches = [
            trend for trend in trends_data
            if any(kw.lower() in trend.get('trend_name', '').lower() for kw in keywords)
        ]
        if matches:
            filtered[cat] = [
                {
                    'name': t.get('trend_name'),
                    'post_count': t.get('post_count') or t.get('tweet_count', 'N/A'),
                    'category': t.get('category', 'N/A'),
                    'trending_since': t.get('trending_since', 'N/A')
                }
                for t in matches[:10]
            ]

    # ── JSON output (best for downstream use) ──
    print(json.dumps(filtered, indent=2, ensure_ascii=False))

    # ── Optional: human-readable output ──
    print("\nFiltered trends by category:")
    for cat in categories:
        print(f"\n{cat}:")
        items = filtered[cat]
        if not items:
            print("Nothing")
        else:
            for item in items:
                print(f"- {item['name']}")
                print(f"  Posts: {item['post_count']}")
                print(f"  Since: {item['trending_since']}")
                print(f"  Platform category: {item['category']}")
else:
    print(f"Error {response.status_code}: {response.text}")
