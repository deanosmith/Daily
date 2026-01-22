import os
import ssl
import sys
import json
import logging
import requests
from datetime import date, datetime
from dotenv import load_dotenv

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Third-party imports
import yfinance as yf
import feedparser
from jinja2 import Environment, FileSystemLoader

# WeasyPrint fix for macOS
if sys.platform == "darwin":
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")

from weasyprint import HTML

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN") # To be provided later
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID") # Optional, or use a default

if not XAI_API_KEY:
    logger.warning("XAI_API_KEY is missing. AI summarization will be skipped or mocked.")

def get_weather_color(code):
    """Return a hex color for WMO weather code."""
    # Sun / Clear
    if code == 0: return "#FFD700"  # Gold
    # Partly Cloudy
    if code in [1, 2, 3]: return "#87CEEB" # Sky Blue
    # Fog
    if code in [45, 48]: return "#708090" # Slate Gray
    # Rain / Drizzle
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return "#4682B4" # Steel Blue
    # Snow
    if code in [71, 73, 75, 77]: return "#E0FFFF" # Light Cyan
    # Thunderstorm
    if code in [95, 96, 99]: return "#9370DB" # Medium Purple
    
    return "#AAAAAA" # Grey fallback

def get_weather_icon(code):
    """Return an emoji icon for WMO weather code."""
    # Sun / Clear
    if code == 0: return "☀️"
    # Partly Cloudy
    if code in [1, 2]: return "⛅"
    if code == 3: return "☁️"
    # Fog
    if code in [45, 48]: return "🌫️"
    # Drizzle
    if code in [51, 53, 55]: return "🌦️"
    # Rain
    if code in [61, 63, 65, 80, 81, 82]: return "🌧️"
    # Snow
    if code in [71, 73, 75, 77]: return "❄️"
    # Thunderstorm
    if code in [95, 96, 99]: return "⛈️"
    
    return "🌡️" # Fallback

def fetch_weather():
    """Fetch weather for Copenhagen using Open-Meteo API with hourly breakdown."""
    logger.info("Fetching weather...")
    # Copenhagen coordinates
    lat, lon = 55.6761, 12.5683
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation_probability", "wind_speed_10m", "wind_direction_10m", "weather_code"],
        "daily": ["sunrise", "sunset"],
        "timezone": "Europe/Berlin",
        "forecast_days": 1
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})
        
        # Helper to get average/max for a slice
        def get_segment_data(start_h, end_h):
            temps = hourly["temperature_2m"][start_h:end_h]
            precips = hourly["precipitation_probability"][start_h:end_h]
            winds = hourly["wind_speed_10m"][start_h:end_h]
            wind_dirs = hourly["wind_direction_10m"][start_h:end_h]
            codes = hourly["weather_code"][start_h:end_h]
            
            # Simple aggregation
            avg_temp = sum(temps) / len(temps) if temps else 0
            max_precip = max(precips) if precips else 0
            max_wind = max(winds) if winds else 0
            # Average wind direction
            avg_wind_dir = sum(wind_dirs) / len(wind_dirs) if wind_dirs else 0
            
            code = max(codes, key=codes.count) if codes else 0
            
            return {
                "temp": round(avg_temp),
                "precip": max_precip,
                "wind": round(max_wind),
                "wind_dir": round(avg_wind_dir),
                "color": get_weather_color(code),
                "icon": get_weather_icon(code)
            }

        weather_data = {
            "morning": get_segment_data(6, 12),
            "afternoon": get_segment_data(12, 18),
            "evening": get_segment_data(18, 24),
            "sunrise": daily.get("sunrise", [""])[0],
            "sunset": daily.get("sunset", [""])[0],
            "daily_precip": max(hourly["precipitation_probability"]) if hourly.get("precipitation_probability") else 0
        }
        return weather_data
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return None

if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def fetch_stocks():
    """Fetch stock data using yfinance."""
    logger.info("Fetching stocks...")
    tickers = {
        "S&P 500": "SPY",
        "Tesla": "TSLA",
        "Nvidia": "NVDA",
        "Bitcoin": "BTC-USD"
    }
    
    stock_data = {}
    for name, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            # Get previous close and current price
            # info = ticker.info # info can be slow/unreliable
            hist = ticker.history(period="2d")
            
            if len(hist) >= 1:
                current_close = hist["Close"].iloc[-1]
                prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_close # Fallback
                
                change = current_close - prev_close
                percent_change = (change / prev_close) * 100
                
                stock_data[name] = {
                    "price": current_close,
                    "change": change,
                    "percent": percent_change,
                    "color": "green" if change >= 0 else "red",
                    "arrow": "↑" if change >= 0 else "↓"
                }
        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")
            stock_data[name] = {"price": 0.0, "change": 0.0, "percent": 0.0, "color": "grey", "arrow": "-"}
            
            
    # Calculate average percent
    total_percent = sum([d["percent"] for d in stock_data.values()])
    avg_percent = total_percent / len(stock_data) if stock_data else 0.0
    stock_data["average_percent"] = avg_percent

    return stock_data

def summarize_with_ai(text, prompt_prefix="Summarize this news item:"):
    """Summarize text using xAI API."""
    if not XAI_API_KEY:
        # Mocking for testing if no key: always return text (not ideal for strict filtering but needed for testing)
        return text 
        
    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        # Updated system prompt - MERGED HEADLINE
        system_prompt = (
            "You are a helpful news assistant. "
            "Merge the news title and description into one single, helpful, engaging sentence (max 20 words). "
            "Do not filter anything out. Be specific."
        )
        payload = {
            "model": "grok-4-1-fast-reasoning",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{prompt_prefix}\n\n{text}"}
            ]
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        return content
    except Exception as e:
        logger.error(f"AI summarization failed: {e}")
        # Fail safe to return original text so we always have content
        return text if text else "Content unavailable"

def fetch_world_news():
    """Fetch and summarize world news."""
    logger.info("Fetching world news...")
    feed_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    news_items = []
    
    try:
        feed = feedparser.parse(feed_url)
        # Check top 5 items
        for entry in feed.entries[:5]:
            # Robust content fetching
            content_text = getattr(entry, 'summary', getattr(entry, 'description', entry.title))
            summary = summarize_with_ai(content_text, "Summarize this news item:")
            if summary:
                news_items.append({
                    "headline": summary,
                    "link": entry.link
                })
                if len(news_items) >= 3: break # Max 3 items
    except Exception as e:
        logger.error(f"Error fetching world news: {e}")
        
    return news_items

def fetch_space_news():
    """Fetch and summarize space news."""
    logger.info("Fetching space news...")
    feed_url = "https://spacenews.com/feed/"
    news_items = []
    
    try:
        feed = feedparser.parse(feed_url)
        # Check top 3 items
        for entry in feed.entries[:3]:
            content_text = getattr(entry, 'summary', getattr(entry, 'description', entry.title))
            summary = summarize_with_ai(content_text, "Summarize this content:")
            if summary:
                news_items.append({
                    "headline": summary,
                    "link": entry.link
                })
                if len(news_items) >= 2: break
    except Exception as e:
        logger.error(f"Error fetching space news: {e}")
        
    return news_items

def fetch_x_trending():
    """Fetch trending topics from X using OAuth1 and the personalized_trends API."""
    logger.info("Fetching X trending topics...")
    
    # X API OAuth credentials
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    
    try:
        from requests_oauthlib import OAuth1Session
        
        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret
        )
        
        url = 'https://api.x.com/2/users/personalized_trends'
        response = oauth.get(url)
        
        if response.status_code != 200:
            logger.error(f"X API error {response.status_code}: {response.text}")
            return {}  # Empty dict for categories
        
        trends_data = response.json().get('data', [])
        
        # Category keyword groups (lowercase only)
        categories = {
            'Science': ['science', 'physics', 'astronomy', 'nasa', 'space', 'discovery', 'quantum'],
            'Tech': ['ai', 'artificial intelligence', 'crypto', 'blockchain', 'tesla', 'spacex'],
            'World News': ['breaking', 'war', 'nuclear', 'elon', 'musk', 'trump'],
            'Copenhagen / Denmark': ['copenhagen', 'cph', 'denmark'],
            'Misc': ['northern lights', 'south africa', 'aurora', 'comet', 'asteroid']
        }
        
        filtered = {cat: [] for cat in categories}
        
        for cat, keywords in categories.items():
            matches = [
                trend for trend in trends_data
                if any(kw.lower() in trend.get('trend_name', '').lower() for kw in keywords)
            ]
            if matches:
                filtered[cat] = []
                for t in matches[:5]:
                     # Clean trending_since
                     raw_since = t.get('trending_since', 'N/A')
                     if raw_since and 'T' in raw_since and raw_since[0].isdigit():
                         # ISO format like 2023-10-27T10:00:00Z -> 10:00
                         display_time = raw_since.split('T')[1][:5]
                     else:
                         # Fallback/Text
                         display_time = raw_since
            
                     filtered[cat].append({
                        'name': t.get('trend_name'),
                        'post_count': t.get('post_count') or t.get('tweet_count', 'N/A'),
                        'category': t.get('category', 'N/A'),
                        'trending_since': display_time,
                        'link': f"https://x.com/search?q={t.get('trend_name', '').replace(' ', '+')}"
                    })
        
        # Count total trends found
        total = sum(len(v) for v in filtered.values())
        logger.info(f"Successfully fetched {total} X trending topics across {len([c for c in filtered if filtered[c]])} categories")
        
        # Sort categories: populated first, then by count (descending), then alphabetical
        # Converting to a list of tuples (category_name, items) to preserve order for the template
        sorted_categories = sorted(
            filtered.items(),
            key=lambda item: (len(item[1]) > 0, len(item[1])), 
            reverse=True
        )
        
        return sorted_categories
        
    except ImportError:
        logger.error("requests_oauthlib not installed - cannot fetch X trends")
        return {}
    except Exception as e:
        logger.error(f"Error fetching X trending: {e}")
        return {}

def fetch_quote():
    """Fetch a Quote of the Day (Stoicism/Proverbs) using AI."""
    logger.info("Fetching Quote of the Day...")
    if not XAI_API_KEY:
        return {"text": "The obstacle is the way.", "author": "Marcus Aurelius"}

    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = "Generate a short, wise quote from Stoic philosophy or the Book of Proverbs. Return JSON format: {\"text\": \"Quote text\", \"author\": \"Author Name\"}."
        
        payload = {
            "model": "grok-4-1-fast-reasoning",
            "messages": [
                {"role": "system", "content": "You are a wise assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"} # Ensure JSON
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error fetching quote: {e}")
        return {"text": "The obstacle is the way.", "author": "Marcus Aurelius"}

def fetch_copenhagen_events():
    """Fetch and summarize Copenhagen events."""
    logger.info("Fetching Copenhagen events...")
    feed_url = "https://cphpost.dk/feed/"
    news_items = []
    
    try:
        feed = feedparser.parse(feed_url)
        count = 0
        for entry in feed.entries:
            if count >= 3: break
            
            # Use AI to judge if it's "unique" or "special"
            content_text = getattr(entry, 'description', getattr(entry, 'summary', entry.title))
            summary = summarize_with_ai(content_text, "Summarize this content.")
            if summary:
                news_items.append({
                    "headline": summary,
                    "link": entry.link
                })
                count += 1
            if len(news_items) >= 2: break
    except Exception as e:
        logger.error(f"Error fetching Copenhagen news: {e}")
        
    return news_items

def generate_pdf(data):
    """Generate PDF from data using Jinja2 and WeasyPrint."""
    logger.info("Generating PDF...")
    try:
        env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
        template = env.get_template("daily_brief_template.html")
        html_out = template.render(**data)
        
        pdf_path = "daily_brief.pdf"
        HTML(string=html_out, base_url=os.path.dirname(__file__)).write_pdf(pdf_path)
        logger.info(f"PDF generated at {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None

def send_to_slack(pdf_path):
    """Upload PDF to Slack using the v2 API."""
    logger.info("Sending to Slack...")
    
    if not os.path.exists(pdf_path):
        logger.error("PDF file does not exist.")
        return

    if not SLACK_BOT_TOKEN:
        logger.warning("SLACK_BOT_TOKEN not found. Skipping upload. (Mock success)")
        return

    # Initialize the official Slack Client
    client = WebClient(token=SLACK_BOT_TOKEN)
    
    try:
        # files_upload_v2 handles the complex 3-step upload process automatically
        response = client.files_upload_v2(
            channel="C09MUE5TGC9",  # Note: argument is 'channel', not 'channels'
            file=pdf_path,
            title=f"Daily Briefing - {date.today().strftime('%Y-%m-%d')}",
            initial_comment="Mr Smith, here is your Daily Briefing."
        )
        
        # The SDK raises an exception on error, so if we get here, it worked.
        logger.info("PDF uploaded to Slack successfully.")

    except SlackApiError as e:
        logger.error(f"Slack API Error: {e.response['error']}")
    except Exception as e:
        logger.error(f"Error uploading to Slack: {e}")

def main():
    logger.info("Starting Daily Briefing generation...")
    
    # 1. Gather Data
    weather = fetch_weather()
    stocks = fetch_stocks()
    world_news = fetch_world_news()
    space_news = fetch_space_news()
    copenhagen = fetch_copenhagen_events()
    x_trending = fetch_x_trending()
    quote = fetch_quote()
    
    
    # Calculate Year Percentage
    today = date.today()
    day_of_year = today.timetuple().tm_yday
    # check for leap year roughly or use 365.25 logic, but simple 366/365 is fine. 
    # using isocalendar or just 365/366 based on year.
    import calendar
    days_in_year = 366 if calendar.isleap(today.year) else 365
    year_percent = (day_of_year / days_in_year) * 100

    data = {
        "date": today.strftime("%A, %B %d"),
        "year_percent": year_percent,
        "weather": weather,
        "stocks": stocks,
        "world_news": world_news,
        "space_news": space_news,
        "copenhagen": copenhagen,
        "x_trending": x_trending,
        "quote": quote
    }
    
    # 2. Generate PDF
    pdf_path = generate_pdf(data)
    
    # 3. Send to Slack
    if pdf_path:
        # print("Skipping Slack for now")
        send_to_slack(pdf_path)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
