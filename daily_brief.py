import os
import ssl
import sys
import re
import json
import logging
import requests
import calendar
from datetime import date, datetime
from dotenv import load_dotenv

import yfinance as yf
import feedparser
from jinja2 import Environment, FileSystemLoader
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys and Config
XAI_API_KEY = os.getenv("XAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "C09MUE5TGC9") # Default from previous use

# WeasyPrint fix for macOS
if sys.platform == "darwin":
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
        "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    )

# Fix SSL context for legacy environments/Mac
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


# ==============================================================================
# CONSTANTS & MAPPINGS
# ==============================================================================

WEATHER_COLORS = {
    0: "#FFD700",  # Sun / Clear (Gold)
    1: "#87CEEB", 2: "#87CEEB", 3: "#87CEEB", # Partly Cloudy (Sky Blue)
    45: "#708090", 48: "#708090", # Fog (Slate Gray)
    51: "#4682B4", 53: "#4682B4", 55: "#4682B4", # Drizzle (Steel Blue)
    61: "#4682B4", 63: "#4682B4", 65: "#4682B4", # Rain
    80: "#4682B4", 81: "#4682B4", 82: "#4682B4", # Showers
    71: "#E0FFFF", 73: "#E0FFFF", 75: "#E0FFFF", 77: "#E0FFFF", # Snow (Light Cyan)
    95: "#9370DB", 96: "#9370DB", 99: "#9370DB"  # Thunderstorm (Medium Purple)
}

WEATHER_TEXT = {
    0: "Clear Sky",
    1: "Partly Cloudy", 2: "Partly Cloudy", 3: "Overcast",
    45: "Foggy", 48: "Rime Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain", 80: "Showers", 81: "Showers", 82: "Showers",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"
}

WEATHER_ICONS = {
    0: "☀️",  # Clear
    1: "⛅️", 2: "⛅️", 3: "☁️",  # Cloudy
    45: "🌫️", 48: "🌫️",  # Fog
    51: "🌦️", 53: "🌦️", 55: "🌧️",  # Drizzle
    61: "🌧️", 63: "🌧️", 65: "🌧️",  # Rain
    80: "🌦️", 81: "🌦️", 82: "🌦️",  # Showers
    71: "🌨️", 73: "🌨️", 75: "🌨️", 77: "🌨️",  # Snow
    95: "⛈️", 96: "⛈️", 99: "⛈️"  # Thunderstorm
}

TRENDING_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")

X_CATEGORIES = {
    'Science': ['science', 'physics', 'astronomy', 'nasa', 'space', 'discovery', 'quantum'],
    'Tech': ['ai', 'artificial intelligence', 'crypto', 'blockchain', 'tesla', 'spacex'],
    'World News': ['breaking', 'war', 'nuclear', 'elon', 'musk', 'trump'],
    'Copenhagen / Denmark': ['copenhagen', 'cph', 'denmark'],
    'Misc': ['northern lights', 'south africa', 'aurora', 'comet', 'asteroid', 'meteor']
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_weather_color(code):
    """Return a hex color for WMO weather code."""
    return WEATHER_COLORS.get(code, "#AAAAAA")  # Default Grey

def get_weather_text(code):
    """Return text description for WMO weather code."""
    return WEATHER_TEXT.get(code, "Unknown")

def get_weather_icon(code):
    """Return icon for WMO weather code."""
    return WEATHER_ICONS.get(code, "?")

def format_trending_since(raw_since):
    """Normalize trending_since to HH:MM, or return None if invalid."""
    if raw_since is None:
        return None
    if isinstance(raw_since, (int, float)):
        if raw_since <= 0:
            return None
        try:
            return datetime.utcfromtimestamp(raw_since).strftime("%H:%M")
        except Exception:
            return None
    if isinstance(raw_since, str):
        value = raw_since.strip()
        if not value:
            return None
        if value.lower() in {"n/a", "na", "none", "null", "unknown"}:
            return None
        if value.isdigit():
            try:
                ts_value = int(value)
                if ts_value > 1_000_000_000_000:
                    ts_value = ts_value / 1000
                return datetime.utcfromtimestamp(ts_value).strftime("%H:%M")
            except Exception:
                return None
        match = TRENDING_TIME_RE.search(value)
        if match:
            hour = int(match.group(1))
            minute = match.group(2)
            if 0 <= hour <= 23:
                return f"{hour:02d}:{minute}"
    return None


# ==============================================================================
# DATA FETCHING FUNCTIONS
# ==============================================================================

def fetch_weather():
    """Fetch weather for Copenhagen using Open-Meteo API with hourly breakdown."""
    logger.info("Fetching weather...")
    # Copenhagen coordinates
    lat, lon = 55.6761, 12.5683
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation_probability", "wind_speed_10m",
                   "wind_direction_10m", "weather_code"],
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
        
        def get_segment_data(start_h, end_h):
            # Extract slices
            temps = hourly["temperature_2m"][start_h:end_h]
            precips = hourly["precipitation_probability"][start_h:end_h]
            winds = hourly["wind_speed_10m"][start_h:end_h]
            wind_dirs = hourly["wind_direction_10m"][start_h:end_h]
            codes = hourly["weather_code"][start_h:end_h]
            
            # Aggregate
            avg_temp = sum(temps) / len(temps) if temps else 0
            max_precip = max(precips) if precips else 0
            max_wind = max(winds) if winds else 0
            avg_wind_dir = sum(wind_dirs) / len(wind_dirs) if wind_dirs else 0
            
            # Most common weather code in segment
            code = max(codes, key=codes.count) if codes else 0
            
            return {
                "temp": round(avg_temp),
                "precip": max_precip,
                "wind": round(max_wind),
                "wind_dir": round(avg_wind_dir),
                "color": get_weather_color(code),
                "condition": get_weather_text(code),
                "icon": get_weather_icon(code)
            }

        return {
            "morning": get_segment_data(6, 12),
            "afternoon": get_segment_data(12, 18),
            "evening": get_segment_data(18, 24),
            "sunrise": daily.get("sunrise", [""])[0],
            "sunset": daily.get("sunset", [""])[0],
            "daily_precip": max(hourly["precipitation_probability"]) if hourly.get("precipitation_probability") else 0
        }
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return None


def fetch_stocks():
    """Fetch stock data using yfinance."""
    logger.info("Fetching stocks...")
    tickers = {
        "S&P 500": "SPY",
        "Tesla": "TSLA",
        "Nvidia": "NVDA",
        "Bitcoin": "BTC-USD",
        "United Health": "UNH",
        "Echo Star": "SATS"
    }
    
    stock_data = {}
    for name, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if len(hist) >= 1:
                current_close = hist["Close"].iloc[-1]
                # Use previous close if available, else standard fallback
                prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_close
                
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
            
    # Calculate average market performance
    if stock_data:
        total_percent = sum([d["percent"] for d in stock_data.values()])
        stock_data["average_percent"] = total_percent / len(stock_data)
    else:
        stock_data["average_percent"] = 0.0

    return stock_data


def summarize_with_ai(text, prompt_prefix="Summarize this news item:"):
    """Summarize text using xAI API."""
    if not XAI_API_KEY:
        return text 
    
    # Simple deduplication or length check could go here if needed
    
    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
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
        # Increased timeout to 60s
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        logger.error(f"AI summarization failed: {e}")
        return text if text else "Content unavailable"


def fetch_rss_feed(url, limit=3, prompt="Summarize this content:"):
    """Generic RSS feed fetcher and summarizer."""
    news_items = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            content_text = getattr(entry, 'summary', getattr(entry, 'description', entry.title))
            summary = summarize_with_ai(content_text, prompt)
            if summary:
                news_items.append({
                    "headline": summary,
                    "link": entry.link
                })
    except Exception as e:
        logger.error(f"Error fetching feed {url}: {e}")
    return news_items


def fetch_world_news():
    """Fetch and summarize world news from BBC."""
    logger.info("Fetching world news...")
    return fetch_rss_feed("http://feeds.bbci.co.uk/news/world/rss.xml", limit=5, prompt="Summarize this news item:")


def fetch_space_news():
    """Fetch and summarize space news."""
    logger.info("Fetching space news...")
    return fetch_rss_feed("https://spacenews.com/feed/", limit=3, prompt="Summarize this content:")


def fetch_copenhagen_events():
    """Fetch and summarize Copenhagen events/news."""
    logger.info("Fetching Copenhagen events...")
    return fetch_rss_feed("https://cphpost.dk/feed/", limit=3, prompt="Summarize this content.")


def fetch_x_trending():
    """Fetch trending topics from X using OAuth1 and personalized_trends API."""
    logger.info("Fetching X trending topics...")
    
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        logger.warning("X Trending: Missing OAuth credentials.")
        return {}

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
            return {}
        
        trends_data = response.json().get('data', [])
        
        # Filter trends into categories
        filtered_results = {}
        
        for category, keywords in X_CATEGORIES.items():
            matches = [
                trend for trend in trends_data
                if any(kw.lower() in trend.get('trend_name', '').lower() for kw in keywords)
            ]
            
            cleaned_matches = []
            for t in matches[:5]: # Top 5 per category
                # Clean trending_since data
                raw_since = t.get('trending_since')
                display_time = format_trending_since(raw_since)
                
                cleaned_matches.append({
                    'name': t.get('trend_name'),
                    'post_count': t.get('post_count') or t.get('tweet_count', 'N/A'),
                    'category': t.get('category', 'N/A'),
                    'trending_since': display_time,
                    'link': f"https://x.com/search?q={t.get('trend_name', '').replace(' ', '+')}"
                })
            
            if cleaned_matches:
                filtered_results[category] = cleaned_matches
        
        # Log success
        total_items = sum(len(v) for v in filtered_results.values())
        logger.info(f"Successfully fetched {total_items} X trending topics across {len(filtered_results)} categories")
        
        # Sort so categories with items appear first, then by count
        sorted_categories = sorted(
            filtered_results.items(),
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
    fallback = {"text": "The obstacle is the way.", "author": "Marcus Aurelius"}
    
    if not XAI_API_KEY:
        return fallback

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
            "response_format": {"type": "json_object"}
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error fetching quote: {e}")
        return fallback


# ==============================================================================
# GENERATION & DELIVERY
# ==============================================================================

def generate_pdf(data):
    """Generate PDF from data using Jinja2 and WeasyPrint."""
    logger.info("Generating PDF...")
    try:
        from weasyprint import HTML
    except Exception as e:
        logger.error(f"WeasyPrint import failed: {e}")
        logger.error("PDF generation skipped. Install WeasyPrint system dependencies.")
        return None
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
        logger.warning("SLACK_BOT_TOKEN not found. Skipping upload.")
        return

    client = WebClient(token=SLACK_BOT_TOKEN)
    
    try:
        client.files_upload_v2(
            channel=SLACK_CHANNEL_ID,
            file=pdf_path,
            title=f"Daily Briefing - {date.today().strftime('%Y-%m-%d')}",
            initial_comment="Mr Smith, here is your Daily Briefing."
        )
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
    # Use generic fetcher wrappers for news
    world_news = fetch_world_news()
    space_news = fetch_space_news()
    copenhagen = fetch_copenhagen_events()
    x_trending = fetch_x_trending()
    quote = fetch_quote()
    
    # Calculate Year Percentage
    today = date.today()
    day_of_year = today.timetuple().tm_yday
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
        # send_to_slack(pdf_path)
        print("Skipping Slack")
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
