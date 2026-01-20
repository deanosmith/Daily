import os
import sys
import json
import logging
import requests
from datetime import date, datetime
from dotenv import load_dotenv

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

def get_weather_icon(code):
    """Return an emoji icon for WMO weather code."""
    if code == 0: return "☀️"
    if code in [1, 2, 3]: return "⛅"
    if code in [45, 48]: return "🌫️"
    if code in [51, 53, 55, 61, 63, 65]: return "🌧️"
    if code in [71, 73, 75, 77]: return "❄️"
    if code in [80, 81, 82]: return "🌦️"
    if code in [95, 96, 99]: return "⛈️"
    return "❓"

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
                "icon": get_weather_icon(code)
            }

        weather_data = {
            "morning": get_segment_data(6, 12),
            "afternoon": get_segment_data(12, 18),
            "evening": get_segment_data(18, 24),
            "sunrise": daily.get("sunrise", [""])[0],
            "sunset": daily.get("sunset", [""])[0]
        }
        return weather_data
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return None

# ... (skip stocks to next function)

def summarize_with_ai(text, prompt_prefix="Summarize this news item in one sentence:"):
    """Summarize text using xAI API."""
# ... (rest of summarize function is fine)

# ... (world and space news fetched same way)

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
            
            # Broader prompt for events/upcoming
            summary = summarize_with_ai(entry.description, "Is this a unique event, cultural happening, or major news in Copenhagen (upcoming or today)?")
            if summary:
                news_items.append({
                    "title": entry.title,
                    "summary": summary,
                    "link": entry.link
                })
                count += 1
            if len(news_items) >= 2: break
    except Exception as e:
        logger.error(f"Error fetching Copenhagen news: {e}")
        
    return news_items

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
            
    return stock_data

def summarize_with_ai(text, prompt_prefix="Summarize this news item in one sentence:"):
    """Summarize text using xAI API."""
    if not XAI_API_KEY:
        # Mocking for testing if no key: always return text (not ideal for strict filtering but needed for testing)
        return text[:100] + "..." 
        
    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        }
        # Updated system prompt for filtering
        system_prompt = (
            "You are a strict news filter. Your goal is to identify MAJOR, UNIQUE news. "
            "If the news item is generic, minor, or not globally significant (e.g. minor politics, sports, celebrity gossip), return the single word 'SKIP'. "
            "If it is major (e.g. new wars, major disasters, breakthrough science), summarize it in 10-15 words. "
            "Be extremely selective. I'd rather have no news than boring news."
        )
        payload = {
            "model": "grok-2-latest", # Or appropriate model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{prompt_prefix}\n\n{text}"}
            ]
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        if "skip" in content.lower():
            return None
        return content
    except Exception as e:
        logger.error(f"AI summarization failed: {e}")
        return None # Fail safe to 'no news' rather than bad news

def fetch_world_news():
    """Fetch and summarize world news."""
    logger.info("Fetching world news...")
    feed_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    news_items = []
    
    try:
        feed = feedparser.parse(feed_url)
        # Check top 5 items
        for entry in feed.entries[:5]:
            summary = summarize_with_ai(entry.summary, "Is this major global news?")
            if summary:
                news_items.append({
                    "title": entry.title,
                    "summary": summary,
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
            summary = summarize_with_ai(entry.summary, "Is this major space news? (Launches, discoveries)")
            if summary:
                news_items.append({
                    "title": entry.title,
                    "summary": summary,
                    "link": entry.link
                })
                if len(news_items) >= 2: break
    except Exception as e:
        logger.error(f"Error fetching space news: {e}")
        
    return news_items

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
            summary = summarize_with_ai(entry.description, "Is this a unique or special event in Copenhagen today?")
            if summary:
                news_items.append({
                    "title": entry.title,
                    "summary": summary,
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
    """Upload PDF to Slack."""
    logger.info("Sending to Slack...")
    
    if not os.path.exists(pdf_path):
        logger.error("PDF file does not exist.")
        return

    if not SLACK_BOT_TOKEN:
        logger.warning("SLACK_BOT_TOKEN not found. Skipping upload. (Mock success)")
        return

    url = "https://slack.com/api/files.upload"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    
    try:
        with open(pdf_path, 'rb') as f:
            data = {
                "channels": SLACK_CHANNEL_ID if SLACK_CHANNEL_ID else "#general", # Default to #general if not set
                "initial_comment": "Here is your Daily Briefing.",
                "title": f"Daily Briefing - {date.today().strftime('%Y-%m-%d')}"
            }
            files = {'file': f}
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            res_json = response.json()
            if not res_json.get("ok"):
                logger.error(f"Slack API Error: {res_json.get('error')}")
            else:
                logger.info("PDF uploaded to Slack successfully.")
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
    
    data = {
        "date": date.today().strftime("%A, %B %d, %Y"),
        "weather": weather,
        "stocks": stocks,
        "world_news": world_news,
        "space_news": space_news,
        "copenhagen": copenhagen
    }
    
    # 2. Generate PDF
    pdf_path = generate_pdf(data)
    
    # 3. Send to Slack
    if pdf_path:
        print("Skipping Slack for now")
        # send_to_slack(pdf_path)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
