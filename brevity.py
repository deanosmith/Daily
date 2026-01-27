import argparse
import os
import ssl
import sys
import re
import html
import hashlib
import json
import logging
import functools
import time
import requests
import calendar
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, quote
from datetime import date, datetime
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import yfinance as yf
import feedparser
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from slack_sdk import WebClient

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
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID") # Default from previous use

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
KEYWORDS_RE = re.compile(r"^\s*\[(?P<keywords>[^\]]+)\]\s*")

KEYWORD_COLOR_PALETTE = [
    ("#294f3a", "#3b6b4c", "#d9f4e2"),
    ("#2f3f5c", "#3f567a", "#d9e6ff"),
    ("#5a3a2e", "#7a4f3e", "#ffe1d3"),
    ("#4a375f", "#5f4c78", "#f0e3ff"),
    ("#3b545a", "#4f6f77", "#e0f3f6"),
    ("#5a5a2e", "#78783e", "#fff6c9"),
]

X_CATEGORIES = {
    'Science': ['science', 'physics', 'astronomy', 'nasa', 'space', 'discovery', 'quantum'],
    'Tech': ['ai', 'artificial intelligence', 'crypto', 'blockchain', 'tesla', 'spacex'],
    'World News': ['breaking', 'war', 'nuclear', 'elon', 'musk', 'trump'],
    'Copenhagen / Denmark': ['copenhagen', 'cph', 'denmark'],
    'Misc': ['northern lights', 'south africa', 'aurora', 'comet', 'asteroid', 'meteor']
}

X_TRENDS_PER_CATEGORY = 12
DATA_OUTPUT_PATH = "resources/brevity.json"
LEGACY_DATA_OUTPUT_PATH = "resources/daily_brief_data.json"
JESUS_QUOTES_PATH = "resources/jesus.json"
HTML_OUTPUT_PATH = "brevity.html"
REFRESH_X_ENDPOINT = "/api/refresh/x"
DEFAULT_HEADERS = {"User-Agent": "Brevity/1.0"}
REQUEST_TIMEOUT = 30
AI_TIMEOUT = 60
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.2
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


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

def keyword_style(keyword):
    """Return a deterministic style string for keyword badges."""
    digest = hashlib.md5(keyword.lower().encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(KEYWORD_COLOR_PALETTE)
    bg, border, text = KEYWORD_COLOR_PALETTE[index]
    return f"--kw-bg: {bg}; --kw-border: {border}; --kw-text: {text};"

def stylize_keywords(text):
    """Wrap leading [keywords] in span badges for styling."""
    if not text:
        return text
    match = KEYWORDS_RE.match(text)
    if not match:
        return text
    keywords = [kw.strip() for kw in match.group("keywords").split(",") if kw.strip()]
    if not keywords:
        return text
    badges = "".join(
        f'<span class="keyword-badge" style="{keyword_style(kw)}">{html.escape(kw)}</span>'
        for kw in keywords
    )
    rest = text[match.end():].strip()
    rest_html = html.escape(rest)
    rest_html = f'<span class="keyword-text">{rest_html}</span>' if rest_html else ""
    # TODO
    return Markup(f'<span class="keyword-badges">{badges}</span>{rest_html}')

def make_json_safe(value):
    """Convert data into JSON-serializable types."""
    if isinstance(value, Markup):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value

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
        lowered = value.lower()
        if "trend" in lowered or "now" in lowered:
            return "Now"
        if len(value) <= 8:
            return value
    return None


def build_retry_session():
    retry_kwargs = {
        "total": RETRY_ATTEMPTS,
        "connect": RETRY_ATTEMPTS,
        "read": RETRY_ATTEMPTS,
        "backoff_factor": RETRY_BACKOFF,
        "status_forcelist": RETRY_STATUS_CODES,
        "raise_on_status": False,
        "respect_retry_after_header": True,
    }
    try:
        retries = Retry(**retry_kwargs, allowed_methods=frozenset(["GET", "POST"]))
    except TypeError:
        retries = Retry(**retry_kwargs, method_whitelist=frozenset(["GET", "POST"]))
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP_SESSION = build_retry_session()


def retry_call(label, func, attempts=RETRY_ATTEMPTS, base_delay=1.0, max_delay=8.0):
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt < attempts:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                logger.warning(
                    "%s attempt %s/%s failed: %s; retrying in %.1fs",
                    label,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("%s failed after %s attempts: %s", label, attempts, exc)
    return None


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
        return True
    return False


def prefer_fallback(value, fallback, label):
    if not is_missing(value):
        return value
    if not is_missing(fallback):
        logger.warning("%s unavailable; using cached data", label)
        return fallback
    return value


def load_previous_data(paths=(DATA_OUTPUT_PATH, LEGACY_DATA_OUTPUT_PATH)):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to read prior data from {path}: {e}")
    return {}


def load_jesus_quotes(path=JESUS_QUOTES_PATH):
    if not path or not os.path.exists(path):
        if path:
            logger.warning("Jesus quotes file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        logger.warning("Failed to read Jesus quotes from %s: %s", path, e)
        return []
    if not isinstance(data, dict):
        logger.warning("Jesus quotes file has unexpected format: %s", type(data))
        return []
    return [(ref, text) for ref, text in data.items() if ref and text]


# ==============================================================================
# DATA FETCHING FUNCTIONS
# ==============================================================================

def fetch_weather(fallback=None):
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
    
    def _request():
        response = HTTP_SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        return response.json()

    data = retry_call("Weather fetch", _request)
    if not data:
        return prefer_fallback(None, fallback, "Weather")

    try:
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})

        required_keys = [
            "temperature_2m",
            "precipitation_probability",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
        ]
        if not all(isinstance(hourly.get(key), list) and hourly.get(key) for key in required_keys):
            if not is_missing(fallback):
                logger.warning("Incomplete weather payload; using cached data")
                return fallback

        def clean_numbers(values):
            return [value for value in values if isinstance(value, (int, float))]

        def safe_avg(values):
            numbers = clean_numbers(values or [])
            return sum(numbers) / len(numbers) if numbers else 0

        def safe_max(values):
            numbers = clean_numbers(values or [])
            return max(numbers) if numbers else 0

        def slice_list(values, start, end):
            return values[start:end] if isinstance(values, list) else []

        def get_segment_data(start_h, end_h):
            # Extract slices
            temps = slice_list(hourly.get("temperature_2m"), start_h, end_h)
            precips = slice_list(hourly.get("precipitation_probability"), start_h, end_h)
            winds = slice_list(hourly.get("wind_speed_10m"), start_h, end_h)
            wind_dirs = slice_list(hourly.get("wind_direction_10m"), start_h, end_h)
            codes = slice_list(hourly.get("weather_code"), start_h, end_h)

            # Aggregate
            avg_temp = safe_avg(temps)
            max_precip = safe_max(precips)
            max_wind = safe_max(winds)
            avg_wind_dir = safe_avg(wind_dirs)

            # Most common weather code in segment
            code = max(codes, key=codes.count) if codes else 0

            return {
                "temp": round(avg_temp) if avg_temp else 0,
                "precip": max_precip,
                "wind": round(max_wind) if max_wind else 0,
                "wind_dir": round(avg_wind_dir) if avg_wind_dir else 0,
                "color": get_weather_color(code),
                "condition": get_weather_text(code),
                "icon": get_weather_icon(code),
            }

        return {
            "morning": get_segment_data(6, 12),
            "afternoon": get_segment_data(12, 18),
            "evening": get_segment_data(18, 24),
            "sunrise": (daily.get("sunrise") or [""])[0],
            "sunset": (daily.get("sunset") or [""])[0],
            "daily_precip": safe_max(hourly.get("precipitation_probability") or []),
        }
    except Exception as e:
        logger.error(f"Error parsing weather data: {e}")
        return prefer_fallback(None, fallback, "Weather")


def fetch_stocks(fallback=None):
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
    
    fallback = fallback if isinstance(fallback, dict) else {}
    stock_data = {}
    for name, symbol in tickers.items():
        def _fetch_history():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if hist is None or hist.empty:
                raise ValueError("No price history returned")
            return hist

        hist = retry_call(f"Stock fetch {name}", _fetch_history)
        if hist is None:
            cached = fallback.get(name)
            if cached:
                logger.warning("Using cached stock data for %s", name)
                stock_data[name] = cached
            else:
                stock_data[name] = {"price": 0.0, "change": 0.0, "percent": 0.0, "color": "grey", "arrow": "-"}
            continue

        try:
            current_close = hist["Close"].iloc[-1]
            # Use previous close if available, else standard fallback
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_close

            change = current_close - prev_close
            percent_change = (change / prev_close) * 100 if prev_close else 0.0

            stock_data[name] = {
                "price": current_close,
                "change": change,
                "percent": percent_change,
                "color": "green" if change >= 0 else "red",
                "arrow": "↑" if change >= 0 else "↓",
            }
        except Exception as e:
            logger.error(f"Error parsing {name} data: {e}")
            cached = fallback.get(name)
            if cached:
                logger.warning("Using cached stock data for %s", name)
                stock_data[name] = cached
            else:
                stock_data[name] = {"price": 0.0, "change": 0.0, "percent": 0.0, "color": "grey", "arrow": "-"}
            
    # Calculate average market performance
    percents = [
        data.get("percent")
        for data in stock_data.values()
        if isinstance(data, dict) and isinstance(data.get("percent"), (int, float))
    ]
    stock_data["average_percent"] = sum(percents) / len(percents) if percents else 0.0

    return stock_data


def summarize_with_ai(text, prompt_prefix="Summarize this news item:"):
    """Summarize text using xAI API."""
    if not XAI_API_KEY:
        return text 
    
    # Simple deduplication or length check could go here if needed
    
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    system_prompt = (
        "You are a helpful news assistant. "
        "Merge the news title and description into one concise sentence (max 22 words). "
        "Start with 2-3 bracketed keywords, e.g. [AI, Nvidia, chips]. "
        "Do not filter anything out. Be specific."
    )
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt_prefix}\n\n{text}"},
        ],
    }

    def _summarize():
        response = HTTP_SESSION.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=AI_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if not content:
            raise ValueError("Empty summary response")
        return content

    summary = retry_call("AI summarization", _summarize)
    if not summary:
        logger.error("AI summarization failed; falling back to raw text")
        return text if text else "Content unavailable"
    return summary


def fetch_feed(url):
    def _request():
        response = HTTP_SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        return response.content

    content = retry_call(f"RSS fetch {url}", _request)
    if not content:
        return None

    feed = feedparser.parse(content)
    if getattr(feed, "bozo", False):
        logger.warning("Feed parse warning for %s: %s", url, getattr(feed, "bozo_exception", "unknown"))
    return feed


def fetch_rss_feed(url, limit=5, prompt="Summarize this content:", fallback=None):
    """Generic RSS feed fetcher and summarizer."""
    news_items = []
    feed = fetch_feed(url)
    if not feed or not getattr(feed, "entries", None):
        return prefer_fallback([], fallback, f"RSS feed {url}")

    for entry in feed.entries[:limit]:
        try:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", getattr(entry, "description", "")) or ""
            content_text = f"{title}. {summary}".strip(". ").strip()
            item_summary = summarize_with_ai(content_text, prompt)
            item_summary = stylize_keywords(item_summary)
            if item_summary:
                news_items.append({
                    "headline": item_summary,
                    "link": entry.get("link", ""),
                })
        except Exception as e:
            logger.warning(f"Error parsing feed entry from {url}: {e}")

    if not news_items:
        return prefer_fallback([], fallback, f"RSS feed {url}")
    return news_items


def fetch_world_news(fallback=None):
    """Fetch and summarize world news from BBC."""
    logger.info("Fetching world news...")
    return fetch_rss_feed(
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        limit=10,
        prompt="Summarize this news item:",
        fallback=fallback,
    )


def fetch_space_news(fallback=None):
    """Fetch and summarize space news."""
    logger.info("Fetching space news...")
    return fetch_rss_feed(
        "https://spacenews.com/feed/",
        limit=5,
        prompt="Summarize this content:",
        fallback=fallback,
    )


def fetch_copenhagen_events(fallback=None):
    """Fetch and summarize Copenhagen events/news."""
    logger.info("Fetching Copenhagen events...")
    return fetch_rss_feed(
        "https://cphpost.dk/feed/",
        limit=5,
        prompt="Summarize this content.",
        fallback=fallback,
    )


def fetch_x_trending(fallback=None):
    """Fetch trending topics from X using OAuth1 and personalized_trends API."""
    logger.info("Fetching X trending topics...")
    
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        logger.warning("X Trending: Missing OAuth credentials.")
        return prefer_fallback([], fallback, "X trending")

    try:
        from requests_oauthlib import OAuth1Session
        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret
        )
        
        url = 'https://api.x.com/2/users/personalized_trends'

        def _request():
            response = oauth.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                raise RuntimeError(f"X API error {response.status_code}: {response.text}")
            return response.json()

        payload = retry_call("X trending fetch", _request)
        if not payload:
            return prefer_fallback([], fallback, "X trending")

        raw_data = payload.get('data', [])
        logger.info(f"X API returned {len(raw_data)} items total.")

        trends_data = raw_data[:10]

        formatted_trends = []
        for t in trends_data:
            trend_name = t.get('trend_name') or "N/A"
            formatted_trends.append({
                'name': trend_name,
                'post_count': t.get('post_count') or t.get('tweet_count', 'N/A'),
                'category': t.get('category', 'N/A'),
                'trending_since': format_trending_since(t.get('trending_since')),
                'link': f"https://x.com/search?q={quote(trend_name)}"
            })

        if not formatted_trends:
            return prefer_fallback([], fallback, "X trending")

        logger.info(f"Successfully fetched {len(formatted_trends)} X trending topics")
        return [("Personalized Trends", formatted_trends)]
        
    except ImportError:
        logger.error("requests_oauthlib not installed - cannot fetch X trends")
        return prefer_fallback([], fallback, "X trending")
    except Exception as e:
        logger.error(f"Error fetching X trending: {e}")
        return prefer_fallback([], fallback, "X trending")


# ==============================================================================
# REFRESH & SERVER
# ==============================================================================

def refresh_x_trending(data_path=DATA_OUTPUT_PATH):
    """Refresh X trending data and persist to JSON."""
    logger.info("Refreshing X trending data...")
    data = {}
    if os.path.exists(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as e:
            logger.warning(f"Failed to read existing data file: {e}")
            data = {}

    new_trends = fetch_x_trending(data.get("x_trending"))
    if new_trends:
        data["x_trending"] = new_trends
    elif "x_trending" not in data:
        data["x_trending"] = []

    try:
        safe_data = make_json_safe(data)
        with open(data_path, "w", encoding="utf-8") as file:
            json.dump(safe_data, file, ensure_ascii=True, indent=2)
        logger.info(f"X trending data saved to {data_path}")
    except Exception as e:
        logger.error(f"Error saving X trending data: {e}")
    return data.get("x_trending", [])


class BrevityHandler(SimpleHTTPRequestHandler):
    """Serve static files and handle refresh endpoints."""

    def do_OPTIONS(self):
        path = urlparse(self.path).path
        if path == REFRESH_X_ENDPOINT:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == REFRESH_X_ENDPOINT:
            try:
                trends = refresh_x_trending()
                count = 0
                if isinstance(trends, list):
                    count = sum(len(items) for _, items in trends if isinstance(items, list))
                self._send_json(200, {"status": "ok", "count": count})
            except Exception as e:
                logger.error(f"Refresh failed: {e}")
                self._send_json(500, {"status": "error"})
            return
        self.send_error(404)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def run_server(host="127.0.0.1", port=8000):
    """Run a simple local server for Brevity."""
    handler = functools.partial(BrevityHandler, directory=os.path.dirname(__file__))
    server = ThreadingHTTPServer((host, port), handler)
    logger.info(f"Serving Brevity at http://{host}:{port}")
    logger.info(f"Refresh X endpoint: {REFRESH_X_ENDPOINT}")
    server.serve_forever()


def fetch_quote(fallback=None):
    """Fetch a Quote of the Day (Stoicism/Proverbs) using AI."""
    logger.info("Fetching Quote of the Day...")
    default_fallback = {"text": "The obstacle is the way.", "author": "Marcus Aurelius"}
    cached_fallback = fallback if not is_missing(fallback) else default_fallback
    
    if not XAI_API_KEY:
        return cached_fallback

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        "Generate a short, wise quote from Stoic philosophy or the Book of Proverbs. "
        "Return JSON format: {\"text\": \"Quote text\", \"author\": \"Author Name\"}."
    )
    
    payload = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [
            {"role": "system", "content": "You are a wise assistant. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    def _request():
        response = HTTP_SESSION.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=AI_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return json.loads(content)

    quote = retry_call("Quote fetch", _request)
    if not isinstance(quote, dict):
        return cached_fallback
    if not quote.get("text"):
        return cached_fallback
    return quote


def fetch_jesus_quote(fallback=None, seed_date=None):
    """Select a deterministic Jesus quote from local JSON."""
    logger.info("Selecting Jesus quote...")
    quotes = load_jesus_quotes()
    if not quotes:
        return prefer_fallback(None, fallback, "Jesus quote")
    seed = seed_date or date.today()
    index = (seed.toordinal() - 1) % len(quotes)
    reference, text = quotes[index]
    return {"text": text, "author": reference}


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
        html_out = render_html(data, render_mode="pdf")
        pdf_path = "brevity.pdf"
        HTML(string=html_out, base_url=os.path.dirname(__file__)).write_pdf(pdf_path)
        logger.info(f"PDF generated at {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None


def render_html(data, render_mode="pdf"):
    """Render Brevity HTML using the Jinja2 template."""
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template("brevity_template.html")
    context = {
        **data,
        "data_path": DATA_OUTPUT_PATH,
        "render_mode": render_mode,
        "refresh_x_endpoint": REFRESH_X_ENDPOINT,
    }
    return template.render(**context)


def generate_html(data, output_path=HTML_OUTPUT_PATH):
    """Generate HTML file from data using Jinja2."""
    logger.info("Generating HTML...")
    try:
        html_out = render_html(data, render_mode="html")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html_out)
        logger.info(f"HTML generated at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating HTML: {e}")
        return None


def save_brief_data(data, output_path=DATA_OUTPUT_PATH):
    """Save Brevity data to JSON."""
    logger.info("Saving Brevity data...")
    try:
        safe_data = make_json_safe(data)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(safe_data, file, ensure_ascii=True, indent=2)
        logger.info(f"Brevity data saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving Brevity data: {e}")
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
    
    def _upload():
        client.files_upload_v2(
            channel=SLACK_CHANNEL_ID,
            file=pdf_path,
            title=f"Brevity - {date.today().strftime('%Y-%m-%d')}",
            initial_comment="Here is your update Mr Smith.",
        )
        return True

    result = retry_call("Slack upload", _upload, attempts=3, base_delay=2.0)
    if result:
        logger.info("PDF uploaded to Slack successfully.")
    else:
        logger.error("Slack upload failed after retries.")


def main():
    logger.info("Starting Brevity generation...")
    
    # 1. Gather Data
    today = date.today()
    previous_data = load_previous_data()
    weather = fetch_weather(previous_data.get("weather"))
    stocks = fetch_stocks(previous_data.get("stocks"))
    # Use generic fetcher wrappers for news
    world_news = fetch_world_news(previous_data.get("world_news"))
    space_news = fetch_space_news(previous_data.get("space_news"))
    copenhagen = fetch_copenhagen_events(previous_data.get("copenhagen"))
    x_trending = fetch_x_trending(previous_data.get("x_trending"))
    jesus_quote = fetch_jesus_quote(previous_data.get("jesus_quote"), today)
    quote = fetch_quote(previous_data.get("quote"))
    
    # Calculate Year Percentage
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
        "jesus_quote": jesus_quote,
        "x_trending": x_trending,
        "quote": quote
    }
    
    # 2. Save data + HTML output
    save_brief_data(data)
    generate_html(data)

    # 3. Generate PDF
    pdf_path = generate_pdf(data)
    
    # 4. Send to Slack
    if pdf_path:
        send_to_slack(pdf_path)
        pass
    
    logger.info("Done.")

def cli():
    parser = argparse.ArgumentParser(description="Brevity generator and server.")
    parser.add_argument("--refresh-x", action="store_true", help="Refresh X trending data only.")
    parser.add_argument("--serve", action="store_true", help="Serve Brevity locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local server.")
    args = parser.parse_args()

    if args.refresh_x:
        refresh_x_trending()
        return
    if args.serve:
        run_server(args.host, args.port)
        return
    main()


if __name__ == "__main__":
    cli()
