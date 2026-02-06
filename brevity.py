import os
import ssl
import sys
import re
import html
import hashlib
import json
import logging
import time
import requests
import calendar
from urllib.parse import quote
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
# CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

# Ensure WeasyPrint can locate system libraries on macOS.
if sys.platform == "darwin":
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
        "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    )

# Allow HTTPS requests in environments with legacy SSL setups.
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context


# ==============================================================================
# CONSTANTS & MAPPINGS
# ==============================================================================

WEATHER_COLORS = {
    0: "#FFD700",  # Sun / Clear (Gold)
    1: "#87CEEB",
    2: "#87CEEB",
    3: "#87CEEB",  # Partly Cloudy (Sky Blue)
    45: "#708090",
    48: "#708090",  # Fog (Slate Gray)
    51: "#4682B4",
    53: "#4682B4",
    55: "#4682B4",  # Drizzle (Steel Blue)
    61: "#4682B4",
    63: "#4682B4",
    65: "#4682B4",  # Rain
    80: "#4682B4",
    81: "#4682B4",
    82: "#4682B4",  # Showers
    71: "#E0FFFF",
    73: "#E0FFFF",
    75: "#E0FFFF",
    77: "#E0FFFF",  # Snow (Light Cyan)
    95: "#9370DB",
    96: "#9370DB",
    99: "#9370DB",  # Thunderstorm (Medium Purple)
}

WEATHER_TEXT = {
    0: "Clear Sky",
    1: "Partly Cloudy",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Rime Fog",
    51: "Light Drizzle",
    53: "Drizzle",
    55: "Heavy Drizzle",
    61: "Light Rain",
    63: "Rain",
    65: "Heavy Rain",
    80: "Showers",
    81: "Showers",
    82: "Showers",
    71: "Light Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}

WEATHER_ICONS = {
    0: "☀️",  # Clear
    1: "⛅️",
    2: "⛅️",
    3: "☁️",  # Cloudy
    45: "🌫️",
    48: "🌫️",  # Fog
    51: "🌦️",
    53: "🌦️",
    55: "🌧️",  # Drizzle
    61: "🌧️",
    63: "🌧️",
    65: "🌧️",  # Rain
    80: "🌦️",
    81: "🌦️",
    82: "🌦️",  # Showers
    71: "🌨️",
    73: "🌨️",
    75: "🌨️",
    77: "🌨️",  # Snow
    95: "⛈️",
    96: "⛈️",
    99: "⛈️",  # Thunderstorm
}

STOCK_BASELINES = {
    "S&P 500": {"price": 99.22, "currency": "EUR"},
    "Tesla": {"price": 339.0, "currency": "USD"},
    "Nvidia": {"price": 152.19, "currency": "USD"},
    "Bitcoin": {"price": 712262.0, "currency": "DKK"},
    "United Health": {"price": 301.0, "currency": "USD"},
    "Echo Star": {"price": 82.90, "currency": "USD"},
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

PLASMA_STOPS = (
    (0.0, "#0d0887"),
    (0.25, "#6a00a8"),
    (0.5, "#b12a90"),
    (0.75, "#e16462"),
    (1.0, "#f0f921"),
)

JESUS_QUOTES_PATH = "resources/jesus.json"
DEFAULT_HEADERS = {"User-Agent": "Brevity/1.0"}
REQUEST_TIMEOUT = 30
AI_TIMEOUT = 60
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.2
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


# ==============================================================================
# HELPERS
# ==============================================================================


def get_weather_color(code):
    """Return a hex color for WMO weather codes."""
    return WEATHER_COLORS.get(code, "#AAAAAA")


def get_weather_text(code):
    """Return description text for WMO weather codes."""
    return WEATHER_TEXT.get(code, "Unknown")


def get_weather_icon(code):
    """Return an icon for WMO weather codes."""
    return WEATHER_ICONS.get(code, "?")


def _hex_to_rgb(value):
    value = value.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def plasma_color(value, vmin=0.0, vmax=40.0):
    """Map a numeric value onto the Plasma colormap."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = vmin
    if vmax == vmin:
        return PLASMA_STOPS[-1][1]
    t = (numeric - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    for idx in range(len(PLASMA_STOPS) - 1):
        left_t, left_color = PLASMA_STOPS[idx]
        right_t, right_color = PLASMA_STOPS[idx + 1]
        if t <= right_t:
            if right_t == left_t:
                return right_color
            local = (t - left_t) / (right_t - left_t)
            r0, g0, b0 = _hex_to_rgb(left_color)
            r1, g1, b1 = _hex_to_rgb(right_color)
            r = int(round(r0 + (r1 - r0) * local))
            g = int(round(g0 + (g1 - g0) * local))
            b = int(round(b0 + (b1 - b0) * local))
            return _rgb_to_hex((r, g, b))
    return PLASMA_STOPS[-1][1]


def keyword_style(keyword):
    """Create a deterministic CSS custom-property string for a keyword badge."""
    digest = hashlib.md5(keyword.lower().encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(KEYWORD_COLOR_PALETTE)
    bg, border, text = KEYWORD_COLOR_PALETTE[index]
    return f"--kw-bg: {bg}; --kw-border: {border}; --kw-text: {text};"

def to_pascal_case(value):
    """Convert a keyword into PascalCase for display."""
    if not value:
        return value
    parts = re.findall(r"[A-Za-z0-9]+", value)
    if not parts:
        return value
    formatted = []
    for part in parts:
        if part.isdigit():
            formatted.append(part)
            continue
        lower = part.lower()
        formatted.append(lower[0].upper() + lower[1:] if lower else "")
    return "".join(formatted)


def stylize_keywords(text):
    """Wrap leading [keywords] in styled span badges."""
    if not text:
        return text
    match = KEYWORDS_RE.match(text)
    if not match:
        return text
    keywords = [kw.strip() for kw in match.group("keywords").split(",") if kw.strip()]
    if not keywords:
        return text
    badges = "".join(
        f'<span class="keyword-badge" style="{keyword_style(kw)}">{html.escape(to_pascal_case(kw))}</span>'
        for kw in keywords
    )
    rest = text[match.end() :].strip()
    rest_html = html.escape(rest)
    rest_html = f'<span class="keyword-text">{rest_html}</span>' if rest_html else ""
    return Markup(f'<span class="keyword-badges">{badges}</span>{rest_html}')


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
    """Retry a callable with exponential backoff and logging."""
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


def load_jesus_quotes(path=JESUS_QUOTES_PATH):
    """Load the local Jesus quotes JSON file into a list of tuples."""
    if not path or not os.path.exists(path):
        if path:
            logger.warning("Jesus quotes file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        logger.warning("Failed to read Jesus quotes from %s: %s", path, exc)
        return []
    if not isinstance(data, dict):
        logger.warning("Jesus quotes file has unexpected format: %s", type(data))
        return []
    return [(ref, text) for ref, text in data.items() if ref and text]


# ==============================================================================
# DATA FETCHING
# ==============================================================================


def fetch_weather():
    """Fetch weather for Copenhagen using Open-Meteo with an hourly breakdown."""
    logger.info("Fetching weather...")
    lat, lon = 55.6761, 12.5683
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "apparent_temperature",
            "precipitation_probability",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
        ],
        "daily": ["sunrise", "sunset"],
        "timezone": "Europe/Berlin",
        "forecast_days": 1,
    }

    def _request():
        response = HTTP_SESSION.get(
            url, params=params, timeout=REQUEST_TIMEOUT, headers=DEFAULT_HEADERS
        )
        response.raise_for_status()
        return response.json()

    data = retry_call("Weather fetch", _request)
    if not data:
        return None

    try:
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})

        required_keys = [
            "temperature_2m",
            "apparent_temperature",
            "precipitation_probability",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
        ]
        if not all(isinstance(hourly.get(key), list) and hourly.get(key) for key in required_keys):
            logger.warning("Incomplete weather payload; skipping weather section")
            return None

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
            temps = slice_list(hourly.get("temperature_2m"), start_h, end_h)
            feels = slice_list(hourly.get("apparent_temperature"), start_h, end_h)
            precips = slice_list(hourly.get("precipitation_probability"), start_h, end_h)
            winds = slice_list(hourly.get("wind_speed_10m"), start_h, end_h)
            wind_dirs = slice_list(hourly.get("wind_direction_10m"), start_h, end_h)
            codes = slice_list(hourly.get("weather_code"), start_h, end_h)

            avg_temp = safe_avg(temps)
            avg_feels = safe_avg(feels)
            max_precip = safe_max(precips)
            max_wind = safe_max(winds)
            avg_wind_dir = safe_avg(wind_dirs)
            code = max(codes, key=codes.count) if codes else 0

            return {
                "temp": round(avg_temp) if avg_temp else 0,
                "feels_like": round(avg_feels) if avg_feels else 0,
                "precip": max_precip,
                "wind": round(max_wind) if max_wind else 0,
                "wind_dir": round(avg_wind_dir) if avg_wind_dir else 0,
                "wind_color": plasma_color(max_wind),
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
    except Exception as exc:
        logger.error("Error parsing weather data: %s", exc)
        return None


def fetch_stocks():
    """Fetch stock data using yfinance."""
    logger.info("Fetching stocks...")
    tickers = {
        "S&P 500": "VUSA.AS",
        "Tesla": "TSLA",
        "Nvidia": "NVDA",
        "Bitcoin": "BTC-USD",
        "United Health": "UNH",
        "Echo Star": "SATS",
    }

    fx_cache = {}

    def fetch_fx_rate(symbol):
        if symbol in fx_cache:
            return fx_cache[symbol]

        def _fetch_history():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist is None or hist.empty:
                raise ValueError("No FX history returned")
            return hist

        hist = retry_call(f"FX fetch {symbol}", _fetch_history)
        if hist is None:
            fx_cache[symbol] = 0.0
            return 0.0
        rate = hist["Close"].iloc[-1]
        fx_cache[symbol] = rate
        return rate

    def convert_usd_to(currency, price):
        if currency == "USD":
            return price
        if currency == "EUR":
            rate = fetch_fx_rate("EURUSD=X")
            return price / rate if rate else price
        if currency == "DKK":
            rate = fetch_fx_rate("USDDKK=X")
            return price * rate if rate else price
        return price

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
            stock_data[name] = {
                "price": 0.0,
                "change": 0.0,
                "percent": 0.0,
                "return_percent": 0.0,
                "color": "grey",
                "arrow": "-",
            }
            continue

        try:
            current_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_close
            change = current_close - prev_close
            percent_change = (change / prev_close) * 100 if prev_close else 0.0
            return_percent = 0.0
            baseline = STOCK_BASELINES.get(name)
            if baseline:
                baseline_price = baseline.get("price", 0.0)
                baseline_currency = baseline.get("currency", "USD")
                current_in_baseline = convert_usd_to(baseline_currency, current_close)
                if baseline_price:
                    return_percent = ((current_in_baseline - baseline_price) / baseline_price) * 100

            stock_data[name] = {
                "price": current_close,
                "change": change,
                "percent": percent_change,
                "return_percent": return_percent,
                "color": "green" if change >= 0 else "red",
                "arrow": "↑" if change >= 0 else "↓",
            }
        except Exception as exc:
            logger.error("Error parsing %s data: %s", name, exc)
            stock_data[name] = {
                "price": 0.0,
                "change": 0.0,
                "percent": 0.0,
                "return_percent": 0.0,
                "color": "grey",
                "arrow": "-",
            }

    percents = [
        data.get("percent")
        for data in stock_data.values()
        if isinstance(data, dict) and isinstance(data.get("percent"), (int, float))
    ]
    stock_data["average_percent"] = sum(percents) / len(percents) if percents else 0.0

    return stock_data


def summarize_with_ai(text, prompt_prefix="Summarize this news item:"):
    """Summarize text using the xAI API."""
    if not XAI_API_KEY:
        return text

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


def fetch_rss_feed(url, limit=5, prompt="Summarize this content:"):
    """Fetch and summarize items from an RSS feed."""
    news_items = []
    feed = fetch_feed(url)
    if not feed or not getattr(feed, "entries", None):
        return []

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
        except Exception as exc:
            logger.warning("Error parsing feed entry from %s: %s", url, exc)

    return news_items


def fetch_world_news():
    """Fetch and summarize world news from BBC."""
    logger.info("Fetching world news...")
    return fetch_rss_feed(
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        limit=10,
        prompt="Summarize this news item:",
    )


def fetch_space_news():
    """Fetch and summarize space news."""
    logger.info("Fetching space news...")
    return fetch_rss_feed(
        "https://spacenews.com/feed/",
        limit=5,
        prompt="Summarize this content:",
    )


def fetch_copenhagen_events():
    """Fetch and summarize Copenhagen events/news."""
    logger.info("Fetching Copenhagen events...")
    return fetch_rss_feed(
        "https://cphpost.dk/feed/",
        limit=5,
        prompt="Summarize this content.",
    )


def fetch_x_trending():
    """Fetch trending topics from X using OAuth1 and personalized_trends API."""
    logger.info("Fetching X trending topics...")

    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        logger.warning("X Trending: Missing OAuth credentials.")
        return []

    try:
        from requests_oauthlib import OAuth1Session

        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret,
        )

        url = "https://api.x.com/2/users/personalized_trends"

        def _request():
            response = oauth.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                raise RuntimeError(f"X API error {response.status_code}: {response.text}")
            return response.json()

        payload = retry_call("X trending fetch", _request)
        if not payload:
            return []

        raw_data = payload.get("data", [])
        logger.info("X API returned %s items total.", len(raw_data))

        trends_data = raw_data[:20]
        formatted_trends = []
        for trend in trends_data:
            trend_name = trend.get("trend_name") or "N/A"
            formatted_trends.append(
                {
                    "name": trend_name,
                    "post_count": trend.get("post_count") or trend.get("tweet_count", "N/A"),
                    "category": trend.get("category", "N/A"),
                    "trending_since": format_trending_since(trend.get("trending_since")),
                    "link": f"https://x.com/search?q={quote(trend_name)}",
                }
            )

        if not formatted_trends:
            return []

        logger.info("Successfully fetched %s X trending topics", len(formatted_trends))
        return [("Personalized Trends", formatted_trends)]

    except ImportError:
        logger.error("requests_oauthlib not installed - cannot fetch X trends")
        return []
    except Exception as exc:
        logger.error("Error fetching X trending: %s", exc)
        return []


def fetch_quote():
    """Fetch a quote of the day (Stoicism/Proverbs) using AI."""
    logger.info("Fetching Quote of the Day...")
    fallback = {"text": "The obstacle is the way.", "author": "Marcus Aurelius"}

    if not XAI_API_KEY:
        return fallback

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
    if not isinstance(quote, dict) or not quote.get("text"):
        return fallback
    return quote


def fetch_jesus_quote(seed_date=None):
    """Select a deterministic Jesus quote from local JSON."""
    logger.info("Selecting Jesus quote...")
    quotes = load_jesus_quotes()
    if not quotes:
        return None
    seed = seed_date or date.today()
    index = (seed.toordinal() - 1) % len(quotes)
    reference, text = quotes[index]
    return {"text": text, "author": reference}


# ==============================================================================
# PDF GENERATION & DELIVERY
# ==============================================================================


def render_html(data):
    """Render the Jinja2 template for PDF generation."""
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template("brevity_template.html")
    return template.render(**data)


def generate_pdf(data):
    """Generate PDF from the HTML template using WeasyPrint."""
    logger.info("Generating PDF...")
    try:
        from weasyprint import HTML
    except Exception as exc:
        logger.error("WeasyPrint import failed: %s", exc)
        logger.error("PDF generation skipped. Install WeasyPrint system dependencies.")
        return None
    try:
        html_out = render_html(data)
        pdf_path = "brevity.pdf"
        HTML(string=html_out, base_url=os.path.dirname(__file__)).write_pdf(pdf_path)
        logger.info("PDF generated at %s", pdf_path)
        return pdf_path
    except Exception as exc:
        logger.error("Error generating PDF: %s", exc)
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

    today = date.today()
    weather = fetch_weather()
    stocks = fetch_stocks()
    world_news = fetch_world_news()
    space_news = fetch_space_news()
    copenhagen = fetch_copenhagen_events()
    x_trending = fetch_x_trending()
    jesus_quote = fetch_jesus_quote(today)
    quote = fetch_quote()

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
        "quote": quote,
    }

    pdf_path = generate_pdf(data)
    if pdf_path:
        send_to_slack(pdf_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
