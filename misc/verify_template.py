import os
from jinja2 import Environment, FileSystemLoader

# Mock data
date = "Monday, January 1, 2024"
quote = {"text": "The obstacle is the way.", "author": "Marcus Aurelius"}
year_percent = 0.27
x_trending = [{"headline": "AI Breakthroughs", "summary": "New AI models show promise.", "link": "https://x.com/search?q=AI"}]
weather_valid = {
    "morning": {"temp": 10, "wind": 20, "wind_dir": 180, "precip": 10, "color": "#FFD700"},
    "afternoon": {"temp": 12, "wind": 25, "wind_dir": 190, "precip": 20, "color": "#87CEEB"},
    "evening": {"temp": 8, "wind": 15, "wind_dir": 170, "precip": 5, "color": "#708090"},
    "sunrise": "2024-01-01T08:00",
    "sunset": "2024-01-01T16:00",
    "daily_precip": 20
}
weather_none = None

stocks = {"Tesla": {"price": 200, "percent": 5, "color": "green"}, "average_percent": 5}
news = [{"headline": "News 1", "link": "#"}]

def test_template():
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("brevity_template.html")
    
    # Test 1: Full Data
    print("--- Test 1: Full Data ---")
    out1 = template.render(
        date=date, 
        year_percent=year_percent,
        weather=weather_valid, 
        stocks=stocks, 
        world_news=news, 
        space_news=news, 
        copenhagen=news,
        x_trending=x_trending,
        quote=quote
    )
    if quote["text"] in out1:
        print("✅ Quote found")
    else:
        print("❌ Quote missing")
        
    if "font-size: 30.0px" in out1 or "font-size: 34.0px" in out1: # 14 + 20*0.8 = 30. 14 + 25*0.8 = 34.
        print("✅ Dynamic wind size calculated")
    else:
        print("❌ Dynamic wind size check failed (might be float formatting, checking substring...)")
        # Check visually in output if needed
        # partial check
        if "font-size: 3" in out1:
             print("✅ Dynamic wind size likely present")

    # Test 2: Missing Weather
    print("\n--- Test 2: Missing Weather ---")
    out2 = template.render(
        date=date, 
        year_percent=year_percent,
        weather=None,
        stocks=stocks, 
        world_news=news, 
        space_news=news, 
        copenhagen=news,
        x_trending=x_trending,
        quote=quote
    )
    if "ERROR: Weather data unavailable" in out2:
        print("✅ Error message found")
    else:
        print("❌ Error message missing")

    # Test 3: No Quote
    print("\n--- Test 3: No Quote ---")
    out3 = template.render(
        date=date, 
        year_percent=year_percent,
        weather=weather_valid, 
        stocks=stocks, 
        world_news=news, 
        space_news=news, 
        copenhagen=news,
        x_trending=x_trending,
        quote=None
    )
    if "The obstacle is the way" not in out3:
        print("✅ Quote correctly omitted")
    else:
        print("❌ Quote present when it should be hidden")

if __name__ == "__main__":
    test_template()
