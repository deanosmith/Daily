import os
from jinja2 import Environment, FileSystemLoader

# Mock data
date = "Monday, January 1"
quote = {"text": "The only way out is through.", "author": "Robert Frost"}
weather_valid = {
    "morning": {"icon": "☀️", "temp": 10, "wind": 20, "wind_dir": 180},
    "afternoon": {"icon": "⛅", "temp": 12, "wind": 25, "wind_dir": 190},
    "evening": {"icon": "🌙", "temp": 8, "wind": 15, "wind_dir": 170},
    "sunrise": "2024-01-01T08:00",
    "sunset": "2024-01-01T16:00"
}
weather_none = None

stocks = {"Tesla": {"price": 200, "percent": 5, "color": "green"}}
news = [{"title": "News 1", "link": "#", "summary": "Summary 1"}]

def test_template():
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("daily_brief_template.html")
    
    # Test 1: Full Data
    print("--- Test 1: Full Data ---")
    out1 = template.render(
        date=date, 
        weather=weather_valid, 
        stocks=stocks, 
        world_news=news, 
        space_news=news, 
        copenhagen=news,
        quote=quote
    )
    if "The only way out is through" in out1:
        print("✅ Quote text found")
    else:
        print("❌ Quote text missing")

    if "Robert Frost" in out1 and "font-style: italic" in out1:
        print("✅ Author found with styling")
    else:
        print("❌ Author missing or not styled")
        
    if "font-size: 1.2em" in out1:
        print("✅ Stock font size increased")
    else:
        print("❌ Stock font size checking failed")

    # Test 2: Missing Weather
    print("\n--- Test 2: Missing Weather ---")
    out2 = template.render(
        date=date, 
        weather=None, # SIMULATE ERROR
        stocks=stocks, 
        world_news=news, 
        space_news=news, 
        copenhagen=news,
        quote=quote
    )
    if "ERROR: Weather data unavailable" in out2:
        print("✅ Error message found")
    else:
        print("❌ Error message missing")

if __name__ == "__main__":
    test_template()
