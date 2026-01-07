import json
import os
import sys
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "verse_history.json")
SEED = date.today().strftime("%A %Y-%m-%d")

if not XAI_API_KEY or not SLACK_WEBHOOK_URL:
    sys.exit("Missing XAI_API_KEY or SLACK_WEBHOOK_URL")


def read_history():
    if not os.path.exists(HISTORY_PATH):
        return set()
    refs = set()
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            combined = (entry.get("full_reference") or "").strip()
            if not combined:
                book = (entry.get("book") or "").strip()
                reference = (entry.get("reference") or "").strip()
                combined = f"{book} {reference}".strip()
            if combined:
                refs.add(combined)
    return refs


def log_reference(full_reference, book, reference):
    entry = {
        "date": date.today().strftime("%d/%m/%Y"),
        "book": book,
        "reference": reference,
        "full_reference": full_reference,
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=True) + "\n")


def ask_for_verses():
    prompt = (
        f"""Using the seed '{SEED}' as the deterministic selection key for today's date,
        Return six NKJV quotes spoken by Jesus.

        Ensure the entire quote is returned if it spans multiple verses.
        Provide the full reference: Book name, verse number(s).
        Include a concise context line explaining the surrounding narrative.
        Provide a brief translation explaining the original language's key words.

        Use the following JSON format example:
        {{"book": "James", "reference": "1:12-15", "text": "Full verse text example here", "context": "James encourages believers to endure trials...", "translation": "Thief [kleptēs - Ancient Greek] a stealer or robber; Life [zōē - Ancient Greek] vital or eternal life;"}}
        """
    )
    payload = {
        "model": "grok-4-fast-reasoning",
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].strip()
        if raw.lower().startswith("json"):
            raw = raw.partition("\n")[2].strip()
    verses = json.loads(raw)
    if not isinstance(verses, list):
        raise ValueError("Model did not return a list")
    return verses


def format_slack_text(verse):
    context = verse.get("context")
    translation = verse.get("translation")

    lines = [f"*{verse['book']} : {verse['reference']}*", "", f"*{verse['text']}*"]

    if context:
        lines.extend(["", context.strip()])

    if translation:
        details = str(translation).strip()
        parts = [part.strip() for part in details.replace("\n", " ").split(";")]
        parts = [part for part in parts if part]
        lines.extend(["", "Translation:", ""])  # blank line after label for readability
        if parts:
            lines.extend(parts)
        else:
            lines.pop()  # remove trailing blank if no parts
    return "\n".join(lines)


def main():
    history = read_history()
    verses = ask_for_verses()

    chosen = None
    for verse in verses:
        book = (verse.get("book") or "").strip()
        reference = (verse.get("reference") or "").strip()
        full_reference = f"{book} {reference}".strip()
        if full_reference and full_reference not in history:
            chosen = {**verse, "book": book, "reference": reference, "full_reference": full_reference}
            break

    if not chosen:
        sys.exit("Generated verses already used. Rerun to try again.")

    requests.post(SLACK_WEBHOOK_URL, json={"text": format_slack_text(chosen)}).raise_for_status()
    log_reference(chosen["full_reference"], chosen["book"], chosen["reference"])


if __name__ == "__main__":
    main()