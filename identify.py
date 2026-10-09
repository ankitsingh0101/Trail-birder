import datetime
import hashlib
import json
import sys
from urllib.parse import quote

import birdnet
import pandas as pd
import requests

MIN_CONFIDENCE = 0.5
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
LIFE_LIST_FILE = "life_list.json"
FACTS_FILE = "facts.json"
CACHE_FILE = "facts_cache.json"
NOTES_FILE = "notes_cache.json"
WIKI_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = birdnet.load("acoustic", "3.0", "onnx")
    return _model


def identify(audio_file):
    model = get_model()
    predictions = model.predict(audio_file)
    predictions.to_csv("predictions.csv")

    # No bird detected at all can produce an empty file
    try:
        df = pd.read_csv("predictions.csv")
    except pd.errors.EmptyDataError:
        return None

    if df.empty or "confidence" not in df.columns:
        return None

    df = df[df["confidence"] >= MIN_CONFIDENCE]
    if df.empty:
        return None

    summary = (
        df.groupby("species_name")["confidence"]
        .agg(detections="count", best="max")
        .sort_values(["detections", "best"], ascending=False)
    )
    scientific, common = summary.index[0].split("_", 1)
    row = summary.iloc[0]
    return {
        "scientific": scientific,
        "common": common,
        "detections": int(row["detections"]),
        "confidence": float(row["best"]),
    }


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_life_list(bird):
    life_list = load_json(LIFE_LIST_FILE)

    today = datetime.date.today().isoformat()
    entry = life_list.get(bird["scientific"])
    is_new = entry is None
    if is_new:
        entry = {"common": bird["common"], "first_heard": today, "times_heard": 0}
    entry["times_heard"] += 1
    entry["last_heard"] = today
    life_list[bird["scientific"]] = entry

    save_json(LIFE_LIST_FILE, life_list)
    return is_new, len(life_list)


def fetch_wikipedia(bird):
    headers = {"User-Agent": "TrailBirder/1.0 (open-source hackathon project)"}
    for name in (bird["scientific"], bird["common"]):
        url = WIKI_URL.format(quote(name.replace(" ", "_")))
        try:
            r = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException:
            return None
        if r.status_code != 200:
            continue
        data = r.json()
        if data.get("type") == "standard" and data.get("extract"):
            return {
                "extract": data["extract"],
                "source": data["content_urls"]["desktop"]["page"],
            }
    return None


def get_facts(bird):
    # 1. Hand-verified facts always win
    verified = load_json(FACTS_FILE).get(bird["scientific"])
    if verified:
        return verified

    # 2. Previously fetched facts (works offline)
    cache = load_json(CACHE_FILE)
    if bird["scientific"] in cache:
        return cache[bird["scientific"]]

    # 3. Fetch once from Wikipedia, then cache
    fetched = fetch_wikipedia(bird)
    if fetched:
        cache[bird["scientific"]] = fetched
        save_json(CACHE_FILE, cache)
    return fetched


def field_note(bird):
    facts = get_facts(bird)
    if facts is None:
        return "(No facts found for this species. Connect to the internet once to fetch them.)"

    # Fingerprint of the facts used: if they change, the cached note is stale
    fingerprint = hashlib.md5(
        json.dumps(facts, sort_keys=True).encode("utf-8")
    ).hexdigest()

    notes = load_json(NOTES_FILE)
    cached = notes.get(bird["scientific"])
    if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
        return cached["note"]

    intro = (
        f"You are a friendly field guide. Write 3 short, warm sentences about "
        f"the {bird['common']} for a birder who just heard it. "
        f"Start directly with the bird, with no greeting and no 'Okay'. "
    )

    if "extract" in facts:
        prompt = (
            intro
            + "Use ONLY information from this text and add nothing else:\n"
            + facts["extract"]
        )
        suffix = f"\n\n(Auto-fetched from {facts['source']}, not hand-verified.)"
    else:
        prompt = (
            intro
            + "Use ONLY these facts and add nothing else:\n"
            + f"Appearance: {facts['appearance']}\n"
            + f"Sound: {facts['sound']}\n"
            + f"Fun fact: {facts['fun_fact']}"
        )
        suffix = ""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    note = response.json()["response"].strip() + suffix

    notes[bird["scientific"]] = {"fingerprint": fingerprint, "note": note}
    save_json(NOTES_FILE, notes)
    return note


def main():
    if len(sys.argv) < 2:
        print("Usage: python identify.py <audio file>")
        return

    audio_file = sys.argv[1]
    bird = identify(audio_file)
    if bird is None:
        print("No bird heard clearly. Try a closer or quieter recording.")
        return

    print(f"\nHeard: {bird['common']} ({bird['scientific']})")
    print(f"Detected in {bird['detections']} clips, best confidence {bird['confidence']:.0%}\n")

    is_new, total = update_life_list(bird)
    if is_new:
        print(f"New bird for your life list! You now have {total} species.\n")
    else:
        print(f"Already on your life list. You have {total} species.\n")

    try:
        print(field_note(bird))
    except requests.exceptions.ConnectionError:
        print("Ollama isn't running. Start it with `ollama serve` and try again.")


if __name__ == "__main__":
    main()