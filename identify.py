import json
import datetime
import sys

import birdnet
import pandas as pd
import requests

MIN_CONFIDENCE = 0.5
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"


def identify(audio_file):
    model = birdnet.load("acoustic", "3.0", "onnx")
    predictions = model.predict(audio_file)
    predictions.to_csv("predictions.csv")

    df = pd.read_csv("predictions.csv")
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


def load_facts():
    with open("facts.json", encoding="utf-8") as f:
        return json.load(f)


def field_note(bird):
    facts = load_facts().get(bird["scientific"])
    if facts is None:
        return "(No verified facts for this species yet, so no field note.)"

    prompt = (
        f"You are a friendly field guide. Write 3 short, warm sentences about "
        f"the {bird['common']} for a birder who just heard it. "
        f"Use ONLY these facts and add nothing else:\n"
        f"Appearance: {facts['appearance']}\n"
        f"Sound: {facts['sound']}\n"
        f"Fun fact: {facts['fun_fact']}"
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

LIFE_LIST_FILE = "life_list.json"

def update_life_list(bird):
    try:
        with open(LIFE_LIST_FILE, encoding="utf-8") as f:
            life_list = json.load(f)
    except FileNotFoundError:
        life_list = {}

    today = datetime.date.today().isoformat()
    entry = life_list.get(bird["scientific"])
    is_new = entry is None
    if is_new:
        entry = {"common": bird["common"], "first_heard": today, "times_heard": 0}
    entry["times_heard"] += 1
    entry["last_heard"] = today
    life_list[bird["scientific"]] = entry

    with open(LIFE_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(life_list, f, indent=2, ensure_ascii=False)

    return is_new, len(life_list)

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