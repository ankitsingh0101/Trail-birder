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


def field_note(bird):
    prompt = (
        f"You are a friendly field guide. A birder just heard a "
        f"{bird['common']} ({bird['scientific']}). In 3 short sentences: "
        f"describe how it looks, what its sound is like, and one fun fact. "
        f"Keep it simple and warm."
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    audio_file = sys.argv[1]
    bird = identify(audio_file)
    if bird is None:
        print("No bird heard clearly. Try a closer or quieter recording.")
        return

    print(f"\nHeard: {bird['common']} ({bird['scientific']})")
    print(f"Detected in {bird['detections']} clips, best confidence {bird['confidence']:.0%}\n")
    try:
        print(field_note(bird))
    except requests.exceptions.ConnectionError:
        print("Ollama isn't running. Start it with `ollama serve` and try again.")


if __name__ == "__main__":
    main()