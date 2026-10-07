import sys
import birdnet
import pandas as pd

audio_file = sys.argv[1]

model = birdnet.load("acoustic", "3.0", "onnx")
predictions = model.predict(audio_file)
predictions.to_csv("predictions.csv")

df = pd.read_csv("predictions.csv")
print(df.columns.tolist())
print(df.head(10))