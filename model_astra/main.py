# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

result = pipe("image.jpg")

print(result)
