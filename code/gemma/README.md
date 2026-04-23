# Evaluating Gemma (HuggingFace Implementation)

## Build
```
docker compose build gemma
```

## Model Directories
There is no need to download the model weights as it will be baked in during build time


## Methods
From GemmaEvaluator class in transcribe.py
1. infer -> transcription inference 
2. infer_translation -> Translation inference
