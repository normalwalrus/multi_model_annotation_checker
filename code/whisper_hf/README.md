# Evaluating Whisper (HuggingFace Implementation)

## Build
```
docker build -f Dockerfile -t asr-eval/whisper-hf:1.0.0 .
```

## Model Directories
You can download MMS-trained fastconformer and lm here: [https://drive.google.com/drive/folders/1F1Lpqg1fF06VxefNB1qcFIM-6XuOr3RB]

You will need the following files saved in the same output directory:

```
# Model things
config.json (from checkpoint-17100)
generation_config.json (from checkpoint-17100)
preprocessor_config.json (from checkpoint-17100)
pytorch_model.bin.index.json (from checkpoint-17100)
pytorch_model-00001-of-00002.bin (from checkpoint-17100)
pytorch_model-00002-of-00002.bin (from checkpoint-17100)

# Tokenizer things
added_tokens.json
merges.txt
normalizer.json
special_tokens_map.json
tokenizer_config.json
vocab.json
```
