#!/bin/bash

MODEL="your_model_path"
BASE_URL="http://localhost:8002/v1"

echo "Evaluating HumanEval dataset..."
evalplus.evaluate \
    --model "$MODEL" \
    --dataset humaneval \
    --base-url "$BASE_URL" \
    --backend openai \
    --temperature 0.2 \
    --n_samples 10

echo "=================================="
echo "HumanEval evaluation completed"
echo "=================================="