#!/bin/bash

BASE_URL="http://localhost:8002/v1"
MODEL_PATH="your_model_path"

export base_url=$BASE_URL

echo "Evaluating Python dataset..."
python3 -m CybersecurityBenchmarks.benchmark.run \
    --benchmark=instruct \
    --prompt-path="PurpleLlama/CybersecurityBenchmarks/datasets/instruct/instruct-python.json" \
    --response-path="your_output_file.json" \
    --stat-path="your_output_file.json" \
    --llm-under-test="OPENAI::$MODEL_PATH::ABC" \
    --num-queries-per-prompt 10 \
    --run-llm-in-parallel=16

echo "=================================="
echo "Python evaluation completed"
echo "=================================="