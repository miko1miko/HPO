CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server\
 --model  Qwen/Qwen2.5-Coder-7B-Instruct \
 --port 8002 --dtype bfloat16 --max-model-len 5000