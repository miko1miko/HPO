python gen_vul_inst.py \ 
--ds-path "bigcode/self-oss-instruct-sc2-exec-filter-50k" \  # Dataset path for instruction generation
--model-name "deepseek-chat" \                               # Model name to use for generation
--base-url "https://api.deepseek.com/" \                     # API base URL for the model service
--sys-prompt-in "cwe-eclicitors/python/cwe-78-python.txt" \  # System prompt file for vulnerability instruction
--fout "vul-inst/python/cwe-78-python.jsonl" \               # Output file path for generated instructions
--from-idx 0 \                                               # Starting index of dataset samples
--to-idx 10 \                                                # Ending index of dataset samples
--api-key-file "api-key.txt"                                 # File containing API key for authentication