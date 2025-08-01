python repair.py \
 --input_file your_input_file.jsonl \                                        # Input JSONL file containing vulnerable code to repair
 --output_file your_output_file.jsonl \                                      # Output JSONL file for repaired/fixed code
 --lang python \                                                              # Programming language of the code to be fixed
 --start_id 0 \                                                               # Starting index for processing dataset samples
 --repair_model "Qwen/Qwen2.5-Coder-7B-Instruct" \                           # Model name/path for code repair and vulnerability fixing
 --clients http://localhost:8002/v1:2                                         # Client endpoints in format base_url:count for parallel processing