python repair.py \ 
 --input_file your_input_file.jsonl \                         # Input file containing vulnerable code to repair
 --repair_model Qwen/Qwen2.5-Coder-32B-Instruct \            # Model name/path for code repair
 --output_file your_output_file.jsonl \                       # Output file path for repaired code
 --lang python \                                              # Programming language of the code to be fixed
 --start_id 0 \                                               # Start index for processing dataset samples
 --port-clients 8002:2 \                                      # Port:client_count pairs for parallel processing
 --ip localhost                                               # IP address of the server