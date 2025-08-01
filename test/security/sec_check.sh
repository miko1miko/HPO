python sec_check.py \ 
    --input_file your_input_file.jsonl \                                     # Input JSONL file containing code snippets to analyze
    --output_file your_output_file.jsonl \                                   # Output JSONL file with security analysis results
    --language python \                                                       # Programming language of the code (python, java, c, etc.)
    --num_workers 32                                                          # Number of parallel worker processes for analysis