docker cp  "path/to/your_input_file.jsonl" docker_container_id:/tmp/codeexec/code_wait  # Copy input file to Docker container

python cor_check.py \ 
 --input path/to/your_input_file.jsonl \                                              # Input file containing code to check for correctness
 --output path/to/your_output_file.jsonl \                                            # Output file path for correctness check results
 --language python \                                                          # Programming language of the code to be checked
 --file_name your_input_file \                                                # File name for analysis (without extension)
 --ip http://localhost:80XX \                                                 # IP address for correctness check API
 --threads 2                                                                  # Number of concurrent tasks/threads to run