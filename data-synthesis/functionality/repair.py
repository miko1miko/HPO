import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, 
                   default="your_input_file.jsonl",
                   help="Input file containing vulnerable code")
parser.add_argument("--repair_model", type=str, 
                   default="Qwen/Qwen2.5-Coder-32B-Instruct")
parser.add_argument("--n", type=int, default=1, help="Number of completions per prompt")
parser.add_argument("--t", type=float, default=0.0, help="Temperature for generation")
parser.add_argument("--output_file", type=str, 
                   default="your_output_file.jsonl",
                   help="Output directory for fixed code")
parser.add_argument("--lang", type=str, choices=['python'], default='python',
                   help="Programming language of the code to be fixed")
parser.add_argument("--start_id", type=int, default=0, help="Start index for processing")
parser.add_argument("--ip", type=str, default="localhost", help="IP address of the server.")
parser.add_argument("--port-clients", type=str, nargs='+', required=True,
                    help="Specify port and client count pairs. Format: 'port1:count1' 'port2:count2' ... (e.g., 8002:4 8003:2)")

args = parser.parse_args()


print(f"args:{args}")

api_key = "YOUR_KEY_HERE"

data = []
with open(args.input_file, 'r') as f:
    for line in f:
        record = json.loads(line)
        if "test" in record:
            record["test"] = [int(x) if isinstance(x, bool) else x for x in record["test"]]
        data.append(record)

ds = Dataset.from_list(data)

print(f"len(ds):{len(ds)}")

def create_cor_prompt(data):
    code = data.get('code', '')
    feedback = data.get('test_result', '')
    question = data.get('question', '')


    system_prompt_template = f"""
Your task is two-fold and requires careful attention to both the main code and its tests to ensure functional correctness:

1.  **Correct the Functionally Incorrect Code:**
    * Analyze the provided "Original (Incorrect) Code" and the "Description of Incorrect Behavior" or "Bug Report".
    * Incorporate all necessary changes to ensure the code behaves correctly according to the "Original problem description" and addresses all identified functional inaccuracies.
    * The fix to the main code should ideally achieve the functional correction with the minimum necessary lines of changes (this includes insertions, deletions, and modifications to existing lines).
    * Ensure the corrected code snippet is complete and functions as specified in the "Original problem description".

2.  **Update the Test Code:**
    * Critically review the "Original test code".
    * Modify the test code so that it accurately and effectively verifies the intended functionality of the **corrected main code** as defined by the "Original problem description".
    * This may involve:
        * Adjusting assertions if expected outputs or behaviors change due to the fix.
        * Updating function calls if signatures are modified (e.g., parameter changes, return type changes).
        * Modifying test setup, mock objects, or input data if the main code's dependencies, internal logic, or data handling are altered by the fix.
        * Ensuring test code is consistent with any logical changes or algorithmic adjustments applied in the main code fix (e.g., if a data structure was changed, an algorithm optimized, or an edge case handled differently, the test code must reflect this for its own setup, calls, or assertions).
        * Adding new test cases to cover aspects of the fix or previously untested scenarios related to the corrected functionality.

Your response MUST contain exactly two separate and complete Python code snippets, each enclosed in its own Markdown code block. No other text, explanations, or summaries should be present outside these blocks (comments within the code are acceptable).

First, provide the corrected main code:
```python
# Your complete corrected Python main code here
```

For both code snippets:
Output the full code, not just the changed parts.
Do not skip lines from the original versions unless the fix or update specifically requires their deletion.
Do not make any changes irrelevant to addressing the vulnerabilities or ensuring test consistency.
"""


    prompt_template = f"""
Context for your task:
The primary goal is to correct functional inaccuracies in the provided Python code based on the problem description and specific feedback. Ensure the accompanying tests are also updated to reflect these corrections accurately.
Original (Incorrect) Code to be corrected:
```{args.lang}
{code}
```
Original problem description or intended functionality:
{question}
The following "Original (Incorrect) Code" does not function as intended or has reported bugs.
Specific feedback or identified issues:
{feedback}
"""
    system_prompt_tokens = tokenizer(system_prompt_template, return_tensors="pt", padding=False, truncation=False)
    prompt_tokens = tokenizer(prompt_template, return_tensors="pt", padding=False, truncation=False)

    system_prompt_len = len(system_prompt_tokens['input_ids'][0])
    prompt_len = len(prompt_tokens['input_ids'][0])
    if system_prompt_len + prompt_len > 4096:
        prompt_template = prompt_template[:4096 - system_prompt_len]
    return system_prompt_template, prompt_template

tokenizer = AutoTokenizer.from_pretrained(args.repair_model)


print(f"create prompt for repair")
messages_list = []


for i, data in enumerate(tqdm(ds, desc="Creating prompts")):
    system_prompt_template, prompt_template = create_cor_prompt(data)
    messages_list.append({
        "messages": [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": prompt_template}
        ],
        "original_data": data
    })

print("\n=== Prompt Statistics ===")
print(f"Total prompts: {len(messages_list)}")
print(f"messages_list's first item: \n{messages_list[0]}")
print(f"create prompt done")

import sys

clients = []
print("\n=== Client Initialization ===")
for port_client_pair in args.port_clients:
    try:
        parts = port_client_pair.split(':')
        if len(parts) != 2:
            raise ValueError("Invalid format. Expected 'port:count'.")
        
        port = parts[0]
        count = int(parts[1])
        
        print(f"Creating {count} client(s) for port {port}...")
        
        for _ in range(count):
            client = openai.OpenAI(
                base_url=f"http://{args.ip}:{port}/v1",
                api_key=api_key,
            )
            clients.append(client)
            
    except ValueError as e:
        print(f"Error processing argument '{port_client_pair}': {e}", file=sys.stderr)
        sys.exit(1)

if not clients:
    print("Error: No clients were created. Please check your --port-clients arguments.", file=sys.stderr)
    sys.exit(1)

print(f"Total clients created: {len(clients)}")
print("===========================\n")

def request_one(prompt):
    global my_worker_id
    msg = prompt['messages']
    idx = my_worker_id
    client = clients[idx % len(clients)]
    try:
        completions = client.chat.completions.create(
            model= args.repair_model,
            messages=msg,
            max_tokens=2048,
            temperature=args.t,
            n=args.n,
            top_p=0.9,
        )
    except Exception as e:
        print(f"Error: {e}")
        return {
            'id': prompt['original_data']['id'],
            'original_data': prompt['original_data'],
            'choices': []
        }

    return {
        'id': prompt['original_data']['id'],
        'original_data': prompt['original_data'],
        'choices': completions.choices
    }

fout = open(args.output_file, "w", encoding='utf-8')
buffer = []
buffer_size = 100

def worker_init(worker_id):
    global my_worker_id
    with worker_id.get_lock():
        my_worker_id = worker_id.value
        worker_id.value += 1
    print(f"Worker {my_worker_id} started")

import re
def extract_and_format_code_block(response_string, language="python"):
    pattern_content = rf"```{re.escape(language)}\n(.*?)\n?```" 
    
    extracted_contents = re.findall(pattern_content, response_string, re.DOTALL)
    
    raw_code_content = ""

    
    if len(extracted_contents) >= 1:
        raw_code_content = extracted_contents[0].strip()
    

        
    formatted_code_block = f"{raw_code_content}\n"
            
    return formatted_code_block

ctx = multiprocessing.get_context('spawn')
worker_id = ctx.Value('i', 0)
pool = multiprocessing.Pool(len(clients), worker_init, (worker_id,))
messages_list = messages_list[args.start_id:]
completions_all = pool.imap(request_one, tqdm(messages_list, desc="Requesting", position=0))

print(f"messages_list's first item: \n{messages_list[args.start_id]}")

for i, completions in enumerate(tqdm(completions_all, total=len(messages_list), desc="Writing", position=1)):
    if completions['choices']:
        response = completions['choices'][0].message.content
        fixed_code = extract_and_format_code_block(response, args.lang)
        if fixed_code:
            output_data = {}
            output_data['id'] = completions['original_data']['id']
            output_data['problem_id'] = completions['original_data']['problem_id']
            output_data['question'] = completions['original_data']['question']
            output_data['code'] = fixed_code
            
            buffer.append(json.dumps(output_data, ensure_ascii=False) + "\n")
            
            if len(buffer) >= buffer_size:
                fout.writelines(buffer)
                buffer.clear()

if buffer:
    fout.writelines(buffer)
    
fout.close()
print(f"\nfinish repair: {args.output_file}")

pool.close()
pool.join()