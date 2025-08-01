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
                   default="Qwen/Qwen2.5-Coder-7B-Instruct")
parser.add_argument("--n", type=int, default=1, help="Number of completions per prompt")
parser.add_argument("--t", type=float, default=0.0, help="Temperature for generation")
parser.add_argument("--output_file", type=str, 
                   default="your_output_file.jsonl",
                   help="Output directory for fixed code")
parser.add_argument("--lang", type=str, choices=['python', 'c', 'java', 'javascript', 'go', 'rust', 'cpp'], default='python',
                    help="Programming language of the code to be fixed")
parser.add_argument("--start_id", type=int, default=0, help="Start index for processing")
parser.add_argument("--clients", type=str, nargs='+', required=True, help="List of clients to start, e.g., http://localhost:8002/v1:2")
parser.add_argument("--api_key", type=str, default="ABC")
args = parser.parse_args()

print(f"args:{args}")

api_key = args.api_key

data = []
with open(args.input_file, 'r') as f:
    for line in f:
        record = json.loads(line)
        data.append(record)

ds = Dataset.from_list(data)

print(f"len(ds):{len(ds)}")

import re

def create_sec_prompt(data):
    code = data.get('code', '')
    cwe_list = data.get('cwe', [])
    issues = data.get('issues', [])
    question = data.get('question', '')
    vulnerability_details = []
        
    desc_pattern = re.compile(r"description=['\"](.*?)['\"]")
    cwe_pattern = re.compile(r"cwe_id=['\"]?([\w-]+)['\"]?")
    line_pattern = re.compile(r"line=(\d+)")
    for issue_str in issues:
        try:
            desc_match = desc_pattern.search(issue_str)
            cwe_match = cwe_pattern.search(issue_str)
            line_match = line_pattern.search(issue_str)

            desc = desc_match.group(1) if desc_match else "N/A"
            cwe = cwe_match.group(1) if cwe_match else "N/A"
            line = line_match.group(1) if line_match else "N/A"

            if desc != "N/A" and line != "N/A":
                vul_info = f"- The code has a CWE vulnerability at line {line}. The vulnerability is of {cwe} type ({desc})."
                vulnerability_details.append(vul_info)
            else:
                print(f"Warning: Issue string not fully parsed (missing desc or line): {issue_str}")
                exit()
        except Exception as e:
            print(f"issues:{issues}")
            print(f"Error parsing issue: {e}")
            continue

    num_vuls = len(cwe_list)
    vul_count = f"{num_vuls} vulnerability" + ("ies" if num_vuls != 1 else "")
    
    num_lines = num_vuls * 5

    system_prompt_template = f"""
Your task is one-fold and requires careful attention to the main code:

1.  **Correct the Vulnerable Code:**
    * Analyze the provided "Vulnerable code" and the vulnerability details.
    * Incorporate all necessary fixes to address a_ll identified vulnerabilities.
    * The fix to the main code should ideally be within approximately {num_lines} lines of changes (this includes insertions, deletions, and modifications to existing lines).
    * Ensure the corrected code snippet is complete and functional according to the "Original problem description".

Your response MUST contain exactly one complete {args.lang} code snippet, enclosed in its own Markdown code block. No other text, explanations, or summaries should be present outside these blocks (comments within the code are acceptable).

First, provide the corrected main code:
```{args.lang}
# Your complete corrected {args.lang} main code here
```

For the code snippet:
Output the full code, not just the changed parts.
Do not skip lines from the original versions unless the fix or update specifically requires their deletion.
Do not make any changes irrelevant to addressing the vulnerabilities.
"""

    prompt_template = f"""
Context for your task:
The following code snippet is reported to have {vul_count} vulnerability/vulnerabilities:
{chr(10).join(vulnerability_details)}
Vulnerable code to fix:
{code}
"""
    system_prompt_tokens = tokenizer(system_prompt_template, return_tensors="pt", padding=False, truncation=False)
    prompt_tokens = tokenizer(prompt_template, return_tensors="pt", padding=False, truncation=False)

    system_prompt_len = len(system_prompt_tokens['input_ids'][0])
    prompt_len = len(prompt_tokens['input_ids'][0])
    if system_prompt_len + prompt_len > 8192:
        prompt_template = prompt_template[:8192 - system_prompt_len]
    return system_prompt_template, prompt_template

tokenizer = AutoTokenizer.from_pretrained(args.repair_model)
def is_secure_code(data):
    if data.get('cwe', []) == []:
        return False
    return True

print(f"create prompt for repair")
messages_list = []
sec_prompt_count = 0
bug_prompt_count = 0
best_code_count = 0

for i, data in enumerate(tqdm(ds, desc="Creating prompts")):
    if is_secure_code(data):
        system_prompt_template, prompt_template = create_sec_prompt(data)
        sec_prompt_count += 1
        messages_list.append({
        "messages": [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": prompt_template}
        ],
        "original_data": data
    })

print("\n=== Prompt Statistics ===")
print(f"Security prompts: {sec_prompt_count}")
print(f"Total prompts: {len(messages_list)}")

print(f"create prompt done")

clients = []
if args.clients:
    for client_info in args.clients:
        try:
            base_url, count_str = client_info.rsplit(':', 1)
            count = int(count_str)
            for _ in range(count):
                clients.append(openai.OpenAI(base_url=base_url, api_key=api_key))
            print(f"Added {count} clients for base_url {base_url}")
        except ValueError:
            print(f"Error: Invalid format for client '{client_info}'. Please use 'base_url:count'.")
            exit(1)
else:
    print("Error: No clients specified. Please use the --clients argument.")
    exit(1)

if not clients:
    print("Error: Client list is empty. Please check your --clients argument.")
    exit(1)

print(f"Total number of clients initialized: {len(clients)}")

def request_one(prompt):
    global my_worker_id
    msg = prompt['messages']
    idx = my_worker_id
    client = clients[idx % len(clients)]
    if msg[-1]['content'] == "best code":
        return {
            'id': prompt['original_data']['id'],
            'original_data': prompt['original_data'],
            'choices': []
        }
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

print(f"Opening output file for appending: {args.output_file}")
fout = open(args.output_file, "a+", encoding='utf-8')

fout.seek(0)
num_existing_lines = len(fout.readlines())
print(f"Found {num_existing_lines} existing records in the output file.")

fout.seek(0, 2)

buffer = []
buffer_size = 100

def worker_init(worker_id):
    global my_worker_id
    with worker_id.get_lock():
        my_worker_id = worker_id.value
        worker_id.value += 1
    print(f"Worker {my_worker_id} started")

import re
def extract_and_format_code_blocks(response_string, language):
    pattern_content = rf"```{re.escape(language)}\n(.*?)\n?```" 
    
    extracted_contents = re.findall(pattern_content, response_string, re.DOTALL)
    
    raw_code_content = ""
    
    if len(extracted_contents) >= 1:
        raw_code_content = extracted_contents[0].strip()

    formatted_code_block = f"```{language}\n{raw_code_content}\n```"
            
    return formatted_code_block

ctx = multiprocessing.get_context('spawn')
worker_id = ctx.Value('i', 0)
pool = multiprocessing.Pool(len(clients), worker_init, (worker_id,))
messages_list = messages_list[args.start_id:]
completions_all = pool.imap(request_one, tqdm(messages_list, desc="Requesting", position=0))

print(f"messages_list's first item: \n{messages_list[0]}")

for i, completions in enumerate(tqdm(completions_all, total=len(messages_list), desc="Writing", position=1)):
    if completions['choices']:
        response = completions['choices'][0].message.content
        fixed_code = extract_and_format_code_blocks(response, args.lang)
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

print("All tasks submitted. Closing and joining pool...")
pool.close()
pool.join()
print("Pool closed.")
    
fout.close()
print(f"\nfinish repair: {args.output_file}")