import datasets
import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import re

print("Loading arguments")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="dataset_path", help="vul-inducing instructions dataset")
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--t", type=float, default=0.6)
parser.add_argument("--col_name", type=str, default="question")
parser.add_argument("--fout", type=str, default="your_output_file_path")
parser.add_argument("--lang", type=str, default="python", choices=["python", "java", "c", "cpp"], help="Language for code generation and extraction.")
parser.add_argument("--clients", type=str, nargs='+', required=True, help="List of clients to start, e.g., http://localhost:8002/v1:4")
parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=-1)
parser.add_argument("--api_key", type=str, default="ABC")
parser.add_argument("--system_prompt", type=str, default="prompt/prompt-python.txt")
args = parser.parse_args()
print(args)


api_key = args.api_key
task_col_name = args.col_name

print(f"Loading dataset from {args.dataset_name}")
ds = datasets.load_dataset('json', data_files=args.dataset_name)['train']
print(f"Loaded {len(ds)} samples")

system_prompt = open(args.system_prompt, "r").read()
print(f"System prompt: {system_prompt}")

messages_list = []
for i, data in enumerate(ds):
    sample_data = data[task_col_name]
    current_message = [
        {"role": "system", "content": f"{system_prompt}\n"},
        {"role": "user", "content": f"{sample_data}\n"},
    ]
    messages_list.append({
        "messages": current_message,
        "idx": i,
        "problem_id": data["problem_id"],
    })

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
    problem_id = prompt['problem_id']
    idx = my_worker_id
    client = clients[idx % len(clients)]

    try:
        completions = client.chat.completions.create(
            model=args.model_name_or_path,
            messages=msg,
            max_tokens=2048,
            temperature=args.t,
            n=args.n,
            top_p=0.9,
        )
    except Exception as e:
        print("Error in worker %d: %s" % (idx, e))
        print("Bad prompt:" + msg[-1]['content'])
        raise e

    return {
        'problem_id': problem_id,
        'prompt': msg[1]['content'],
        'choices': completions.choices,
    }

fout = open(args.fout, "a+")
fout.seek(0)
lines = fout.readlines()
line_no = len(lines)
fout.seek(0, 2)
start_id = args.start_id
end_id = args.end_id
if end_id != -1:
    messages_list = messages_list[start_id:end_id]
else:
    messages_list = messages_list[start_id:]
print(f"Starting from line {start_id} to {end_id}")

worker_id = 0

def worker_init(worker_id_lock):
    global my_worker_id
    with worker_id_lock.get_lock():
        my_worker_id = worker_id_lock.value
        worker_id_lock.value += 1
    print(f"Worker {my_worker_id} started")


ctx = multiprocessing.get_context('spawn')
worker_id = ctx.Value('i', 0)
pool = multiprocessing.Pool(len(clients), worker_init, (worker_id,))
completions_all = pool.imap(request_one, tqdm(messages_list, desc="Requesting", position=0))

def extract_and_format_code_blocks(response_string, language="python"):
    pattern_content = rf"```{re.escape(language)}\n(.*?)\n?```"
    extracted_contents = re.findall(pattern_content, response_string, re.DOTALL)

    raw_code_content = ""

    if len(extracted_contents) >= 1:
        raw_code_content = extracted_contents[0].strip()

    formatted_code_block = f"```{language}\n{raw_code_content}\n```"

    return formatted_code_block

current_id = line_no

code_snippets_all = []
empty_code_block_str = f"```{args.lang}\n\n```"
error_num = 0


for i, completions in enumerate(tqdm(completions_all, total=len(messages_list), desc="Writing", position=1)):
    code_snippets = [c.message.content for c in completions['choices']]
    code_snippets_all.append(code_snippets)
    for code_snippet in code_snippets:
        extracted_code_block = extract_and_format_code_blocks(code_snippet, args.lang)
        
        if code_snippet and extracted_code_block != empty_code_block_str:
            fout.write(json.dumps({"id": current_id, "problem_id": completions['problem_id'], "question": completions['prompt'], "code": extracted_code_block}) + "\n")
            current_id += 1
            fout.flush()
        else:
            error_num += 1

print("All tasks submitted. Closing and joining pool...")
pool.close()
pool.join()
print("Pool closed.")

fout.close()