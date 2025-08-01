import datasets
import argparse
import json
import multiprocessing
import openai
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import re

parser = argparse.ArgumentParser(description="Batch process requests with multiple clients on multiple ports.")
parser.add_argument("--lang", type=str, default="python", help="Language for code generation.")
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct", help="Path to the model or its name.")
parser.add_argument("--n", type=int, default=5, help="Number of completions to generate for each prompt.")
parser.add_argument("--t", type=float, default=0.8, help="Temperature for sampling.")
parser.add_argument("--col_name", type=str, default="question", help="Column name in the dataset containing the prompt.")
parser.add_argument("--fout", type=str, default="", help="Output file path.")
parser.add_argument("--start-id", type=int, default=0, help="Starting index of the dataset to process.")
parser.add_argument("--end-id", type=int, default=-1, help="Ending index of the dataset to process. -1 for end of dataset.")
parser.add_argument("--ip", type=str, default="localhost", help="IP address of the server.")
parser.add_argument(
    "--port-clients",
    nargs='+',
    required=True,
    help='List of port:client_count pairs. Example: --port-clients 8000:4 8001:2'
)

args = parser.parse_args()
print("--- Script Arguments ---")
print(args)
print("------------------------")


task_col_name = args.col_name

ds = datasets.load_from_disk("../../datasets/APPS")
train_ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)


ds = train_ds["train"]
print(f"{ds.num_rows} rows in the dataset")


prompts_list = []
lang = args.lang

for i, data in enumerate(ds):
    system_content = f"You are a code generator.\nYour responses MUST:\n1. Contain ONLY a single ```{lang} code block\n2.No explanations, comments or text outside the code block\nPlease provide only the final answer directly, without any thinking process or explanation."
    try:
        sample_data = data[task_col_name]
        if not sample_data.get("fn_name"):
            sample_data = "\nUse Standard Input format." + sample_data
        else:
            sample_data = "\nUse Call-Based format." + sample_data
    except:
        sample_data = "\nUse Standard Input format." + sample_data

    user_content = f"Generate {lang} code for: {sample_data}\n"


    current_message = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompts_list.append({
        "messages": current_message,
        "problem_id": data["problem_id"],
        "idx": i,
    })


clients = []
print("Initializing OpenAI clients...")
for pc_pair in args.port_clients:
    try:
        port, num_clients_str = pc_pair.split(':')
        num_clients = int(num_clients_str)
        port = int(port)
        
        if num_clients <= 0:
            print(f"Warning: Client count for port {port} is {num_clients}. Skipping.")
            continue
            
        print(f"Creating {num_clients} client(s) for port {port}...")
        for _ in range(num_clients):
            client = openai.OpenAI(
                base_url=f"http://{args.ip}:{port}/v1",
                api_key="no-key-required",
            )
            clients.append(client)
    except ValueError:
        print(f"Error: Invalid format for --port-clients. Expected 'port:count', but got '{pc_pair}'.")
        exit(1)

if not clients:
    print("Error: No clients were created. Please check your --port-clients argument.")
    exit(1)

print(f"\nTotal clients created: {len(clients)}\n")


def request_one(prompt_data):
    global my_worker_id
    problem_id = prompt_data['problem_id']
    worker_idx = my_worker_id
    client = clients[worker_idx % len(clients)]


    prompt_to_send = prompt_data['messages']
    original_question = prompt_data['messages'][-1]['content']
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    text_for_tokenizing = prompt_to_send if isinstance(prompt_to_send, str) else tokenizer.apply_chat_template(prompt_to_send, tokenize=False)
    prompt_tokens = tokenizer(text_for_tokenizing, return_tensors="pt")
    prompt_len = len(prompt_tokens['input_ids'][0])
    
    if prompt_len > 2048:
        print(f"Prompt too long: {prompt_len}")
        return {
            'problem_id': problem_id,
            'prompt': original_question,
            'choices': []
        }

    try:

            completions = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=prompt_to_send,
                max_tokens=2048,
                temperature=args.t,
                n=args.n,
                top_p=0.9,
            )

            normalized_choices = completions.choices
    except Exception as e:
        print(f"Error in worker {worker_idx}: {e}")
        return {
            'problem_id': problem_id,
            'prompt': original_question,
            'choices': [],
            'error': str(e)
        }
    
    return {
        'problem_id': problem_id,
        'prompt': original_question,
        'choices': normalized_choices,
    }

fout = open(args.fout, "a+")
fout.seek(0)
lines = fout.readlines()
line_no = len(lines)
fout.seek(0, 2)

if args.end_id == -1:
    prompts_list_to_process = prompts_list[args.start_id:]
else:
    prompts_list_to_process = prompts_list[args.start_id:args.end_id]

print(f"Processing {len(prompts_list_to_process)} prompts, starting from original index {args.start_id}.")

worker_id_counter = 0

def worker_init(counter):
    global my_worker_id
    with counter.get_lock():
        my_worker_id = counter.value
        counter.value += 1

def exact_code_block(resp:str, lang:str):
    pattern_content = rf"```{re.escape(lang)}\n(.*?)\n?```"
    extracted_contents = re.findall(pattern_content, resp, re.DOTALL)
    
    raw_code_content = ""
    if extracted_contents:
        raw_code_content = extracted_contents[0].strip()
        
    formatted_code_block = f"{raw_code_content}\n"
    return formatted_code_block

ctx = multiprocessing.get_context('spawn')
worker_id_val = ctx.Value('i', 0)
pool = multiprocessing.Pool(len(clients), initializer=worker_init, initargs=(worker_id_val,))

current_id = line_no

try:
    completions_all = pool.imap(request_one, tqdm(prompts_list_to_process, desc="Requesting", position=0))
    
    for completions in tqdm(completions_all, total=len(prompts_list_to_process), desc="Writing", position=1):
        if not completions or not completions.get('choices'):
            if completions and completions.get('error'):
                print(f"Skipping result for problem_id {completions.get('problem_id')} due to error.")
            continue
            
        responses = [c.message.content for c in completions['choices']]
        for response in responses:
            code_block = exact_code_block(response, args.lang)
            if code_block:
                fout.write(json.dumps({
                    "id": current_id,
                    "problem_id": completions['problem_id'],
                    "question": completions['prompt'],
                    "code": code_block
                }) + "\n")
                current_id += 1
        fout.flush()
finally:
    pool.close()
    pool.join()
    fout.close()
    print("\nProcessing complete. Output saved to", args.fout)