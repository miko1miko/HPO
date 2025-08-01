import argparse
import os
import json
from tqdm import tqdm
import openai
import datasets
import pickle
import math
import threading
from queue import Queue
import traceback

parser = argparse.ArgumentParser(description="generate vul instructions")
parser.add_argument("--ds-path", type=str, default="bigcode/self-oss-instruct-sc2-exec-filter-50k")
parser.add_argument("--model-name", type=str, default="deepseek-chat")
parser.add_argument("--base-url", type=str, default="https://api.deepseek.com/")
parser.add_argument("--sys-prompt-in", type=str, default="cwe-eclicitors/python/cwe-78-python.txt")
parser.add_argument("--fout", type=str, default="vul_inst/python/cwe-78-python.jsonl")
parser.add_argument("--from-idx", type=int, default=0)
parser.add_argument("--to-idx", type=int, default=10)
parser.add_argument("--api-key-file", type=str, default="api-key.txt")
args = parser.parse_args()

print(args)

api_keys_files = [args.api_key_file]
api_keys = [open(f, "r").read().strip() for f in api_keys_files]

ds = datasets.load_dataset(args.ds_path, split="train")

to_idx = min(args.to_idx, len(ds))
print("There are total {} samples.".format(len(ds)))
print("Selecting samples from {} to {}.".format(args.from_idx, to_idx))
ds_selected = ds.select(range(args.from_idx, to_idx))
print(f"ds_selected's first item: {ds_selected[0]}")

num_clients = 1
print(f"Initializing for {num_clients} clients...")
total_samples = len(ds_selected)
samples_per_client = math.ceil(total_samples / num_clients)
ds_splits = []
for i in range(num_clients):
    start_idx = i * samples_per_client
    end_idx = min((i + 1) * samples_per_client, total_samples)
    if start_idx < end_idx:
        ds_splits.append(ds_selected.select(range(start_idx, end_idx)))

print(f"Total samples: {total_samples}")
print(f"Samples per client (calculated): {samples_per_client}")
for i, split in enumerate(ds_splits):
    print(f"Client {i} samples: {len(split)}")


class ModelPrompter:
    def query(self, model_name, messages, system_prompt, temperature, max_tokens, n):
        raise NotImplementedError

class OpenAIModelPrompter(ModelPrompter):
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def query(self, messages, system_prompt, temperature, max_tokens, n):
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        results = []
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error in OpenAI query: {e}")
            traceback.print_exc()
            results.append(None)
        
        return results

sys_prompt = open(args.sys_prompt_in, "r").read()
print(f"sys_prompt: {sys_prompt[:200]}...")

prompters = []
for i in range(num_clients):
    api_key_to_use = api_keys[i % len(api_keys)]
    prompter = OpenAIModelPrompter(
        api_key_to_use,
        args.base_url,
        args.model_name
    )
    prompters.append(prompter)
    print(f"Prompter {i} created.")


def request_one(prompt, prompter):
    msg = prompt["messages"]
    idx = prompt["ori_id"]
    sys_prompt = prompt["sys_prompt"]
    try:
        completions = prompter.query(
            messages=msg,
            system_prompt=sys_prompt,
            temperature=0.8,
            max_tokens=4096,
            n=1,
        )
    except Exception as e:
        print(f"Error in request_one: {e}")
        traceback.print_exc()
        return None
    return {
        "idx": idx,
        "prompt": prompt["ori_inst"],
        "choices": completions
    }

messages_lists = []
for ds_split in ds_splits:
    current_messages = []
    for data in ds_split:
        sample_data = data["instruction"]
        current_message = [
            {"role": "user", "content": sample_data},
        ]
        current_messages.append({
            "sys_prompt": sys_prompt,
            "messages": current_message,
            "ori_id": data.get("id", data.get("problem_id", "unknown_id")),
            "ori_inst": sample_data,
        })
    messages_lists.append(current_messages)

for i, msg_list in enumerate(messages_lists):
    print(f"Messages list {i} length: {len(msg_list)}")


def process_messages(messages_list, prompter, results_queue, progress_queue, thread_id):
    try:
        completions = []
        for msg in messages_list:
            result = request_one(msg, prompter)
            if result is not None:
                completions.append(result)
            progress_queue.put(1)
        results_queue.put((thread_id, completions))
    except Exception as e:
        print(f"Error in thread {thread_id}: {str(e)}")
        traceback.print_exc()
        results_queue.put((thread_id, []))


os.makedirs(os.path.dirname(args.fout), exist_ok=True)
fout = open(args.fout, "w")
results_queue = Queue()

total_tasks = sum(len(lst) for lst in messages_lists)
pbar = tqdm(total=total_tasks, desc="Overall Progress")

progress_queue = Queue()

def progress_updater():
    while True:
        try:
            pbar.update(progress_queue.get(timeout=2))
        except:
            if pbar.n >= total_tasks:
                break

threads = []
for i in range(len(messages_lists)):
    if not messages_lists[i]:
        continue
    thread = threading.Thread(
        target=process_messages,
        args=(messages_lists[i], prompters[i], results_queue, progress_queue, i)
    )
    threads.append(thread)
    thread.start()

progress_thread = threading.Thread(target=progress_updater)
progress_thread.daemon = True
progress_thread.start()


all_results = [None] * len(messages_lists)
completed_threads_count = 0
while completed_threads_count < len(threads):
    try:
        thread_id, results = results_queue.get(timeout=5)
        all_results[thread_id] = results
        completed_threads_count += 1
    except:
        alive_threads = sum(1 for t in threads if t.is_alive())
        if alive_threads < len(threads) - completed_threads_count:
            print("A thread may have died unexpectedly. Exiting loop.")
            break
        continue

for thread in threads:
    thread.join()
pbar.close()


print("\nWriting results to file...")
final_results_list = []
for res_list in all_results:
    if res_list:
        final_results_list.extend(res_list)

for result in final_results_list:
    fout.write(json.dumps({
        "id": result["idx"],
        "prompt": result["prompt"],
        "responses": result["choices"]
    }) + "\n")
    fout.flush()

fout.close()

print(f"\nCompleted processing {len(final_results_list)} samples in total")