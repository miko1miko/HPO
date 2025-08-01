from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch
import os
import json
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
parser.add_argument("--output_dir", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct-sft")
parser.add_argument("--experiment_name", type=str, default="Qwen2.5-Coder-7B-Instruct-sft")
args = parser.parse_args()


dataset_name = [
    "../datasets/hpo-pref/apps-pref.jsonl",
    "../datasets/hpo-pref/python-pref.jsonl",
    "../datasets/hpo-pref/c-pref.jsonl",
    "../datasets/hpo-pref/cpp-pref.jsonl",
    "../datasets/hpo-pref/java-pref.jsonl"
]
model_name = args.model_name
output_dir = args.output_dir
experiment_name=args.experiment_name


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  
    device_map="auto",
    use_cache=False,
    low_cpu_mem_usage=True,
)

model.gradient_checkpointing_disable()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"  

def convert_dpo_to_sft(sample):
    chosen_code = sample["chosen"]
    diff_pattern = r'<diff>(.*?)</diff>'
    match = re.search(diff_pattern, chosen_code, re.DOTALL)

    if match:
        start_tag_pos = match.start()
        end_tag_pos = match.end()
        diff_content = match.group(1)
        cleaned_text = chosen_code[:start_tag_pos] + (diff_content if diff_content is not None else "") + chosen_code[end_tag_pos:]
    else:
        cleaned_text = chosen_code

    return {
        "prompt": sample["prompt"].strip(),
        "completion": cleaned_text
    }

train_dataset = load_dataset('json', data_files=dataset_name, split='train')


train_dataset = train_dataset.map(convert_dpo_to_sft)

print(f"train_dataset: {len(train_dataset)}")

train_dataset = train_dataset.shuffle(seed=42)

training_args = SFTConfig(output_dir=output_dir,
                            per_device_train_batch_size=1,
                            per_device_eval_batch_size=1,
                            gradient_accumulation_steps=1,
                            num_train_epochs=1,           
                            optim="adamw_torch",
                            fp16=False,                    
                            bf16=True,
                            remove_unused_columns=True,     
                            logging_steps=1,               
                            save_strategy="epoch",
                            max_length=4096,
                            report_to="none", 
                          )

import swanlab
from swanlab.integration.transformers import SwanLabCallback
swanlab_callback = SwanLabCallback(
    project = "firebird", 
    experiment_name = experiment_name
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=training_args,
    callbacks=[swanlab_callback],
)

trainer.train()