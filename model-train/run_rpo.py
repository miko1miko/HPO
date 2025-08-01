from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
parser.add_argument("--output_dir", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct-dpo")
parser.add_argument("--experiment_name", type=str, default="Qwen2.5-Coder-7B-Instruct-dpo")
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

import re
def delete_diff(sample):
    chosen_code = sample["chosen"]
    rejected_code = sample["rejected"]
    diff_pattern = r'<diff>(.*?)</diff>'
    
    match_chosen = re.search(diff_pattern, chosen_code, re.DOTALL)
    if match_chosen:  
        start_tag_pos = match_chosen.start()
        end_tag_pos = match_chosen.end()
        diff_content = match_chosen.group(1)
        cleaned_chosen = chosen_code[:start_tag_pos] + (diff_content if diff_content is not None else "") + chosen_code[end_tag_pos:]
    else:
        cleaned_chosen = chosen_code
    
    match_rejected = re.search(diff_pattern, rejected_code, re.DOTALL)
    if match_rejected:  
        start_tag_pos = match_rejected.start()
        end_tag_pos = match_rejected.end()
        diff_content = match_rejected.group(1)
        cleaned_rejected = rejected_code[:start_tag_pos] + (diff_content if diff_content is not None else "") + rejected_code[end_tag_pos:]
    else:
        cleaned_rejected = rejected_code

    return {
        "prompt": sample["prompt"].strip(),
        "chosen": cleaned_chosen,
        "rejected": cleaned_rejected
    }

model.gradient_checkpointing_disable()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"  

dataset = load_dataset("json", data_files=dataset_name)
train_dataset = dataset["train"]


train_dataset = train_dataset.map(delete_diff)
train_dataset = train_dataset.shuffle(seed=42)

print("-"*100)
print(f"train_dataset length: {len(train_dataset)}")
print("-"*100)

print(train_dataset[0])


training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_grad_norm=0.5,         
    optim="adamw_torch",
    fp16=False,              
    bf16=True,
    remove_unused_columns=False,
    max_prompt_length=2048,
    max_length=4096,
    beta=0.4,
    save_strategy="epoch",
    report_to="none",
    logging_steps=1,
    learning_rate=1e-5,
    rpo_alpha=1.0, 
)

import swanlab
from swanlab.integration.transformers import SwanLabCallback

swanlab_callback = SwanLabCallback(
    project="firebird", 
    experiment_name=experiment_name
)




trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[swanlab_callback], 
)


trainer.train()