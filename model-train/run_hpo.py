from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from HPOconfig import HPOConfig
from HPOtrainer import HPOTrainer   
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
parser.add_argument("--output_dir", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct-hpo")
parser.add_argument("--experiment_name", type=str, default="Qwen2.5-Coder-7B-Instruct-hpo")
args = parser.parse_args()

model_name = args.model_name
dataset_name = [
    "../datasets/hpo-pref/apps-pref.jsonl",
    "../datasets/hpo-pref/python-pref.jsonl",
    "../datasets/hpo-pref/c-pref.jsonl",
    "../datasets/hpo-pref/cpp-pref.jsonl",
    "../datasets/hpo-pref/java-pref.jsonl"
]
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

dataset = load_dataset("json", data_files=dataset_name)
train_dataset = dataset["train"]
train_dataset = train_dataset.shuffle(seed=42)

print("-"*100)
print(f"train_dataset length: {len(train_dataset)}")
print("-"*100)

training_args = HPOConfig(
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
    kl_prefix_weight=1.0, 
    kl_suffix_weight=1.0, 
    mid_dpo_weight=1.0,
    rpo_alpha=1.0,
)


from swanlab.integration.transformers import SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="firebird", 
    experiment_name=experiment_name
)

trainer = HPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    callbacks=[swanlab_callback],  
)

trainer.train()