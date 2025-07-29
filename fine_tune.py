import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
# Model and dataset configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "MoazIrfan/Llama-2-7b-chat-finetune"

# LoRA hyperparameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1


# BitsAndBytes configuration

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training configuration

output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 30


# SFTTrainer parameters

max_seq_length = None
packing = False

# Device map

device_map = {"": 0}


# Logging setup

logging.set_verbosity_info()

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# BitsAndBytes configuration for QLoRA
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant
)

if compute_dtype == torch.float16 and use_4bit:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: you can accelerate training with bf16=True")
            print("=" * 80)
        else:
            print("=" * 80)
            print("⚠ Your GPU does NOT support bfloat16.")
            print("=" * 80)
    else:
        print("=" * 80)
        print("ℹ Running on CPU: using float16 / 4-bit quantization may not give speed benefits.")
        print("=" * 80)


#load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LlaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRa configration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

#Set training parameters
training_arguments=TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to = "tensorboard"
)

#Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model= model,
    train_dataset=dataset,
    peft_config = peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer= tokenizer,
    args=training_arguments,
    packing=packing,

)

#Train model
trainer.train()


#Save Trained model
trainer.model.save_pretrained(new_model)


#ignore warnings
logging.set_verbosity(logging.CRITICAL)
#run TEXT generation pipeline with our next model
prompt="What is a Generative Al ?"
pipe =pipeline(task="text-generation", model=model, tokenizer=tokenizer,max_length=100)
result=pipe(f"<s>[INST]{prompt}[/INST]")
print(result[0]["generated_text"])



#Empty Vram
# del model
# del pipe
# del trainer
# import gc
# gc.collect()
# gc.collect()

#Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model=PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

#Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

import locale
locale.getpreferredencoding = lambda: "UTF-8"

# pip install huggingface_hub
# huggingface-cli login
# model.push_to_hub()

# from peft import PeftModel
# trainer.model.push_to_hub("MoazIrfan/Llama-2-7b-chat-finetune")

# from huggingface_hub import create_repo, upload_folder

# repo_id = "MoazIrfan/Llama-2-7b-chat-finetune"

# # Create repo if it doesn't exist
# create_repo(repo_id, private=True)

# Upload local folder
# upload_folder(
#     folder_path=new_model,
#     repo_id=repo_id
# )

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("MoazIrfan/Llama-2-7b-chat-finetune")
# tokenizer = AutoTokenizer.from_pretrained("MoazIrfan/Llama-2-7b-chat-finetune")
