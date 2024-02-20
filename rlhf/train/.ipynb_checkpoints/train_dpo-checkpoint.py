from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Union
import random
from datasets import Dataset, load_dataset
import json
import argparse
from trl import DPOTrainer
import os
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig
from typing import Dict, Optional
import ast 
import pandas as pd

SEED = 42
set_seed(SEED)

print("Parsing the arguments...")

parser = argparse.ArgumentParser()

####################################################
# Change these
# parser.add_argument("--train_data", type=str, default="/data/kn22/dpo_data/bio/dpo_train/mistral-instruct_dpo_bio_uniform_batch_123_filtered_dpo.json")
# parser.add_argument("--output_dir", type=str, default="/data/kn22/spock_bio_dpo/spock_bio_mistral-instruct_dpo_r12")
# parser.add_argument("--model", type=str, default="/data/kn22/spock_bio/spock_bio_mistral-instruct_r3/")
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--beta", type=float, required=True)
####################################################

parser.add_argument("--use_lora", action='store_true', default=False, help="whether to train with lora")
parser.add_argument("--ipo", action='store_true', default=False, help="whether to use ipo")
parser.add_argument("--kto", action='store_true', default=False, help="whether to use kto")
parser.add_argument("--rso", action='store_true', default=False, help="whether to use rso")
parser.add_argument("--cdpo", action='store_true', default=False, help="whether to use cdpo")
parser.add_argument("--dpo", action='store_true', default=False, help="whether to use dpo")

parser.add_argument("--cache_dir", type=str, default="/data/kn22/cache")
parser.add_argument("--test_data", type=str, default="")
parser.add_argument("--do_eval", action='store_true', default=False, help="whether to do eval")

args = parser.parse_args()
    
def format_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
        
    df = pd.DataFrame(dataset)
    dataset = Dataset.from_pandas(df)

    print(dataset)
    return dataset

        # final_dataset = list()

        # for i in range(len(dataset["prompt"])):
        #     prompt = dataset["prompt"][i] # [3735:]

        #     # inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, add_special_tokens=False, truncation=False)
        #     # if (inputs.input_ids.shape[1] >= 3300):
        #     #     continue
            
        #     # if not validate_json(dataset["chosen"][i]):
        #     #     continue
            
        #     chosen = dataset["chosen"][i]
        #     rejected = dataset["rejected"][i]
        #     final_dataset.append({
        #         "prompt":prompt,
        #         "chosen":chosen,
        #         "rejected":rejected
        #     })
            
        # random.shuffle(final_dataset)
        # return final_dataset
    
print("Loading the models...")

if args.use_lora == True:
    print("Applying LoRA...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        nb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    device_map = {'':torch.cuda.current_device()}

    load_in_4bit = True
else:
    quantization_config = None
    peft_config = None
    device_map = None
    load_in_4bit = False

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=device_map,
    load_in_4bit=load_in_4bit,
    quantization_config=quantization_config,
    cache_dir=args.cache_dir
)

model_ref = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=device_map,
    load_in_4bit=load_in_4bit,
    quantization_config=quantization_config,
    cache_dir=args.cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.unk_token

print("Loading the datasets...")

train_dataset = format_dataset(args.train_data)
print("# train data points:", len(train_dataset["prompt"]))

if args.do_eval:
    test_dataset = format_dataset(args.test_data)
    print("# test data points:", len(test_dataset["prompt"]))

else:
    test_dataset = None

training_args = TrainingArguments(
    num_train_epochs=3,
    # max_steps=120,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-5 if args.use_lora else 1e-7,

    evaluation_strategy="no",

    save_strategy="steps",
    save_steps=153,
    save_total_limit=5,
        
    logging_steps=1,
    output_dir=args.output_dir,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.05,
    remove_unused_columns=False,
    bf16=True,
    optim="paged_adamw_32bit",
    seed=SEED,
)

print("Training arguments:\n", training_args.to_dict())

print("Start DPO training...")

if args.ipo:
    loss_type = "ipo"
elif args.kto:
    loss_type = "kto_pair"
elif args.rso:
    loss_type = "hinge"
elif args.cdpo:
    loss_type = "cdpo"
elif args.dpo:
    loss_type = "sigmoid"
else:
    exit("Unsupported algorithm!")
    
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    peft_config=peft_config,
    beta=args.beta,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=4096,
    max_prompt_length=3600,

    loss_type=loss_type,
)

print("loss type:", dpo_trainer.loss_type)
print("beta:", dpo_trainer.beta)
dpo_trainer.train()
dpo_trainer.save_model(args.output_dir)