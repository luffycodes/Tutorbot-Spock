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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators to ensure reproducibility.")

    parser.add_argument("--model_path", type=str, required=True, help="Model identifier for the pretrained model.")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and outputs.")
    parser.add_argument("--cache_dir", type=str, default="/data/kn22/cache", help="Directory to cache the preprocessed datasets and models.")
    
    parser.add_argument("--loss", type=str, required=True, help="The type of loss function to use.")
    parser.add_argument("--use_lora", action='store_true', default=False, help="Whether to train with LoRA.")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate.")
    parser.add_argument("--beta", type=float, required=True, help="Beta parameter for DPO training.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=False, help="Use gradient checkpointing to save memory.")

    parser.add_argument("--evaluation_strategy", type=str, default="no", choices=["no", "steps", "epoch"], help="Evaluation strategy to use.")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy to use.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log metrics every X updates steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="Type of learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay if we apply some.")
    parser.add_argument("--bf16", action='store_true', help="Use bf16 precision instead of the default floating point precision.")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", choices=["adamw", "paged_adamw_32bit"], help="Optimizer to use.")

    parser.add_argument("--max_length", type=int, default=4096, help="The maximum length of the model input.")
    parser.add_argument("--max_prompt_length", type=int, default=3600, help="The maximum length of the prompts.")

    args = parser.parse_args()

    return args
    
def format_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
        
    df = pd.DataFrame(dataset)
    dataset = Dataset.from_pandas(df)
    print(dataset)
    return dataset

if __name__ == "__main__":
    print("Parsing the arguments...")    
    args = parse_args()
    
    set_seed(args.seed)
    
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
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        quantization_config=quantization_config,
        cache_dir=args.cache_dir
    )
    
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        quantization_config=quantization_config,
        cache_dir=args.cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.unk_token
    
    print("Loading the datasets...")
    
    train_dataset = format_dataset(args.train_dataset_path)
    print("# train data points:", len(train_dataset["prompt"]))
    
    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    
        bf16=args.bf16,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing,
        
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
    
        seed=args.seed,
        remove_unused_columns=False,
    )
    
    print("Training arguments:\n", training_args.to_dict())
    
    print("Start DPO training...")
    
    loss_map = {
        "dpo": "sigmoid",
        "ipo": "ipo",
        "kto": "kto_pair",
        "rso": "hinge",
        "cdpo": "cdpo"
    }
    
    loss_type = loss_map.get(args.loss)
    if not loss_type:
        raise ValueError("Unsupported algorithm!")
        
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        peft_config=peft_config,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    
        loss_type=loss_type,
    )
    
    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)