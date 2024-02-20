from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from datasets import Dataset, load_dataset
import json
import argparse
from trl import DPOTrainer
import os
import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig, PeftModel
from typing import Dict, Optional
import csv
import time
import ast
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process model and dataset paths.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='Path to the test dataset')

    parser.add_argument("--cache_dir", type=str, default="/data/kn22/cache", help="Directory to cache the preprocessed datasets and models.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators to ensure reproducibility.")
    parser.add_argument("--use_lora", action='store_true', default=False, help="Whether to merge the base model and the LoRA adapter.")
    args = parser.parse_args()

    return args

def spock(model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, add_special_tokens=False, truncation=False).to(model.device)
        
        tokens = model.generate(
            **inputs,
            min_new_tokens=2,
            max_new_tokens=1024,
            temperature=float(0.3),
            
            do_sample=True,
            repetition_penalty=float(1.2)
        )[0]

        tokens = tokens[inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        
        response_json = ast.literal_eval(response.strip())
    except Exception as err:
        print("Error:", err)
        response_json = response
        
    return response_json

def store(file_name, prompt, response, chosen):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([prompt, response, chosen])  # Write data

    print(f'Data written to {file_name}')

if __name__ == "__main__":
    print("Parsing the arguments...")
    args = parse_args()
    set_seed(args.seed)
    
    print(f"Loading the evaluation dataset...")
    
    with open(args.test_dataset_path, 'r') as file:
        dataset = json.load(file)
        
        test_dataset = list()
        
        for i in range(len(dataset["prompt"])):
            prompt = dataset["prompt"][i]
            
            chosen = dataset["chosen"][i]
            test_dataset.append({
                "prompt":prompt,
                "chosen":chosen,
            })
            
        print("# data points:", len(test_dataset))
        
    print(f"Loading the model...")
    
    if not args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    else:
        peft_config = PeftConfig.from_json_file(os.path.join(args.model_path, 'adapter_config.json'))
        model = AutoModelForCausalLM.from_pretrained(
            peft_config['base_model_name_or_path'],
            return_dict=True,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    
        tokenizer = AutoTokenizer.from_pretrained(peft_config['base_model_name_or_path'])
        
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    output_file = args.output_dir + "/responses.csv"
    with open(output_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "response", "gpt"])
            
    responses = []
        
    print("Start generating responses...")
    
    for i, test in enumerate(test_dataset):
        print(f"Testing prompt {i}...")
        prompt = test["prompt"]
        response_json = spock(model, tokenizer, prompt)
    
        responses.append(response_json)
    
        store(output_file, prompt, str(response_json), test["chosen"])
    print("Done generating.")