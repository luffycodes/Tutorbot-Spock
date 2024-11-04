import json
import random
import argparse

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import (
    DPOTrainer,
    KTOTrainer,
    DPOConfig,
    KTOConfig,
)
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators to ensure reproducibility.")

    parser.add_argument("--model_path", type=str, required=True, help="Model identifier for the pretrained model.")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and outputs.")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to cache the preprocessed datasets and models.")
    
    parser.add_argument("--loss", type=str, required=True, help="The type of RL algorithm to use.", choices=["dpo", "ipo", "kto", "rso", "cdpo"])
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate.")
    parser.add_argument("--beta", type=float, required=True, help="Beta parameter for DPO training.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.")
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
    accelerate_print("Dataset:\n", dataset)
    return dataset

def prep_kto_data(dataset):
    data = [d for d in dataset]

    kto_data = list()
    for d in data:
        kto_data.extend([
            {"prompt": d["prompt"], "completion": d["chosen"], "label": True},
            {"prompt": d["prompt"], "completion": d["rejected"], "label": False}
        ])
    random.shuffle(kto_data)
    return Dataset.from_list(kto_data)

def accelerate_print(*args, **kwargs):
    accelerator.print(*args, **kwargs)

if __name__ == "__main__":
    accelerator = Accelerator()

    accelerate_print("Parsing the arguments...")    
    args = parse_args()
    
    set_seed(args.seed)

    loss_map = {
        "dpo": "sigmoid",
        "ipo": "ipo",
        "kto": "kto",
    }

    config_map = {
        "dpo": DPOConfig,
        "ipo": DPOConfig,
        "kto": KTOConfig,
    }

    trainer_class_map = {
        "dpo": DPOTrainer,
        "ipo": DPOTrainer,
        "kto": KTOTrainer,
    }
    
    loss_type = loss_map.get(args.loss)
    if not loss_type:
        raise ValueError(f"Unsupported algorithm: {args.loss}")
    
    accelerate_print("Loading the models...")
    
    quantization_config = None
    peft_config = None
    device_map = None
    load_in_4bit = False
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        quantization_config=quantization_config,
        cache_dir=args.cache_dir
    )
    
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        quantization_config=quantization_config,
        cache_dir=args.cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    
    training_args = config_map[args.loss](
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
    
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        beta=args.beta,

        seed=args.seed,
        remove_unused_columns=False,
    )

    if args.loss != "kto":
        training_args.loss_type = loss_type
    
    accelerate_print("Training arguments:\n", training_args.to_dict())

    accelerate_print("Loading the datasets...")
    train_dataset = format_dataset(args.train_dataset_path)

    if args.loss == "kto":
        train_dataset = prep_kto_data(train_dataset)
        
    accelerate_print("# train data points:", len(train_dataset["prompt"]))

    accelerate_print("Start training...")
    
    trainer = trainer_class_map[args.loss](
        model,
        model_ref,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)

    accelerate_print("Done.")
