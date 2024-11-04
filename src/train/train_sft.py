import json

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    Trainer
)
import argparse
from accelerate import Accelerator

IGNORE_INDEX = -100

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators to ensure reproducibility.")

    parser.add_argument("--model_path", type=str, required=True, help="Model identifier for the pretrained model.")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="Path to the evaluation data.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model and outputs.")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to cache the preprocessed datasets and models.")
 
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=False, help="Use gradient checkpointing to save memory.")

    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation.")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy to use.")
    parser.add_argument("--eval_accumulation_steps", type=int, default=20, help="Evaluation steps to use.")

    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy to use.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log metrics every X updates steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="Type of learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Linear warmup over warmup_ratio fraction of total steps.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay if we apply some.")
    parser.add_argument("--bf16", action='store_true', help="Use bf16 precision instead of the default floating point precision.")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", choices=["adamw", "paged_adamw_32bit"], help="Optimizer to use.")

    parser.add_argument("--max_seq_length", type=int, default=4096, help="The maximum length of the sequence.")
    args = parser.parse_args()

    return args

def format_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    formatted_data = list()

    roles = {
        "human": "Student",
        "gpt": "Tutorbot",
    }

    for convo in dataset:
        messages = list()

        for i, convo_round in enumerate(convo["conversations"]):
            if i == 0:
                messages.extend([
                    {"role": "System", "content": convo_round["value"]}
                ])
            else:
                role = roles[convo_round["from"]]
                content = convo_round["value"]
    
                messages.append(
                    {"role": role, "content": content}
                )
        formatted_data.append({"messages": messages})

    dataset = Dataset.from_list(formatted_data)
    
    accelerate_print("Dataset:\n", dataset)
        
    return dataset

def extend_tokenizer_and_embedding(
    special_tokens_dict,
    tokenizer,
    model,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def accelerate_print(*args, **kwargs):
    accelerator.print(*args, **kwargs)

def get_assistant_start_end_indices(messages, conversation_text):
    start_indices = []
    current_index = 0
    for message in messages:
        message_text = message["content"]
        match_index = conversation_text[current_index:].find(message_text)
        start_indices.append(current_index + match_index)
        current_index += match_index + len(message_text)
    end_indices = [len(conversation_text) if i == len(start_indices) - 1 else start_indices[i+1] for i, x in enumerate(start_indices)]
    roles = [message["role"] for message in messages]
    return [(s, e) for s, e, r in zip(start_indices, end_indices, roles) if r == "Tutorbot"]

def get_masked_labels(conversation_ids, assistant_ranges):
    for id_, (id_s, id_e) in list(zip(conversation_ids["input_ids"], conversation_ids["offset_mapping"])):
        if any(id_s >= s and id_e <= e for s, e in assistant_ranges):
            yield id_
        else:
            yield IGNORE_INDEX

def combine_messages(messages):
    text = ""

    for m in messages:
        role =  m["role"]
        content = m["content"].strip()

        if role == "Tutorbot":
            content += tokenizer.eos_token

        text += "\n\n" + role + ": " + content
    
    return text
    
def tokenize_messages(messages):
    conversation_text = combine_messages(messages)

    conversation_ids = tokenizer(conversation_text, return_offsets_mapping=True, padding="max_length", max_length=args.max_seq_length)

    assistant_ranges = get_assistant_start_end_indices(messages, conversation_text)
    labels = get_masked_labels(conversation_ids, assistant_ranges)
    conversation_ids["labels"] = list(labels)
    del conversation_ids["offset_mapping"]

    return conversation_ids
        
if __name__ == "__main__":
    accelerator = Accelerator()

    accelerate_print("Parsing the arguments...")    
    args = parse_args()
    
    set_seed(args.seed)
    
    accelerate_print("Loading the models...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_4bit=False,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        accelerate_print(f"Adding pad token as '<pad>'")
        extend_tokenizer_and_embedding(
            special_tokens_dict=dict(pad_token="<pad>"),
            tokenizer=tokenizer,
            model=model,
        )

    accelerate_print("Loading the datasets...")

    train_dataset = format_dataset(args.train_dataset_path)
    eval_dataset = format_dataset(args.eval_dataset_path)
    
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
    
    accelerate_print("Training arguments:\n", training_args.to_dict())

    accelerate_print("Start training...")

    def collator(example):
        tokenized_example = [tokenize_messages(sample["messages"]) for sample in example]

        labels = torch.vstack([torch.tensor(sample["labels"]) for sample in tokenized_example])
        input_ids = torch.vstack([torch.tensor(sample["input_ids"]) for sample in tokenized_example])
        attention_masks = torch.vstack([torch.tensor(sample["attention_mask"]) for sample in tokenized_example])
        
        return {
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }
        
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)

    accelerate_print("Done.")
