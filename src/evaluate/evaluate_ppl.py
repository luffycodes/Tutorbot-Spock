import json
import ast
from tqdm import tqdm
import argparse
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")

    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file")
    parser.add_argument('--cache_dir', type=str, default="cache", help="Path to the model file")
    parser.add_argument('--ppl_dataset_path', type=str, default="datasets/bio-dataset-ppl.json", help="Path to the alternative dataset")

    args = parser.parse_args()
    return args

def load_model(model_path, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    
    return model, tokenizer

def is_valid(action, action_alter):
    if action == "1":
        return action_alter == "2"

    if action == "4":
        return action_alter == "5"

    return False

def format_prompt(d):
    def build_convo_history(convo):
        roles = {
            "human": "Student: ",
            "gpt": "Tutorbot: "
        }
        
        history = ""
        for turn in convo:
            role = ""
            if history != "":
                history += "\n\n"
                role += roles[turn["from"]]

            history += role
            
            if turn["from"] == "gpt":
                history += turn["value"]   
            else:
                content = turn["value"]
                if history == "":
                    content = content.replace('''Student: "\nQ''', '''Student: Q''')
                history += content
                
        return history + f'''\n\nTutorbot:'''

    history = build_convo_history(d["conversations"][:-2]) 

    action = ast.literal_eval(d["conversations"][-2]["value"])["Action Based on Evaluation"]
    action_alter = ast.literal_eval(d["conversations"][-1]["value"])["Action Based on Evaluation"]
    
    valid = is_valid(action, action_alter)
    
    formatted_d = {
        "id": d["id"],
        "prompt": history,
        "response_origin": d["conversations"][-2]["value"],
        "response_alter": d["conversations"][-1]["value"],
        "is_action1": action == "1",
        "is_valid": valid
    }

    return formatted_d
    
def compute_nll(model, input_ids, target_ids):
    
    non_masked_indices = target_ids != -100
    num = non_masked_indices.sum().item() - 1
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    nll = neg_log_likelihood.item() * num
    return nll, num

def collate_fn(samples):
    keys = ["id", 'prompt', 'input_ids_origin', 'target_ids_origin', 'input_ids_alter', 'target_ids_alter', "len_origin", "len_alter"]
    
    batch = {key: [item[key] for item in samples] for key in keys}

    for key in keys:
        if key in ["id", "prompt"]:
            continue
        batch[key] = torch.vstack(batch[key])

    return batch

def tokenize_function(examples):
    tokenized_prompt = tokenizer(examples["prompt"], truncation=False)
    prompt_len = len(tokenized_prompt.input_ids)

    tokenized_response_origin = tokenizer(examples["prompt"] + " " + examples["response_origin"], return_tensors='pt', truncation=False)
    
    target_ids_origin = tokenized_response_origin.input_ids.clone()
    target_ids_origin[:, :prompt_len] = -100
    
    tokenized_response_alter = tokenizer(examples["prompt"] + " " + examples["response_alter"], return_tensors='pt', truncation=False)
    
    target_ids_alter = tokenized_response_alter.input_ids.clone()
    target_ids_alter[:, :prompt_len] = -100
    
    return {
        "prompt": examples["prompt"],
        "input_ids_origin": tokenized_response_origin.input_ids,
        "target_ids_origin": target_ids_origin,
        "input_ids_alter": tokenized_response_alter.input_ids,
        "target_ids_alter": target_ids_alter,
        "is_action1": torch.tensor(examples["is_action1"], dtype=torch.bool)
    }


if __name__ == "__main__":
    args = parse_args()

    with open(args.ppl_dataset_path, 'r') as file:
        alter_data = json.load(file)

    formatted_alter_data = [format_prompt(d) for d in alter_data]
    formatted_alter_data = [d for d in formatted_alter_data if d["is_valid"]]
    print("# data points:", len(formatted_alter_data))

    alter_dataset = datasets.Dataset.from_list(formatted_alter_data)

    model, tokenizer = load_model(args.model_path, args.cache_dir)
    tokenized_alter_dataset = alter_dataset.map(tokenize_function, batched=False).with_format("torch")

    total_nlls = {
        "action1": 0.0,
        "action2": 0.0,
        "action4": 0.0,
        "action5": 0.0,
    }

    total_nums = {
        "action1": 1e-8,
        "action2": 1e-8,
        "action4": 1e-8,
        "action5": 1e-8,
    }

    for i, sample in tqdm(enumerate(tokenized_alter_dataset)):

        is_action1 = sample["is_action1"].item() == True

        nlls_origin, nums_origin = compute_nll(model, sample["input_ids_origin"], sample["target_ids_origin"])
        nlls_alter, nums_alter = compute_nll(model, sample["input_ids_alter"], sample["target_ids_alter"])

        if is_action1:
            action = "action1"
            action_alter = "action2"
        else:
            action = "action4"
            action_alter = "action5"
        
        total_nlls[action] += nlls_origin
        total_nlls[action_alter] += nlls_alter

        total_nums[action] += nums_origin
        total_nums[action_alter] += nums_alter

    actions = ["action1", "action2", "action4", "action5"]

    nll_14_sum = sum([total_nlls[action] for action in ["action1", "action4"]])
    num_14_sum = sum([total_nums[action] for action in ["action1", "action4"]])

    nll_25_sum = sum([total_nlls[action] for action in ["action2", "action5"]])
    num_25_sum = sum([total_nums[action] for action in ["action2", "action5"]])

    results = {
        **{action + "_ppl": np.exp(total_nlls[action] / total_nums[action]) for action in actions}, 
        **{"action1+4": np.exp(nll_14_sum / num_14_sum), "action2+5": np.exp(nll_25_sum / num_25_sum)}
    }

    for key, value in results.items():
        print(f"{key}: {round(value, 2)}")
