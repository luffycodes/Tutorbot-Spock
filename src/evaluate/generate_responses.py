import os
import json
import time
import ast
import argparse
from tqdm import tqdm
import pandas as pd
import time

from transformers import (
    set_seed,
)
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description='Process model and dataset paths.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='Path to the test dataset')

    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to cache the preprocessed datasets and models.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generators to ensure reproducibility.")
    parser.add_argument("--batch_size", type=int, required=True, help="Per device batch size for inference.")
    args = parser.parse_args()

    return args

def spock(model, sampling_params, prompts):
    outputs = model.generate(prompts, sampling_params)

    response_jsons = list()
    
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        try:
            response_json = ast.literal_eval(response.strip())
        except Exception as err:
            response_json = response
        
        response_jsons.append(response_json)
        
    return response_jsons

def save(file_name, all_prompts, all_responses, all_chosens):
    data = {
        "prompt": all_prompts,
        "response": all_responses,
        "gpt": all_chosens
    }
    df = pd.DataFrame(data)

    df.to_csv(file_name, index=False)

    print(f'Data written to {file_name}.')


if __name__ == "__main__":
    print("Parsing the arguments...")
    args = parse_args()
    set_seed(args.seed)

    print("Loading the model and tokenizer...")

    model = LLM(model=args.model_path, tokenizer=args.model_path, dtype="bfloat16", tensor_parallel_size=4, seed=args.seed, download_dir=args.cache_dir) # 4 gpus with 128 batch size
    
    sampling_params = SamplingParams(temperature=0.3, seed=args.seed, min_tokens=2, max_tokens=512, skip_special_tokens=True)

    print("Loading the evaluation dataset...")
    
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
    
    print("Start generating responses...")
    os.makedirs(args.output_dir, exist_ok=True)

    all_responses = list()
    all_chosens = list()
    all_prompts = list()
    
    start_time = time.time()
    for idx in tqdm(range(0, len(test_dataset), args.batch_size)):

        subset = test_dataset[idx:idx + args.batch_size]
        prompts = [d["prompt"] for d in subset]
        chosens = [d["chosen"] for d in subset]

        response_jsons = spock(model, sampling_params, prompts)
        
        all_responses.extend(response_jsons)
        all_prompts.extend(prompts)
        all_chosens.extend(chosens)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time:", elapsed_time)
    
    # Save to CSV
    print("Saving...")
    
    output_file = args.output_dir + f"/responses.csv"
    
    save(output_file, all_prompts, all_responses, all_chosens)
    print("Done.")
