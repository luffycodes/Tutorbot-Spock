"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Optional
import sys
import warnings

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.model.chatglm_model import chatglm_generate_stream
import torch
import torch.nn.functional as F
import faiss
import pandas as pd
import re


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                              min=1e-9)


def count_user_occurrences(string):
    matches = re.findall(r'STUDENT:', string)
    return len(matches)


def find_user_occurrence(string):
    match = re.search(r'STUDENT:', string)
    if match:
        return match.start()
    else:
        return -1


def get_relevant_para(prompt):
    loc = find_user_occurrence(prompt)
    if loc == -1:
        return prompt

    dataframe = pd.read_csv('openstax_biology_2e.csv')
    dataframe = dataframe[dataframe['p_id'].str.startswith('fs-').fillna(False)]
    paragraphs = dataframe['p_content'].tolist()
    index = faiss.read_index('paragraph_index.faiss')

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Load the index from the file

    # Tokenize and compute embedding for the query
    encoded_query = tokenizer(prompt[loc+4:], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_embedding = mean_pooling(query_output, encoded_query['attention_mask'])
    normalized_query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # Perform a search using Faiss
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(normalized_query_embedding.squeeze().unsqueeze(dim=0).numpy(), k)

    # Get the relevant paragraphs
    relevant_paragraphs = [paragraphs[i] for i in indices[0]]
    info = "\nHelpful Information for Tutorbot: "
    for para in relevant_paragraphs:
        info = info + "\n" + str(para)
    info = info + "\n End of Helpful Information for Tutorbot.\n"

    info = info + "\n" + '''
Now, let's begin. Your goal as a Tutorbot is to help the student with a question.

Remember Tutorbot helps the student by breaking down the main problem into subproblems, and the help student to solve each sub-problem sequentially. Tutorbot only provide hints.
Use the following json format for your reply:

Put all the output in the following JSON structure
{{
   "Thoughts of Tutorbot": ".."
   "Evaluation of Student Response": "a,b,c,d,e,f,g"
   "Action Based on Evaluation": "1,2,3,4,5,6,7,8,9,10,11,12"
   "Subproblem State": "w,x,y,z"
   "Subproblem": ".."
   "Tutorbot": "..",
}}

Also, make sure that all your responses/ statements to the student are factually correct and TRUE.

Now the conversation is starting. Help the student with the question. '''
    
    prompt = prompt[:loc] + info + prompt[loc:]
    return prompt


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    prompt = get_relevant_para(prompt)
    num_matches = count_user_occurrences(prompt)
    if num_matches == 1:
        prompt = prompt[:-10] + " Instructions to Tutorbot: Please break down the problem into subproblems for the student, and help the student to solve each sub-problem sequentially." + prompt[-10:] 
    else:
        prompt = prompt[:-10] + " Instructions to Tutorbot: Please make sure that all your responses/ statements to the student are factually correct. Do make lie or provide false facts to the student. " + prompt[-10:] 

    print("New prompt:", prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                pos = output.rfind(stop_str, rfind_start)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    if conv_template:
        conv = get_conv_template(conv_template)
    else:
        conv = get_conversation_template(model_path)

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = chatglm_generate_stream
            prompt = conv.messages[conv.offset :]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        outputs = chatio.stream_output(output_stream)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
