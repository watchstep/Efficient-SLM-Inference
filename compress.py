import os
import time

import numpy as np
import torch
import json
import re
import argparse

from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, 
                          Phi3ForCausalLM,
                          Phi3Config,
                          pipeline,
                          )
from transformers.pipelines.pt_utils import KeyDataset
from llmlingua import PromptCompressor

import utils

parser = argparse.ArgumentParser(description="compress any prompt.")
parser.add_argument("--model_path", type=str, default="models/")
parser.add_argument("--model_name", type=str, default="Phi-3-medium-4k-instruct")
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--dataset", type=str, default="test_dataset.jsonl")
parser.add_argument("--output", type=str, default="compressed.jsonl")
parser.add_argument("--compressor", type=str, default="roberta")
parser.add_argument(
    "--compression_rate", help="compression rate", type=float, default=0.7
)
parser.add_argument(
    "--target_token", help="number of target tokens", type=int, default=-1
)
parser.add_argument(
    "--force_tokens",
    help="the tokens which will be forcely preserved, empty space separated",
    type=str,
    default=None,
)
parser.add_argument("--keep_last_sentence", type=int, default=0)
parser.add_argument("--use_llmlingua2", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use_context_level_filter", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--seed',type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)


config = parser.parse_args([])
utils.seed_everything(config.seed)
CURR_PATH = os.getcwd()

torch.cuda.empty_cache()
#######  Set up  #######
model_id = os.path.join(CURR_PATH, config.model_path, config.model_name) # please replace with local model path

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # Running flash attention
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    batch_size=config.batch_size,
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

#######  GPU Warm up  #######
messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]
output = pipe(messages, **generation_args)
# print(output[0]['generated_text'])

#######  Prompt Compression  #######
COMPRESSOR_MAP = {
   "gpt2" : "lgaalves/gpt2-dolly",
   "phi2": "microsoft/phi-2",
   "llama2": "TheBloke/Llama-2-7b-Chat-GPTQ",
   "roberta" : "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
}
compressor_id = COMPRESSOR_MAP[config.compressor]

llm_lingua = PromptCompressor(compressor_id, device_map="auto", use_llmlingua2=config.use_llmlingua2)

if config.force_tokens is not None:
    config.force_tokens = [str(item).replace("\\n", "\n") for item in config.force_tokens.split()]
else:
    config.force_tokens = []

def compress(content):
    data = content.split('\n\n')
    instruction = '\n' + data[0]
    ddata = data[1].split('\n')
    question = ddata[0]
    choice = ddata[1]
  
    comp_dict = llm_lingua.compress_prompt(
        context='\n'.join([instruction, question]),
        keep_last_sentence=config.keep_last_sentence,
        rate=config.compression_rate  if len(question) < 150 else 0.5,
        target_token=-1 if len(question) < 150 else config.target_token,
        force_tokens=config.force_tokens,
        use_context_level_filter= config.use_context_level_filter,
    )

    comp = re.sub(r'\n\s+', '\n', comp_dict['compressed_prompt']).replace('question:', '\nquestion:')
    return f"{comp}\n{choice}\n"
    
def compressed_jsonl(input_file_path, output_file_path):
    with open(input_file_path, 'r') as reader:
        with open(output_file_path, 'w') as writer:
            for line in reader:
                line = json.loads(line)
                line["message"][0]["content"] = compress(line["message"][0]["content"])
                
                json.dump(line, writer)
                writer.write("\n")

input_file_path = os.path.join(CURR_PATH, config.data_path, config.dataset)
output_file_path = os.path.join(CURR_PATH, config.data_path, config.output)
compressed_jsonl(input_file_path, output_file_path)

#######  Load data and Inference -> Performance evaluation part  #######
start = time.time()
data = load_dataset("json", data_files=output_file_path)['train']
outs = pipe(KeyDataset(data, 'message'), **generation_args)
end = time.time()

#######  Accuracy (Just for leasderboard)  #######
print("===== Answers =====")
correct = 0
for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n","")
    if answer == correct_answer:
        correct += 1
    # print(answer)
 
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")