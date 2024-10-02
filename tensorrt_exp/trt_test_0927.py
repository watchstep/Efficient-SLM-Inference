import os
import time

import numpy as np
import torch

import argparse

import torch.onnx

from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer, 
                          Phi3ForCausalLM,
                          Phi3Config,
                          pipeline,
                          )
from transformers.pipelines.pt_utils import KeyDataset
from huggingface_hub import hf_hub_download
import torch_tensorrt 
import utils
import torch.backends.cudnn as cudnn
device = torch.device( "cuda" if torch.cuda.is_available() else cpu )

parser = argparse.ArgumentParser(description="TensorRT")
parser.add_argument("--model_path", type=str, default="models/")
parser.add_argument("--model_name", type=str, default="Phi-3-medium-4k-instruct")
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--dataset", type=str, default="test_dataset.jsonl")
parser.add_argument('--seed',type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument('--dtype',type=str, default="auto")
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.0)

config = parser.parse_args()

utils.seed_everything(config.seed)
CURR_PATH = os.getcwd()

# 1. 모델 및 토크나이저 로드
model_id = os.path.join(CURR_PATH, config.model_path, config.model_name)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

max_memory = {
    0: "10GB",  # GPU 0 
    'cpu': "20GB"  # CPU 
}

llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    max_memory = max_memory 
)

trt_model = torch.compile(llm, backend="torch_tensorrt", dynamic=False)


torch.cuda.empty_cache() #cache를 지움
generation_args = {
    "max_new_tokens": 1024,  
    "temperature": 0.0,
    # "top_p": 0.95,
    "do_sample": False,
}

# torch.cuda.synchronize()

# 추론 시작 시간 측정
start = time.perf_counter()

# 데이터 로드
data = load_dataset("json", data_files="./data/test_dataset.jsonl")['train']
messages = KeyDataset(data, 'message')

# 시스템 프롬프트 정의
sys = "You are a helpful AI Assistant. Do not hallucinate."

# 메시지 준비
messages = list(messages)
token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
token_ids.insert(0, "<|system|>" + sys + "<|end|>")

# 입력 텐서로 변환 및 데이터 타입 변경
inputs = tokenizer(token_ids, return_tensors="pt", padding=True).to("cuda")
inputs = {k: v.type(torch.int32).to("cuda") for k, v in inputs.items()}

# 추론 실행
with torch.no_grad():
    outs = trt_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        **generation_args
    )
# 생성된 토큰을 텍스트로 디코딩
generated_texts = tokenizer.batch_decode(outs, skip_special_tokens=True)
processed_outs = []
for idx, text in enumerate(generated_texts):
    if idx == 0:
        continue  # 첫 번째 출력은 건너뜀
    # 마지막 \n 이후의 부분만 추출
    last_newline_pos = text.rfind('\n')
    if last_newline_pos != -1:
        extracted_text = text[last_newline_pos + 1:].strip()
    else:
        extracted_text = text.strip()
    # 결과를 딕셔너리로 감싸고 리스트에 추가
    processed_outs.append([{"generated_text": extracted_text}])

# torch.cuda.synchronize()

# 추론 종료 시간 측정
end = time.perf_counter()

print("===== Answers =====")
correct = 0
for i, out in enumerate(processed_outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n","")
    if answer == correct_answer:
        correct += 1
    # print(answer)
 
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")
