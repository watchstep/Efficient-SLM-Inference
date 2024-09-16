import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from vllm import LLM, SamplingParams

import re

# my local models
MODELZOO = {
    "phi3-14b" : "./models/Phi-3-medium-4k-instruct",
    "phi3-3.8b" : "./models/Phi-3-mini-4k-instruct",
    "bloom-560m": "./models/bloom-560m",
}

llm = LLM(
    model=MODELZOO["phi3-14b"],
    tensor_parallel_size=1,
    speculative_model=MODELZOO["phi3-3.8b"],
    num_speculative_tokens=5,
    use_v2_block_manager=True,
    max_model_len=100,  # Decrease this value to match the cache limit
)

data = load_dataset("json", data_files="./data/test_dataset.jsonl")['train']

def data_preprocessing(data):
    new_set = []
    for i in range(len(data)):
        temp = data[i]['message'][0]['content']
        new_set.append(temp)
    return new_set
data_list = data_preprocessing(data)

sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

outputs = llm.generate(data_list, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

####### Section 3. Load data and Inference -> Performance evaluation part #######
start = time.perf_counter()
outs = llm.generate(data_list, sampling_params)
end = time.perf_counter()

####### Section 4. Accuracy (Just for leasderboard) #######
print("===== Answers =====")
correct = 0
for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    # 생성된 답변에서 불필요한 텍스트 제거
    answer = out.outputs[0].text
    cleaned_answer = re.sub(r"A:\n\n\n### response ###\n\n|\n### response ###\n\n|A: |\nB:", "", answer).lstrip().replace("\n","")
    
    # 정답과 출력된 답변을 비교
    print(f"Correct Answer: {correct_answer}")
    print(f"Generated Answer: {cleaned_answer}")
    if answer == cleaned_answer:
        correct += 1
        print(answer,"correct!!")
 
print("===== Perf result =====")
print("Elapsed_time: ", end-start)
print(f"Correctness: {correct}/{len(data)}")