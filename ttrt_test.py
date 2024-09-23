import time
import torch
import torch_tensorrt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

torch.cuda.empty_cache()
####### Section 1. Set up #######
torch.random.manual_seed(0)
model_id = "./models/Phi-3-medium-4k-instruct"  # Replace with your local model path

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
model.eval()  # Set the model to evaluation mode
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare a sample input for tracing
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Define a wrapper to simplify the model's forward method
class TRTWrapper(torch.nn.Module):
    def __init__(self, model):
        super(TRTWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Instantiate the wrapper
trt_model = TRTWrapper(model).to("cuda")

# Trace the model with TorchScript
scripted_model = torch.jit.trace(
    trt_model, 
    (inputs["input_ids"], inputs["attention_mask"]),
    strict=False
)

# Compile the model with Torch-TensorRT
trt_compiled_model = torch_tensorrt.compile(
    scripted_model,
    inputs=[
        torch_tensorrt.Input(
            inputs["input_ids"].shape, dtype=torch.int32
        ),
        torch_tensorrt.Input(
            inputs["attention_mask"].shape, dtype=torch.int32
        ),
    ],
    enabled_precisions={torch.float},
)

# Update the pipeline to use the compiled model
pipe = pipeline(
    "text-generation",
    model=trt_compiled_model,
    tokenizer=tokenizer,
    device=0,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

####### Section 2. GPU Warm-up #######
messages = [
    {
        "role": "user",
        "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
    },
    {
        "role": "assistant",
        "content": (
            "Sure! Here are some ways to eat bananas and dragonfruits together: "
            "1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. "
            "2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."
        ),
    },
    {
        "role": "user",
        "content": "What about solving a 2x + 3 = 7 equation?",
    },
]
output = pipe(messages, **generation_args)
# print(output[0]['generated_text'])

####### Section 3. Load data and Inference -> Performance Evaluation Part #######
start = time.time()
data = load_dataset("json", data_files="./data/test_dataset.jsonl")["train"]
outs = pipe(KeyDataset(data, "message"), **generation_args)
end = time.time()

####### Section 4. Accuracy (Just for Leaderboard) #######
print("===== Answers =====")
correct = 0
for i, out in enumerate(outs):
    correct_answer = data[i]["answer"]
    answer = out[0]["generated_text"].lstrip().replace("\n", "")
    if answer == correct_answer:
        correct += 1
    # print(answer)

print("===== Perf Result =====")
print("Elapsed Time: ", end - start)
print(f"Correctness: {correct}/{len(data)}")
