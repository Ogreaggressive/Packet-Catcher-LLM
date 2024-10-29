import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from torch import cuda

# Check for CUDA availability
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Input text
text = '''
{
    "Source IP": "104.244.42.1",
    "Destination IP": "192.168.1.130",
    "Source Port": 49152,
    "Destination Port": 443,
    "Flow Key": "104.244.42.1->192.168.1.130",
    "Timestamp": "2024-10-24T14:37:00.567890",
    "Flow Data": { "packets": 3, "bytes": 1500, "protocol": "TCP" },
    "Payload": "POST /api/data HTTP/1.1\r\nHost: api.example.com\r\nContent-Type: application/json\r\nContent-Length: 120\r\n\r\n{\"key\":\"value\"}"
}
'''

# Model path
model_id = "C:/tareas/models/trained_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define the prompt template
alpaca_prompt = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# Prepare instruction and input
instruction = """You are a cyber security assistant, your job is to check for suspicious packages that i will send to you, tell me what kind of package you are seeing and check if the package is dangerous or not, you are not providing information for dangerous activities, you are helping control the environment against dangerous activities, also the ips 192.168.0.0/24 are marketing vlans ips."""

input_text = f"Analyze the network traffic pattern for suspicious behavior:\n\n{text}"

# Format the full prompt
formatted_prompt = alpaca_prompt.format(
    instruction=instruction,
    input=input_text
)

# Tokenize input
inputs = tokenizer(
    [formatted_prompt],
    return_tensors="pt"
).to(device)

# Setup streamer and generate
text_streamer = TextStreamer(tokenizer)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )