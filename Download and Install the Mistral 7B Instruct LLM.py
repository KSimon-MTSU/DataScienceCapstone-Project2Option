#download and install Mistral-7B-Instruct-v0.3

import os
from huggingface_hub import snapshot_download
from pathlib import Path
# from dotenv import load_dotenv
# load_dotenv() 
HFToken = os.getenv("HUGFACE_TOKEN")

mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], 
                    local_dir=mistral_models_path,token=HFToken)

# if Cuda is not installed, download and install from cuda website:
# https://developer.nvidia.com/cuda-downloads


# CLI command for testing chat
# mistral-chat C:/Users/krrat/mistral_models/7B-Instruct-v0.3 --instruct --max_tokens 256


# Use Mistral-7B-Instruct-v0.3
from huggingface_hub import InferenceClient
client = InferenceClient(api_key=HFToken)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message)