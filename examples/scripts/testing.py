from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch

# typings
from typing import List, Tuple

# Load the model and tokenizer, FIX: Phi-3 has issues
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)       # tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)    # tokenizer


# Load the model and tokenizer, FIX: Phi-3 has issues
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to get top-k tokens and probabilities
def get_top_k(prompt: str, k: int = 10) -> List[Tuple[str, float]]:
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Get the probabilities for the last token in the sequence
    last_token_probabilities = probabilities[0, -1, :]

    # Get the top-k token indices and their corresponding probabilities
    top_k_indices = last_token_probabilities.argsort()[-k:][::-1]
    top_k_probs = last_token_probabilities[top_k_indices].cpu().numpy()
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

    return list(zip(top_k_tokens, top_k_probs))

# Prompt
prompt = "What is the capital of France?"

# Get the top-k tokens and probabilities
top_k = get_top_k(prompt, k=10)

# Print the results
for token, prob in top_k:
    print(f"Token: {token}, Probability: {prob:.4f}")
