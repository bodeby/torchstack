import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class Layer:
    def __init__(self, model) -> None:
        self.model = model
        self.model = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

    def set_prompt(self, prompt):
        prompt = self.tokenizer(prompt, return_tensors="pt")

    def get_logits(self, inputs):
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)

        # Get the probabilities for the last token
        last_token_probabilities = probabilities[0, -1, :]

        # Convert probabilities to a more readable format
        probs = last_token_probabilities.cpu().numpy()

        # Get the top 10 probabilities
        top_k = 10
        top_k_indices = probs.argsort()[-top_k:][::-1]
        top_k_probs = probs[top_k_indices]
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices)

        for token, prob in zip(top_k_tokens, top_k_probs):
            print(f"Token: {token}, Probability: {prob:.4f}")


class Stage:
    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, layer: Layer):
        if layer not in self.layers:
            self.layers.append(layer)
    
    def pop_layer(self, layer: Layer):
        if layer in self.layers:
            self.layers.pop(layer)
