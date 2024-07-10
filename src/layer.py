import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class Layer:
    def __init__(self, model, weight) -> None:
        """
        Initialize the Layer with a pre-trained model and tokenizer.
        :param model_name: The name of the pre-trained model.
        """
        self.model = model
        self.weigth = 1.0
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

    def set_prompt(self, prompt):
        prompt = self.tokenizer(prompt, return_tensors="pt")

    def get_top_k(self, k: int = 10):
        """
        Get the top-k token probabilities from the model output.
        :param k: The number of top tokens to retrieve.
        :return: A list of tuples containing the top-k tokens and their probabilities.
        """
        if self.prompt is None:
            raise ValueError("Prompt not set. Use set_prompt() to set the prompt before calling get_top_k.")

        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**self.inputs)
            logits = outputs.logits

        probabilities = F.softmax(logits, dim=-1)           # Convert logits to probabilities
        last_token_probabilities = probabilities[0, -1, :]  # Get the probabilities for the last token

        # Convert probabilities to a more readable format
        probs = last_token_probabilities.cpu().numpy()

        # Get the top 10 probabilities
        top_k = 10
        top_k_indices = probs.argsort()[-top_k:][::-1]
        top_k_probs = probs[top_k_indices]
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices)

        top_k_indices = last_token_probabilities.argsort()[-k:][::-1]
        top_k_probs = last_token_probabilities[top_k_indices]
        top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices)

        return list(zip(top_k_tokens, top_k_probs))

        return zip(top_k_tokens, top_k_probs)
