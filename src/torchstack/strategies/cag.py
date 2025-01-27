# Paper title: Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling
# arXiv: 2406.1285v2
# Authors: Yao-Ching Yu, Chun-Chih Kuo, Ziqi Ye, Yu-Cheng Chang, Yueh-Se Li

import torch
import numpy as np

# base class with required abstractclasses
from torchstack.strategies import BaseStrategy

class GenerationAsClassification(BaseStrategy):
    def __init__(self, models=None, tokenizers=None, device=None):
        super().__init__()
        self.models = models
        self.tokenizers = tokenizers
        self.device = device
        self.initialized: bool = False

        # Strategy-specific attributes
        self.union_vocab = None
        self.union_vocab_with_index = None
        self.mappings = []
        self.weights = None  # Initialize weights as None

    def _create_vocab(self):
        vocabularies = [tokenizer.get_vocab() for tokenizer in self.tokenizers]
        union_vocab = set(token for vocab in vocabularies for token in vocab.keys())
        self.union_vocab = sorted(union_vocab)
        self.union_vocab_with_index = {token: idx for idx, token in enumerate(self.union_vocab)}

    def _create_mappings(self):
        if not self.tokenizers:
            raise ValueError("Tokenizers must be provided.")
        if not self.union_vocab_with_index:
            raise ValueError("Union vocabulary must be created first.")

        self.mappings = []
        for tokenizer in self.tokenizers:
            local_vocab = tokenizer.get_vocab()
            mapping = np.zeros(len(local_vocab), dtype=int)
            for token, index in local_vocab.items():
                mapping[index] = self.union_vocab_with_index.get(
                    token, -1
                )  # Use -1 for missing tokens
            self.mappings.append(torch.tensor(mapping, device=self.device))


    def initialize(self, models=None, tokenizers=None, device=None):
        super().initialize(models=models, tokenizers=tokenizers, device=device)

    def prepare(self):
        """Prepare the strategy by creating vocabularies and mappings."""
        if self.initialized:
            raise RuntimeError("Strategy has already been prepared.")

        # Check that all necessary attributes are set
        if not self.models or not self.tokenizers or not self.device:
            raise ValueError("Strategy must be initialized with models, tokenizers, and a device before preparing.")

        # Create the union vocabulary and mappings
        try:
            self._create_vocab()
            print("Union vocabulary created successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to create union vocabulary: {e}")

        try:
            self._create_mappings()
            print("Vocabulary mappings created successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to create vocabulary mappings: {e}")

        # Mark the strategy as initialized
        self.initialized = True
        print("Strategy preparation complete.")

    @torch.no_grad()
    def generate(self, prompt, max_length=25):
        generated_text = prompt

        # create input ids
        input_ids = [
            tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            for tokenizer in self.tokenizers
        ]

        # check for device
        if self.device is None:
            self.device = input_ids[0].device

        # Initialize weights if not set
        if self.weights is None:
            self.weights = torch.tensor(
                [1.0 / len(self.models)] * len(self.models), device=self.device
            )

        # Check if mappings are empty
        if len(self.mappings) == 0:
            raise ValueError("Mappings are empty. Ensure `prepare` has been called.")

        while len(generated_text.split()) < max_length:
            # Step 1: Compute next-token probabilities for each model
            computed_probs = [
                torch.nn.functional.softmax(model(input_id).logits[:, -1, :], dim=-1)
                for model, input_id in zip(self.models, input_ids)
            ]

            # Check if computed_probs is empty
            if len(computed_probs) == 0:
                raise ValueError("Computed probabilities are empty. Ensure models and inputs are properly initialized.")

            # Step 2: Map probabilities to the union vocabulary
            mapped_probs = []
            for prob, mapping in zip(computed_probs, self.mappings):
                q = torch.zeros(len(self.union_vocab), device=prob.device)
                q.scatter_add_(0, mapping, prob.squeeze(0))
                mapped_probs.append(q)

            if len(mapped_probs) == 0:
                raise ValueError("Mapped probabilities are empty. Ensure mappings and computed_probs are valid.")

            # Step 3: Average probabilities
            average_probs = torch.stack(mapped_probs).mean(dim=0)

            # Step 4: Sample the next token
            next_token_idx = torch.multinomial(average_probs, num_samples=1).item()
            sampled_token = list(self.union_vocab_with_index.keys())[next_token_idx]

            # Step 6: Decode the sampled token
            generated_token = ""
            for tokenizer in self.tokenizers:
                vocab = tokenizer.get_vocab()
                if sampled_token in vocab:
                    token_id = tokenizer.convert_tokens_to_ids(sampled_token)
                    generated_token = tokenizer.decode([token_id])
                    break

            # Fallback if token is not found in any tokenizer
            if not generated_token:
                generated_token = ""  # You could use "<unk>" as a fallback

            # Step 7: Append decoded token to the generated text
            if sampled_token not in [tokenizer.eos_token for tokenizer in self.tokenizers if tokenizer.eos_token]:
                generated_text += generated_token

            # Update input_ids for the next iteration
            input_ids = [
                torch.cat([ids, torch.tensor([[token_id]], device=ids.device)], dim=-1)
                for ids, token_id in zip(input_ids, [tokenizer.convert_tokens_to_ids(sampled_token) or tokenizer.unk_token_id for tokenizer in self.tokenizers])
            ]

            # Handle stopping criterion
            if any(sampled_token == tokenizer.eos_token for tokenizer in self.tokenizers):
                break

        return generated_text
