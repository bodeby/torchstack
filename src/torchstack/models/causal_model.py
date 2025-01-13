import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class EnsembleModelForCausalLM(nn.Module):
    def __init__(self, model_names, weights=None):
        super().__init__()
        self.models = nn.ModuleList(
            [AutoModelForCausalLM.from_pretrained(name) for name in model_names]
        )
        self.tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names]
        self.union_vocab = self._create_union_vocab(self.tokenizers)

        # Default weights if not provided
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def _create_union_vocab(self, tokenizers):
        vocab_sets = [set(tokenizer.vocab.keys()) for tokenizer in tokenizers]
        union_vocab = set.union(*vocab_sets)
        self.union_vocab = {token: i for i, token in enumerate(union_vocab)}
        self.vocab_to_token_id = [
            {token: tokenizer.convert_tokens_to_ids(token) for token in union_vocab}
            for tokenizer in tokenizers
        ]
        return self.union_vocab

    def forward(self, input_ids, attention_mask=None):
        # Step 2: Generate next-token probabilities for each model
        logits_list = [
            torch.functional.softmax(
                model(input_ids, attention_mask=attention_mask).logits[:, -1, :], dim=-1
            )
            for model in self.models
        ]

        # Step 3: Map probabilities to the union vocabulary
        union_probs = torch.zeros(
            input_ids.size(0), len(self.union_vocab), device=input_ids.device
        )
        for logits, mapping, weight in zip(
            logits_list, self.vocab_to_token_id, self.weights
        ):
            for token, idx in mapping.items():
                if idx is not None:  # Map only valid tokens
                    union_probs[:, self.union_vocab[token]] += weight * logits[:, idx]

        return union_probs

    def generate(self, input_ids, max_length=20):
        generated_ids = input_ids.clone()
        for _ in range(max_length):
            # Forward pass
            next_token_probs = self.forward(generated_ids)

            # Sample next token
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Stopping criteria (e.g., EOS token)
            if any(
                next_token_id.item()
                in [tokenizer.eos_token_id for tokenizer in self.tokenizers]
            ):
                break

        return generated_ids
