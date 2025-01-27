# Paper title: Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling
# arXiv: 2406.1285v2
# Authors: Yao-Ching Yu, Chun-Chih Kuo, Ziqi Ye, Yu-Cheng Chang, Yueh-Se Li

import torch
import numpy as np

# base class with required abstractclasses
from torchstack.strategies import BaseStrategy

class GenerationAsClassification(BaseStrategy):
    def __init__(self, models, tokenizers, device=None):
        super().__init__()
        self.initialized: bool = False

        # strategy specific attributes
        self.union_vocab = None
        self.union_vocab_with_index = None
        self.mappings = []

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

    def initialize(self):
        # Automatically create vocab and mappings during initialization
        try:
          self._create_vocab()
        except Exception as e:
            print(f"strategy: could not create union vocabulary: {e}")

        try:
          self._create_mappings()
        except Exception as e:
            print(f"strategy: could not create vocabulary mappings: {e}")

        self.initialized = True

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

        # Prepare inputs for all models
        model_inputs = [ids.clone() for ids in input_ids]

        while len(generated_text.split()) < max_length:
            # step 1: generate next-token probabilities
            computed_probs = []
            for model, index in zip(self.models, range(len(self.models))):
                tokenized = input_ids[index]
                p = torch.nn.functional.softmax(
                    model(tokenized).logits[:, -1, :], dim=-1
                )

                # add computed probabilities
                computed_probs.append(p)

            # step 2: map probabilities to the union vocabulary
            mapped_probs = []
            for prob, index in zip(computed_probs, range(len(computed_probs))):
                mapping = self.mappings[index]
                q = torch.zeros(len(self.union_vocab), device=prob.device)
                q.scatter_add_(
                    0, torch.tensor(mapping, device=prob.device), prob.squeeze(0)
                )

                # add mapped probiliities
                mapped_probs.append(q)

            # step 3: average probabilities to get ensemble distribution
            # q = (q1 + q2) / 2  # Ensemble by averaging
            average = torch.stack(mapped_probs).mean(dim=0)

            # step 4: Sample the next token
            next_token_idx = torch.multinomial(average.squeeze(0), num_samples=1).item()  # Scalar index
            sampled_token = list(self.union_vocab_with_index.keys())[next_token_idx]

            print(sampled_token)

            # Step 5: convert sampled token back to token IDs for each tokenizer
            vocabularies = [tokenizer.get_vocab() for tokenizer in self.tokenizers]
            token_ids = []
            for tokenizer, index in zip(self.tokenizers, range(len(self.tokenizers))):
                token_id = (
                    tokenizer.convert_tokens_to_ids(sampled_token)
                    if sampled_token in vocabularies[index]
                    else tokenizer.unk_token_id
                )
                token_ids.append(token_id)

            # step 6: decode and append, TODO: fix manual decoding

            if sampled_token in vocabularies[0]:
                generated_token = self.tokenizers[0].decode(
                    [self.tokenizers[0].convert_tokens_to_ids(sampled_token)]
                )
            elif sampled_token in vocabularies[1]:
                generated_token = self.tokenizers[1].decode(
                    [self.tokenizers[1].convert_tokens_to_ids(sampled_token)]
                )
            else:
                generated_token = ""  # Handle missing tokens

            # step 7: append decoded token to generated text
            generated_text += (
                generated_token
                if sampled_token not in [t1.eos_token, t2.eos_token]
                else ""
            )

            # step 8: update inputs, so that next iter gets correct text
            t1_token_id = (
                self.tokenizers[0].convert_tokens_to_ids(sampled_token)
                or self.tokenizers[0].unk_token_id
            )
            t2_token_id = (
                self.tokenizers[1].convert_tokens_to_ids(sampled_token)
                or self.tokenizers[1].unk_token_id
            )

            # step 9.1: update input_ids
            if t1_token_id is None:
                t1_token_id = (
                    t1.unk_token_id if t1.unk_token_id is not None else 0
                )  # Fallback to 0 if unk_token_id is None
                print(
                    f"Warning: Token '{sampled_token}' not found in t1 vocabulary. Using {t1_token_id} instead."
                )
            else:
                I1 = torch.cat(
                    [
                        input_ids[0],
                        torch.tensor([[t1_token_id]], device=input_ids[0].device),
                    ],
                    dim=-1,
                )

            # step 9.2: update input_ids
            if t2_token_id is None:
                t2_token_id = (
                    t2.unk_token_id if t2.unk_token_id is not None else 0
                )  # Fallback to 0 if unk_token_id is None
                print(
                    f"Warning: Token '{sampled_token}' not found in t1 vocabulary. Using {t2_token_id} instead."
                )
            else:
                I2 = torch.cat(
                    [
                        input_ids[1],
                        torch.tensor([[t2_token_id]], device=input_ids[1].device),
                    ],
                    dim=-1,
                )

            # step 9: handle stopping criterion
            if sampled_token in [t1.eos_token, t2.eos_token]:
                print(f"hit eos token: {t1.eos_token} or {t2.eos_token}")
                break

        return generated_text
