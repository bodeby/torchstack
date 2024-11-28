import numpy as np
from typing import List
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

# library structures
from torchstack.member import AutoModelMember

# from torchstack.tokenizer import Tokenizer
from torchstack.configuration import Configuration


class Ensemble:
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.members: List[AutoModelMember] = []  # Store Members
        self.tokenizers: List[AutoTokenizer] = []  # Store Tokenizers
        self.tokenizer_maps: List[np.ndarray] = None
        self.vocabulary: set = None
        self.vocabulary_index: dict = None
        self.aligned: bool = False

    def __repr__(self):
        # Create a string representation of the ensemble architecture
        ensemble_info = [
            f"Model: {member[0].__class__.__name__}, Tokenizer: {member[1].__class__.__name__}, Weight: {member[0].weight}"
            for member in self.members
        ]
        return f"Ensemble(config={self.config}, Members=[{', '.join(ensemble_info)}])"

    # REQUIRES TOKENIZER
    def _update_union_vocab(self):
        if len(self.tokenizers) <= 0:
            raise ValueError(
                "Can not created union vocab when ensemble tokenizers is not initialized"
            )

        # step 1: Create list of vocabularies in the tokenizers
        vocabularies = [tokenizer.get_vocab() for tokenizer in self.tokenizers]
        vocab_keys = [set(vocab.keys()) for vocab in vocabularies]

        # step 2: Iteratively compute the union of all vocabularies
        union_vocab = set()
        for keys in vocab_keys:
            union_vocab.update(keys)

        # step 3: Sort the union vocabulary for consistent indexing
        union_vocab_sorted = sorted(union_vocab)
        union_vocab_index = {token: idx for idx, token in enumerate(union_vocab_sorted)}

        # step 4: update class variables to contain mapping and index
        self.vocabulary = union_vocab_sorted
        self.vocabulary_index = union_vocab_index

    # REQUIRES UNION VOCABULARY
    def _create_tokenizer_mapping(self, tokenizer: AutoTokenizer):
        if not self.vocabulary:
            raise ValueError("Union Vocabulary have been created yet")

        # grab vocabulary from tokenizer
        local_vocab = tokenizer.get_vocab()  # Local Tokenizer, is the current tokenizer
        local_length = len(local_vocab)

        mapping = np.zeros(local_length, dtype=int)
        for token, index in local_vocab.items():
            mapping[index] = self.vocabulary_index[token]

        self.tokenizer_maps.append(mapping)

    def add_member(self, model: AutoModelMember, tokenizer: AutoTokenizer):
        if not isinstance(model, AutoModelMember):
            raise ValueError("Member must be an instance of AutoModelMember")

        # handle member models
        self.members.append(model)

        if not isinstance(tokenizer, AutoTokenizer):
            raise ValueError("Tokenizer must be an instance of AutoTokenizer")

        # handle member tokenizers
        self.tokenizers.append(tokenizer)
        self._update_union_vocab()
        self._create_tokenizer_mapping(tokenizer)

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 10):
        if not self.aligned:
            raise ValueError("Tokenizers must be aligned before generating responses.")
        
        # Step 1: Tokenize input prompt
        #I1 = t1(prompt, return_tensors="pt").input_ids.to('cuda')  # Token IDs for model 1
        #I2 = t2(prompt, return_tensors="pt").input_ids.to('cuda')  # Token IDs for model 2

        tokenized_inputs = []
        for tokenizer in self.tokenizers:
            tokenized = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
            tokenized_inputs.append(tokenized)

        # Initialize generation
        generated_text = prompt

        for _ in range(max_length):  # TODO: Update example max generation length
            # Step 2: Generate next-token probabilities

            #p1 = F.softmax(m1(I1).logits[:, -1, :], dim=-1)  # Shape: (1, |V1|)
            #p2 = F.softmax(m2(I2).logits[:, -1, :], dim=-1)  # Shape: (1, |V2|)

            computed_probabilities = []
            for model, index in self.models:
                tokenized = tokenized_inputs[index]
                probability = F.softmax(model(tokenized).logits[:, -1, :], dim=-1)
                computed_probabilities.append(probability)

            # Step 3: Map probabilities to the union vocabulary

            # q1 = torch.zeros(len(self.vocabulary), device=p1.device)  # Union vocab size
            # q2 = torch.zeros(len(self.vocabulary), device=p2.device)  # Union vocab size

            # q1.scatter_add_(0, torch.tensor(t1_mapping, device=p1.device), p1.squeeze(0))
            # q2.scatter_add_(0, torch.tensor(t2_mapping, device=p2.device), p2.squeeze(0))

            mapped_probabilities = []
            for prob, index in computed_probabilities:
                mapping = self.tokenizer_maps[index]
                q = torch.zeros(len(self.vocabulary), device=prob.device)
                q.scatter_add_(0, torch.tensor(mapping, device=prob.device), prob.squeeze(0))
                mapped_probabilities.append(q)

            # Step 4: Average probabilities to get ensemble distribution

            # q = (q1 + q2) / 2  # Ensemble by averaging
            average = torch.stack(mapped_probabilities).mean(dim=0)

            # Step 5: Sample the next token

            next_token_idx = torch.multinomial(average.squeeze(0), num_samples=1).item()  # Scalar index
            sampled_token = list(self.vocabulary_index.keys())[next_token_idx]

            # Step 6: Convert sampled token back to token IDs for each tokenizer

            # t1_token_id = t1.convert_tokens_to_ids(sampled_token) if sampled_token in vocabularies[0] else t1.unk_token_id
            # t2_token_id = t2.convert_tokens_to_ids(sampled_token) if sampled_token in vocabularies[1] else t2.unk_token_id

            token_ids = []
            for tokenizer, index in self.tokenizers:
                token_id = tokenizer.convert_tokens_to_ids(sampled_token) if sampled_token in self.vocabularies[index] else tokenizer.unk_token_id
                token_ids.append(token_id)

            # Step 7: Update input sequences

            # I1 = torch.cat([I1, torch.tensor([[t1_token_id]], device=I1.device)], dim=-1)
            # I2 = torch.cat([I2, torch.tensor([[t2_token_id]], device=I2.device)], dim=-1)

            inputs = []
            for token_id, index in token_ids:
                tokenized_input = tokenized_inputs[index]
                new_input = torch.cat([tokenized_input, torch.tensor([[token_id]], device=tokenized_input.device)], dim=-1)
                inputs.append(new_input)

            # Add the sampled token to the generated text
            generated_text += self.tokenizers[0].decode([token_ids[0]])  # Use tokenizer 1 for decoding

            # Stopping criteria: e.g., max length or EOS token
            if sampled_token in [t.eos_token for t in self.tokenizers] or len(generated_text.split()) >= max_length:
                break
        
        return generated_text

