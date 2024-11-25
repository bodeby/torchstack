import numpy as np
from typing import List
from transformers import AutoTokenizer

# library structures
from consensus.member import AutoModelMember

# from consensus.tokenizer import Tokenizer
from consensus.configuration import Configuration


class Ensemble:
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.members: List[AutoModelMember] = []  # Store Members
        self.tokenizers: List[AutoTokenizer] = []  # Store Tokenizers
        self.vocabulary = None  # TODO: find the type for union vocab
        self.vocabulary_index = None  # TODO: find the type for index
        self.aligned: bool = False

    def __repr__(self):
        # Create a string representation of the ensemble architecture
        ensemble_info = [
            f"Model: {member[0].__class__.__name__}, Tokenizer: {member[1].__class__.__name__}, Weight: {member[0].weight}"
            for member in self.members
        ]
        return f"Ensemble(config={self.config}, Members=[{', '.join(ensemble_info)}])"

    def generate(self, prompt: str):
        if not self.aligned:
            raise ValueError("Tokenizers must be aligned before generating responses.")
        pass

    def add_member(self, model: AutoModelMember, tokenizer: AutoTokenizer):
        if not isinstance(model, AutoModelMember):
            raise ValueError("Member must be an instance of AutoModelMember")

        if not isinstance(tokenizer, AutoTokenizer):
            raise ValueError("Tokenizer must be an instance of AutoTokenizer")

        self.members.append(model)
        self.tokenizers.append(tokenizer)

    def create_union_vocab(self):
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

    def create_tokenizer_mapping(self, tokenizer: AutoTokenizer):
        # grab vocabulary from tokenizer
        local_vocab = tokenizer.get_vocab()  # Local Tokenizer, is the current tokenizer

        # get vocabulary sizes
        local_length = len(local_vocab)
        union_length = len(self.vocabulary)

        # create mapping matrices for tokenizer 1
        tk_to_union = np.zeros((len(local_length), len(union_length)), dtype=int)

        for token, t1_idx in local_vocab.items():
            union_idx = self.vocabulary_index[token]
            tk_to_union[t1_idx, union_idx] = 1

        return tk_to_union
