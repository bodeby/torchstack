from typing import List, Tuple
from transformers import AutoTokenizer

# library structures
from consensus.member import AutoModelMember
#from consensus.tokenizer import Tokenizer
from consensus.configuration import Configuration


class Ensemble:
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.members: List[AutoModelMember] = []  # Store Members
        self.tokenizers: List[AutoTokenizer] = [] # Store Tokenizers
        self.vocabulary = None # TODO: find the type of union vocab

    def __repr__(self):
        # Create a string representation of the ensemble architecture
        ensemble_info = [
            f"Model: {member[0].__class__.__name__}, Tokenizer: {member[1].__class__.__name__}, Weight: {member[0].weight}"
            for member in self.members
        ]
        return f"Ensemble(config={self.config}, Members=[{', '.join(ensemble_info)}])"

    def generate(prompt: str):
        pass

    def add_member(self, model_member: AutoModelMember, tokenizer: AutoTokenizer):
        if not isinstance(model_member, AutoModelMember):
            raise ValueError("Member must be an instance of AutoModelMember")
        
        if not isinstance(tokenizer, AutoTokenizer):
            raise ValueError("Tokenizer must be an instance of AutoTokenizer")

        self.members.append(model_member)
        self.tokenizers.append(tokenizer)
    
    def create_union_vocab(self):
        if (len(self.tokenizers) <= 0):
            raise ValueError("Can not created union vocab when ensemble tokenizers is not initialized")

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
