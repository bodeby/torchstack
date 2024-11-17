import torch
from typing import List, Tuple

# library structures
from consensus.member import AutoModelMember
from consensus.tokenizer import Tokenizer
from consensus.configuration import Configuration


class Ensemble:
    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.members: List[
            Tuple[AutoModelMember, Tokenizer]
        ] = []  # Store (model, tokenizer) pairs

    def add_member(self, model_member: AutoModelMember, tokenizer: Tokenizer):
        if not isinstance(model_member, AutoModelMember):
            raise ValueError("Member must be an instance of AutoModelMember")
        if not isinstance(tokenizer, Tokenizer):
            raise ValueError("Tokenizer must be an instance of Tokenizer")

        self.members.append((model_member, tokenizer))

    def _tokenizer_from_model(self):
        """Helper to extract tokenizers from models if needed."""
        return [member[1] for member in self.members]
