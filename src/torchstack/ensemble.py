from transformers import AutoTokenizer
import torch

# library structures
from torchstack.member import AutoModelMember
from torchstack.configuration import Configuration
from torchstack.strategies import BaseStrategy

# MAIN CLASS
class EnsembleForCausalLM(torch.nn.Module):
    def __init__(self, strategy, config = Configuration, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.members = []
        self.strategy: BaseStrategy = strategy
        self.locked: bool = False
    
    # BUILDER: handles new models in the ensemble
    def add_member(self, model: AutoModelMember, tokenizer: AutoTokenizer):
        if self.locked:
            raise ValueError("The ensemble is locked and cannot accept new members.")
        self.members.append((model.to(self.device), tokenizer))

    # BUILDER: handles the ensemble preparation and locking
    def prepare(self):
        if self.locked:
            raise ValueError("The ensemble has already been prepared.")
        
        if not self.strategy:
            raise ValueError("A valid strategy must be provided.")
        
        if not self.members:
            raise ValueError("No members added to the ensemble.")

        # Dependency injection: Pass members to the strategy
        models, tokenizers = zip(*self.members)  # Unzip members into separate lists
        
        try:
            print("Preparing Specialization strategy...")
            self.strategy.prepare(models=models, tokenizers=tokenizers, device=self.device)  # Explicitly call the strategy's prepare method
        except Exception as e:
            raise ValueError(f"PREPARE: Could not prepare strategy: {e}")
        
        # lock the model to prevent further changes
        self.locked = True

       
    def generate(self, prompt, max_length=25):
        if not self.locked:
            raise ValueError("The ensemble must be prepared before generation.")
        
        # Dependency injection: let strategy handle the generation
        return self.strategy.generate(prompt, max_length)
