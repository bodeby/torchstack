from abc import ABC, abstractmethod
import torch

class BaseStrategy(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.models = None
        self.tokenizers = None
        self.device = None

    def initialize(self, models, tokenizers, device):
        self.models = models
        self.tokenizers = tokenizers
        self.device = device

    @abstractmethod
    def prepare(self):
        """Abstract method that must be implemented by subclasses"""
        pass

    @abstractmethod
    def generate(self, prompt, max_length=25):
        """Abstract method that must be implemented by subclasses"""
        pass
