from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

class AutoModelMember(AutoModelForCausalLM):
    def __init__(self) -> None:
        super().__init__()

class Tokenizer:
    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        pass

    def from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)