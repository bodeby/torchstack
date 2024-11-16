from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class BaseLearner(AutoModelForCausalLM):
    def __init__(self) -> None:
        super().__init__()