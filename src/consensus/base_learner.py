from transformers import AutoModelForCausalLM


class BaseLearner(AutoModelForCausalLM):
    def __init__(self) -> None:
        super().__init__()
