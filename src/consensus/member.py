from transformers import AutoModelForCausalLM


class AutoModelMember(AutoModelForCausalLM):
    def __init__(self, path: str, weight: float) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.weight = self._is_valid_weight(weight)

    def _is_valid_weight(self, weight: float):
        if not (0.0 <= weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0")
        return weight
