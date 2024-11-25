from base import Layer


class CodeLayer(Layer):
    def __init__(self, model: str, weight: float) -> None:
        super().__init__(model, weight)
