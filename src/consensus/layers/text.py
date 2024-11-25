from base import Layer


class TextLayer(Layer):
    def __init__(self, model: str, weight: float) -> None:
        super().__init__(model, weight)
