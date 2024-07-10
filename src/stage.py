from layer import Layer


class Stage:
    def __init__(self, topic: str) -> None:
        """
        Initialize the Stage with a specific topic.

        :param topic: The topic for the stage.
        """
        self.layers = []
        self.topic = topic

    def add_layer(self, layer: Layer) -> None:
        """
        Add a Layer to the Stage.

        :param layer: The Layer to add.
        """
        if layer not in self.layers:
            self.layers.append(layer)

    def remove_layer(self, layer: Layer) -> None:
        """
        Remove a Layer from the Stage.

        :param layer: The Layer to remove.
        """
        if layer in self.layers:
            self.layers.remove(layer)

    def debate(self) -> List[List[Tuple[str, float]]]:
        """
        Conduct a debate among the Layers in the Stage.

        :return: A list of top-k tokens and their probabilities from each Layer.
        """
        if not self.layers:
            raise ValueError("No layers added. Add layers before calling debate.")

        results = []
        for layer in self.layers:
            top_k = layer.get_top_k()
            results.append(top_k)
            for token, prob in top_k:
                print(f"Token: {token}, Probability: {prob:.4f}")
        return results


class StageOld:
    def __init__(self, prompt) -> None:
        self.layers = []
        self.graders = []
        self.prompt = prompt  # consensus topic

    def add_layer(self, layer: Layer):
        if layer not in self.layers:
            self.layers.append(layer)

    def pop_layer(self, layer: Layer):
        if layer in self.layers:
            self.layers.pop(layer)

    def debate() -> str:
        for layer in self.layers:
            top_k = layer.get_top_k
            print(f"Token: {top_k.token}, Probability: {top_k.prob:.4f}")
