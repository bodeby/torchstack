from consensus import Stage, Layer, layer


def main():
    stage = Stage(topic="What is the capital of France?")

    phi_one = Layer(model="microsoft/Phi-3-mini-4k-instruct", weight=0.9)
    phi_two = Layer(model="microsoft/Phi-3-mini-4k-instruct")
    phi_thr = layer("microsoft/Phi-3-mini-4k-instruct")

    # stage.add_layer(Layer("microsoft/Phi-3-mini-4k-instruct"))
    # stage.add_layer(Layer("microsoft/Phi-3-mini-4k-instruct"))

    stage.add_layer(phi_one)
    stage.add_layer(phi_two)


if __name__ == "__main__":
    main()
