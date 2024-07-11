from stage import Stage
from layer import Layer


def main():
    stage = Stage(topic="What is the capital of France?")
    phi_one = Layer(model="microsoft/Phi-3-mini-4k-instruct", weight=0.9)
    stage.add_layer(phi_one)

    for layer in stage.layers:
        print(layer)


if __name__ == "__main__":
    main()
