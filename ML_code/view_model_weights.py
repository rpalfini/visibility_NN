import numpy as np
from tensorflow import keras

def main(model_path):
    # Load or create your Keras model
    model = keras.models.load_model(model_path)

    # Get the weights of the model
    weights = model.get_weights()

    # Print the weights
    for i, layer_weights in enumerate(weights):
        print(f"Layer {i} Weights:")
        print(layer_weights)


if __name__ == "__main__":
    model_path = ""
    main(model_path)