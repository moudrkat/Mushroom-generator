
from src.vae_model import decoder
import tensorflow as tf
from tensorflow.keras import layers, models


# Create the model


# Create another model to extract activations
def get_activations_model(model):
    layer_outputs = [layer.output for layer in model.layers]  # Get the output of all layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # Create a new model
    return activation_model

# activation_model = get_activations_model(model)

# Function to get activations for a given input
def get_layer_activations(model, input_data):
    activations = model.predict(input_data)
    return activations

