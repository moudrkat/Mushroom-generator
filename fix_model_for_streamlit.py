import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import layers, models


# Load the existing model
model = load_model('trained_decoder_VAE_mushroom_finalANNEAL.h5')

# Print the model architecture to inspect its layers
model.summary()

new_model_layers = []
for layer in model.layers:
    if isinstance(layer, layers.Conv2DTranspose):
        # Get the layer configuration
        config = layer.get_config()
        
        # If 'groups' is present in the config, remove it
        if 'groups' in config:
            del config['groups']
        
        # Create a new layer with the modified config (without 'groups')
        new_layer = layers.Conv2DTranspose(**config)
        new_model_layers.append(new_layer)
    else:
        # If the layer is not 'Conv2DTranspose', just append it unchanged
        new_model_layers.append(layer)

# Now, build a new model with the modified layers
new_model = models.Sequential(new_model_layers)


# Save the modified model
new_model.save('fixed_VAE_model.h5')

# Verify if the model was saved correctly
new_model = load_model('fixed_VAE_model.h5')
new_model.summary()