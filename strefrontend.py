import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved generator model
generator = load_model("trained_generator_final.h5")  # Replace with your model filename

# Streamlit UI
st.title("Dragon Generator")
st.write("Click below to generate your dragon!")

# Color picker to choose the dragon color
color = st.color_picker("Choose the dragon's color", "#0000FF")  # Default color blue

# Function to generate and display the dragon
def generate_dragon():
    latent_dim = 100  # Latent space size for the generator
    noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise for the latent vector
    generated_image = generator.predict(noise)  # Generate the image using the generator

    # Adjust the color of the generated dragon
    dragon_image = generated_image[0, :, :, 0]  # Get the single generated image (28x28)

    # Convert the color from hex to RGB
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Color the image by applying the selected color
    colored_image = np.stack([dragon_image * (rgb_color[0] / 255),  # Red channel
                              dragon_image * (rgb_color[1] / 255),  # Green channel
                              dragon_image * (rgb_color[2] / 255)], axis=-1)  # Blue channel

    # Display the generated dragon
    plt.figure(figsize=(5, 5))
    plt.imshow(colored_image)
    plt.axis('off')
    st.pyplot()  # Show the image in Streamlit

# Button to trigger the dragon generation
if st.button("Generate Dragon"):
    generate_dragon()