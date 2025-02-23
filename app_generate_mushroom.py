import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved generator model
generator = load_model("trained_generator_final.h5")  # Replace with your model filename

# Streamlit UI
st.title("Mushroom Generator")
st.write("Click below to generate your mushroom!")

# Color picker to choose the dragon color
color = st.color_picker("Choose the mushrooms's color", "#0000FF")  # Default color blue

# Function to generate and display the dragon
def generate_mushroom():
    latent_dim = 100  # Latent space size for the generator
    noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise for the latent vector
    generated_image = generator.predict(noise)  # Generate the image using the generator

    # Adjust the color of the generated dragon
    mushroom_image = generated_image[0, :, :, 0]  # Get the single generated image (28x28)

    # Convert the color from hex to RGB
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Color the image by applying the selected color


    colored_image = np.stack([mushroom_image * (rgb_color[0] / 255),  # Red channel
                              mushroom_image * (rgb_color[1] / 255),  # Green channel
                              mushroom_image * (rgb_color[2] / 255)], axis=-1)  # Blue channel

    # Create a plot figure
    fig, ax = plt.subplots(figsize=(1, 1))  
    plt.gcf().set_facecolor('black')
    ax.imshow(colored_image)
    ax.axis('off')  # Turn off axis

    # Display the image in Streamlit
    st.pyplot(fig)

# Button to trigger the dragon generation
if st.button("Generate Mushroom"):
    generate_mushroom()