import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
# import potrace


# Define a function to load the model and apply caching
@st.cache_resource
def load_keras_model():
    try:
        # Attempt to load the Keras model
        generator = load_model('trained_decoder_VAE_mushroom_final.h5')
        return generator
    except Exception as e:
        # Handle any errors that may occur during model loading
        st.error(f"Error loading the model: {e}")
        raise e  # Re-raise the exception after logging it
    
# Function to generate and display the dragon
def generate_mushroom():
    latent_dim = 100  # Latent space size for the generator
    # noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise for the latent vector
    generated_image = generator.predict(latent_vector)  # Generate the image using the generator
    generated_image = (generated_image + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
    generated_image = np.clip(generated_image, 0.0, 1.0)  # Clip values to be in [0, 1]

    # Adjust the color of the generated dragon
    mushroom_image = generated_image[0, :, :, 0]  # Get the single generated image (28x28)

    # Convert the color from hex to RGB
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Color the image by applying the selected color
    colored_image = np.stack([mushroom_image * (rgb_color[0] / 255),  # Red channel
                              mushroom_image * (rgb_color[1] / 255),  # Green channel
                              mushroom_image * (rgb_color[2] / 255)], axis=-1)  # Blue channel
    
    # Convert to PIL Image
    image_pil = Image.fromarray((colored_image * 255).astype(np.uint8))

    # Step 2: Apply Sharpening filter using PIL
    sharpness_filter = ImageFilter.UnsharpMask(radius=8, percent=200, threshold=1)
    sharpened_image = image_pil.filter(sharpness_filter)

    # Create a plot figure
    fig, ax = plt.subplots(figsize=(1, 1))  
    plt.gcf().set_facecolor('black')
    ax.imshow(sharpened_image)
    ax.axis('off')  # Turn off axis
    # Display the image in Streamlit
    st.pyplot(fig,use_container_width=True)


# Load the model into session state if it is not already there
if 'generator' not in st.session_state:
    try:
        st.session_state.generator = load_keras_model()
    except Exception:
        st.stop()  # Stop execution if the model loading failed

# Access the model from session state
generator = st.session_state.generator

# Streamlit UI
st.title("Mushroom Generator")

col1, col2, col3 = st.columns([0.5, 0.1, 0.4])

with col1:
    # Color picker to choose the dragon color
    color = st.color_picker("Choose the mushrooms's color", "#FFFFFF")  # Default color blue

    hat_size = st.slider("Choose the the mushrooms's cap size", 0.0, 4.5, 2.5, 0.1)  # Hat size slider
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; margin-top: -20px;">
            <span>Small</span>
            <span>Large</span>
        </div>
        """, unsafe_allow_html=True
    )

    # Add more vertical space
    st.markdown("<br>", unsafe_allow_html=True)  

    leg_size = st.slider("Choose the the mushrooms's stem length", 0.0, 4.5, 2.5, 0.1)  # Leg size slider
    st.markdown(
    """
    <div style="display: flex; justify-content: space-between; margin-top: -20px;">
        <span>Short</span>
        <span>Long</span>
    </div>
    """, unsafe_allow_html=True
)

    # Generate the latent vector
    latent_vector = np.array([-(hat_size-3.0), -(leg_size-3.0)])
    # Reshape it to (1, 2) to represent a batch of size 1 with 2 dimensions
    latent_vector = latent_vector.reshape(1, 2)  # Shape becomes (1, 2)

with col3:
    # Add more vertical space
    st.markdown("<br>", unsafe_allow_html=True)  
    generate_mushroom()

# Add more vertical space
st.markdown("<br>", unsafe_allow_html=True)  

show_details = st.toggle("Show how the mushroom is generated")

if show_details:
    st.write(f"Latent vector: {latent_vector}")