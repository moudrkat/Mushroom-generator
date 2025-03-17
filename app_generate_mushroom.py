import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
from src.show_activations_vae import get_activations_model, get_layer_activations
from tensorflow.keras import layers, models 
from tensorflow.keras.utils import plot_model
import io
from streamlit_frontend import load_keras_model, generate_mushroom, show_latent_contour, show_mushroom_grow, contact_form, link_to_other_apps


# Load the model into session state if it is not already there
if 'generator' not in st.session_state:
    try:
        st.session_state.generator = load_keras_model()
    except Exception:
        st.stop()  # Stop execution if the model loading failed

# Access the model from session state
generator = st.session_state.generator

# Streamlit UI
st.title("Got vectors? Create mushrooms.")

st.markdown("""
    **Hello! I’m a GEN AI model designed to generate mushrooms with a Variational Autoencoder (VAE) architecture.** 

    Using just one latent vector —**two numbers**— I can create a unique mushroom! \\
    These two numbers— :red[$z_1$] and :red[$z_2$] —specify a position in the latent space, which is continuous and can be easily interpolated, allowing for smooth transitions and manipulation of key features of the mushroom.

    #### What you can do:
    - Adjust :red[$z_1$] to change the **cap size**.
    - Adjust :red[$z_2$] to control the **stem length**.
    - Watch how these two numbers combine to generate completely different mushrooms.
    - Have fun :)

""")

col1, col2, col3 = st.columns([0.4, 0.1, 0.5])

with col1:
    # Color picker to choose the dragon color
    color = st.color_picker("Choose the mushrooms's color", "#FFFFFF")  # Default color blue

    hat_size = st.slider("Choose :red[$z_1$] (~the mushrooms's cap size)",-3.0, 3.0, 0.0, 0.05)  # Hat size slider
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; margin-top: -20px;">
            <span>Large</span>
            <span>Small</span>
        </div>
        """, unsafe_allow_html=True
    )

    # Add more vertical space
    st.markdown("<br>", unsafe_allow_html=True)  

    leg_size = st.slider("Choose :red[$z_2$] (~the mushrooms's stem length)", -3.0, 3.0, 0.0, 0.05)  # Leg size slider
    st.markdown(
    """
    <div style="display: flex; justify-content: space-between; margin-top: -20px;">
        <span>Long</span>
        <span>Short</span>
    </div>
    """, unsafe_allow_html=True
)

    # Generate the latent vector
    latent_vector = np.array([(hat_size), (leg_size)])
    # Reshape it to (1, 2) to represent a batch of size 1 with 2 dimensions
    latent_vector = latent_vector.reshape(1, 2)  # Shape becomes (1, 2)

with col3:
    latex_str = r"\left( \begin{matrix} \color{red}z_1" r" \\ " r"\color{red}z_2 \end{matrix} \right) =   \left( \begin{matrix} \color{red}" + str(hat_size) + r" \\ \color{red}" + str(leg_size) + r" \end{matrix} \right)"
    st.latex(latex_str)
    generate_mushroom(generator, latent_vector, color)

# Create an expander
with st.expander("Click to see your mushrooms position in latent space:"):
    # Add more vertical space
    st.markdown("<br>", unsafe_allow_html=True)  
    show_latent_contour(hat_size, leg_size)
    st.markdown("<br>", unsafe_allow_html=True) 

# Create an expander
with st.expander("Click to see how your mushroom 'grows' from the latent vector:"):
    st.write("The latent vector passes through six layers in the decoding process to generate a mushroom image. Each layer contributes to gradually transforming the latent vector into a full-sized image. Let's inspect each layer in detail.")
    show_mushroom_grow(generator, latent_vector)

st.markdown("<br> <br>", unsafe_allow_html=True) 
st.markdown("---")
contact_form()
st.markdown("<br> <br>", unsafe_allow_html=True)
link_to_other_apps()

