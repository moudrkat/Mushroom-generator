import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from PIL import Image

# Load your trained GAN model (generator)
generator = tf.keras.models.load_model('trained_generator_final.h5')

# Function to generate image from latent space vector
def generate_image_from_latent(z):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    z = tf.expand_dims(z, axis=0)  # Add batch dimension
    generated_image = generator(z, training=False)

    # Normalize image to [0, 1] if it is in the range [-1, 1]
    generated_image = (generated_image[0].numpy() + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

    # Ensure the image is in the correct shape (height, width, channels)
    generated_image = np.clip(generated_image, 0.0, 1.0)  # Clip values to be in [0, 1]

    # Check if the image is grayscale (1 channel)
    if generated_image.shape[-1] == 1:
        # If grayscale, squeeze the last dimension
        generated_image = np.squeeze(generated_image, axis=-1)
        # Display grayscale image
        return generated_image
    else:
        # If already RGB, return as is
        return generated_image

# Streamlit interface
st.title("GAN Latent Space Exploration")
st.write("Use the sliders to explore the latent space and generate images!")

# Define the original latent vector dimension
original_latent_dim = 100  # Assuming the original latent dimension is 100
latent_dim = 2  # Reduced latent dimension after PCA

# Generate random latent vectors for PCA fitting (500 random samples)
latent_samples = np.random.uniform(-3, 3, (500, original_latent_dim))

# Apply PCA to reduce from 100 to 10 dimensions
pca = PCA(n_components=latent_dim)
pca.fit(latent_samples)

col1, col2 = st.columns(2)
with col1:
    # Display PCA components (if needed)
    #st.write(f"PCA Components (10 principal components): {pca.components_}")

    # Create sliders for each of the 10 principal components
    latent_vector_pca = np.zeros(latent_dim)
    for i in range(latent_dim):
        latent_vector_pca[i] = st.slider(f"PC {i+1}", -3.0, 3.0, 0.0, 0.1)

    # Transform the latent vector back from 10 principal components to the original 100 dimensions
    reconstructed_latent_vector = pca.inverse_transform(latent_vector_pca.reshape(1, -1))

with col2:

    # Generate the image using the reconstructed latent vector
    generated_image = generate_image_from_latent(reconstructed_latent_vector[0])

    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_container_width=True)


# For 2D visualization, we can simply plot the first two principal components
# Here we take the first two elements of the latent vector to plot them in a 2D space
fig = go.Figure(data=go.Scatter(x=[latent_vector_pca[0]], y=[latent_vector_pca[1]], mode='markers'))
fig.update_layout(title="Latent Space Visualization (PCA)", xaxis_title="PC1", yaxis_title="PC2")
st.plotly_chart(fig)