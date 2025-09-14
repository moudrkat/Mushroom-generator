import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.title("Latent space mycelium :)")

loading_message = st.text("Loading mushroom generator... Please wait.")

# Load the model
# generator = tf.keras.models.load_model('trained_generator_final_mushrooms.h5')
decoder = tf.keras.models.load_model('trained_decoder_VAE_mushroom_finalANNEAL.h5')

# Once the model is loaded, remove the loading message
loading_message.empty()

# Function to generate image from latent space vector
def generate_image_from_latent(z):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    z = tf.expand_dims(z, axis=0)  # Add batch dimension

    # generated_image = generator(z, training=False)

    generated_image = decoder.predict(z)

    # Normalize image to [0, 1] if it is in the range [-1, 1]
    # generated_image = (generated_image[0].numpy() + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
    generated_image = (generated_image[0] + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
    generated_image = np.clip(generated_image, 0.0, 1.0)  # Clip values to be in [0, 1]
    return generated_image

# Latent vector count selection
latent_vector_count = st.select_slider(
    'Select number of mushrooms (latent vectors)',
    options=[100, 200, 300, 400, 500,1000],
    value=100  # Default value
)

# Button to generate mushrooms
generate_button = st.button('Generate Mushrooms (and explore the latent space)')

# Ensure mushrooms are only generated when button is clicked
if generate_button:
    # Generate random latent vectors with the selected count
    # latent_vectors = np.random.uniform(-3, 3, (latent_vector_count, 2))  # Random latent vectors with the selected count
    latent_vectors = np.random.normal(0, 1, (latent_vector_count, 2))  # Random latent vectors from a Gaussian distribution with mean=0, std=1

    # # Apply PCA to reduce from 100 to 2 dimensions for visualization
    pca = PCA(n_components=2)
    latent_vectors_pca = pca.fit_transform(latent_vectors)

    # # Step 2: Apply t-SNE on PCA result
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)  # 2D t-SNE for visualization
    latent_vectors_tsne = tsne.fit_transform(latent_vectors)
    # # latent_vectors_tsne = latent_vectors_pca

    latent_vectors_tsne =latent_vectors

    # Generate corresponding images for the latent vectors
    generated_images = np.array([generate_image_from_latent(z) for z in latent_vectors])

    # Store latent vectors, images, and selected latent_vector_count in session state
    st.session_state.latent_vectors = latent_vectors
    st.session_state.generated_images = generated_images
    st.session_state.latent_vectors_pca = latent_vectors_pca
    st.session_state.latent_vectors_tsne = latent_vectors_tsne
    st.session_state.latent_vector_count = latent_vector_count  # Store the latent vector count in session state
else:
    # If button is not pressed, load stored data from session state (if exists)
    if 'latent_vectors' in st.session_state and 'generated_images' in st.session_state:
        latent_vectors = st.session_state.latent_vectors
        generated_images = st.session_state.generated_images
        latent_vectors_pca = st.session_state.latent_vectors_pca
        latent_vectors_tsne = st.session_state.latent_vectors_tsne
    else:
        latent_vectors = []
        generated_images = []
        latent_vectors_pca = []
        latent_vectors_tsne = []

# Create a scatter plot of the 2D latent vectorss


    fig = go.Figure(data=go.Scatter(x=latent_vectors_tsne[:, 0], y=latent_vectors_tsne[:, 1], mode='markers'))

    fig.update_layout(
        title="Latent Space Visualization (PCA + t-SNE)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        clickmode='event+select'  # Enable click events
    )

    # Capture the selected points using the plotly event (on_select)
    selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    # If selected points exist, display details
    if selected_points:
        # If "selection" key is in selected_points, display the points data
        if "selection" in selected_points:

            # Extract points and point_indices
            points = selected_points["selection"].get("points", [])
            point_indices = selected_points["selection"].get("point_indices", [])

            if points:

                # Calculate how many columns you need
                num_columns = len(points)

                # Create columns
                columns = st.columns(num_columns)

                # Loop over each point and display the images
                for i, point in enumerate(points):
                    point_index = point.get('point_index', None)  # Get point index if available

                    if point_index is not None:
                        # Retrieve the corresponding image from generated_images
                        selected_image = generated_images[point_index]

                        # Display the image in the corresponding column
                        columns[i].image(selected_image, caption=f"{point_index}", use_container_width=True)

                    else:
                        st.write("Point index not available.")
            else:
                st.write("No points selected.")
