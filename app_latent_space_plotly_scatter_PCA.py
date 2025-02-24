########
#JUST PCA
########

import streamlit as st
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


generator = tf.keras.models.load_model('trained_generator_epoch_1900.h5')

# Function to generate image from latent space vector
def generate_image_from_latent(z):
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    z = tf.expand_dims(z, axis=0)  # Add batch dimension
    generated_image = generator(z, training=False)

    # Normalize image to [0, 1] if it is in the range [-1, 1]
    generated_image = (generated_image[0].numpy() + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
    generated_image = np.clip(generated_image, 0.0, 1.0)  # Clip values to be in [0, 1]
    return generated_image

# Check if the latent vectors and generated images are in session state
if 'latent_vectors' not in st.session_state or 'generated_images' not in st.session_state:
    # Generate random latent vectors 
    latent_vectors = np.random.uniform(-3, 3, (1000, 100))  # Random latent vectors in 100D

    # Apply PCA to reduce from 100 to 2 dimensions
    pca = PCA(n_components=50)
    latent_vectors_pca = pca.fit_transform(latent_vectors)

    # Step 2: Apply t-SNE on PCA result
    tsne = TSNE(n_components=2, random_state=42, perplexity = 3)  # 2D t-SNE for visualization
    latent_vectors_tsne = tsne.fit_transform(latent_vectors)

    # Generate corresponding images for the latent vectors
    generated_images = np.array([generate_image_from_latent(z) for z in latent_vectors])

    # Store latent vectors and generated images in session state
    st.session_state.latent_vectors = latent_vectors
    st.session_state.generated_images = generated_images
    st.session_state.latent_vectors_pca = latent_vectors_pca
    st.session_state.latent_vectors_tsne = latent_vectors_tsne

else:
    # If already in session state, load them
    latent_vectors = st.session_state.latent_vectors
    generated_images = st.session_state.generated_images
    latent_vectors_pca = st.session_state.latent_vectors_pca
    latent_vectors_tsne = st.session_state.latent_vectors_tsne

# Create a scatter plot of the 2D latent vectors
fig = go.Figure(data=go.Scatter(x=latent_vectors_tsne[:, 0], y=latent_vectors_tsne[:, 1], mode='markers'))

fig.update_layout(
    title="Latent Space Visualization (PCA+TSNE)",
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

            # create columns
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

