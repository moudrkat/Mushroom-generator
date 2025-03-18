import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
from src.show_activations_vae import get_activations_model, get_layer_activations
from tensorflow.keras import layers

# Define a function to load the model and apply caching
@st.cache_resource
def load_keras_model():
    try:
        # Attempt to load the Keras model
        generator = load_model('trained_decoder_VAE_mushroom_finalANNEAL.h5')
        return generator
    except Exception as e:
        # Handle any errors that may occur during model loading
        st.error(f"Error loading the model: {e}")
        raise e  # Re-raise the exception after logging it

# Function to generate and display the mushroom
def generate_mushroom(generator, latent_vector, color):
    # latent_dim = 100  # Latent space size for the generator
    # noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise for the latent vector
    generated_image = generator.predict(latent_vector)  # Generate the image using the generator
    generated_image = (generated_image + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
    generated_image = np.clip(generated_image, 0.0, 1.0)  # Clip values to be in [0, 1]

    # Adjust the color of the generated shroom
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
    fig.patch.set_facecolor('none')  # Set the figure background to transparent
    ax.patch.set_facecolor('none')   # Set the axes background to transparent
    st.pyplot(fig,use_container_width=True)

    return sharpened_image


def show_latent_contour(hat_size, leg_size):
        # Create a grid of points (latent space)
    x = np.linspace(-3, 3, 100)  # Range of values for latent dimension 1
    y = np.linspace(-3, 3, 100)  # Range of values for latent dimension 2
    x_grid, y_grid = np.meshgrid(x, y)

    # Define the standard Gaussian function (mean=0, std=1)
    def gaussian_2d(x, y, sigma=1):
        return (1 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    # Compute the Gaussian PDF for each point in the grid
    z = gaussian_2d(x_grid, y_grid)

    # Create the contour plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cp = ax.contour(x_grid, y_grid, z, levels=10, cmap='gray', alpha=0.5)

    # Add labels and title
    ax.set_xlabel('$z_1$', color='red')
    ax.set_ylabel('$z_2$', color='red')
    # ax.set_title("Contour plot of latent 'mycelium'", color='white')
    fig.patch.set_facecolor('black')  # Set the figure background to transparent
    ax.patch.set_facecolor('none')

    # Set the color of the axis ticks to white
    ax.tick_params(axis='both', colors='white')

    # Add the image annotation at the point (hat_size, leg_size)
    # You can control the image size by adjusting 'extent' parameters.
    # imagebox = ax.imshow(mushroom, aspect='auto', extent=[hat_size - 0.5, hat_size + 0.5, leg_size - 0.5, leg_size + 0.5], zorder=5)

    # Show the point (z1, z2) on the plot
    ax.scatter(hat_size, leg_size, color='red', s=100, label=f'Point ({hat_size}, {leg_size})')  # Add the point in red

    # Annotate the point with its coordinates
    ax.annotate(f'({hat_size}, {leg_size})', (hat_size, leg_size), textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    font_properties_funny = {
    'family': 'Chalkboard', 
    'weight': 'bold',           
    'size': 10,                 
    }

    # Add quadrant labels
    ax.text(-2.2, 2.5, 'Tiny Toadstools', color='white', ha='center',fontdict=font_properties_funny)
    ax.text(2.5, 2.5, 'Bizzare Schrooms',  color='white', ha='center',fontdict=font_properties_funny)
    ax.text(-2.2, -2.5, 'Cap-tastic Giants',  color='white', ha='center',fontdict=font_properties_funny)
    ax.text(2.5, -2.5, 'Stretchy Stems',  color='white', ha='center',fontdict=font_properties_funny)

    # Add center label
    ax.text(0, 0, 'Average Mushrooms',  color='white', ha='center', va='center',fontdict=font_properties_funny)

    # Show the plot in Streamlit
    st.pyplot(fig)

def show_mushroom_grow(generator, latent_vector):
        # Get activations for each layer in the model
    model = generator

    activation_model = get_activations_model(model)
    activations = get_layer_activations(activation_model, latent_vector)

    # Display the activations and layer descriptions in Streamlit
    for i, layer_activation in enumerate(activations):
        st.write(f"### Layer {i + 1}: {model.layers[i].name}")
        
        # Show a description of the layer
        if isinstance(model.layers[i], layers.InputLayer):
            st.write("The input layer receives the latent vector. This is the starting point for the decoder model.")

        elif isinstance(model.layers[i], layers.Dense):
            st.write("This is a Dense layer that performs a fully connected transformation. It expands the latent vector into a higher-dimensional tensor suitable for the next layers.")

        elif isinstance(model.layers[i], layers.Reshape):
            st.write("This is a Reshape layer. It converts the output of the Dense layer into a 3D shape (height, width, channels), preparing it for the convolutional operations that follow.")

        elif isinstance(model.layers[i], layers.Conv2DTranspose):
            if model.layers[i].filters == 128:
                st.write("This is the first Conv2DTranspose layer. It works by applying 128 learned filters to the input tensor, upscaling it from 7x7 to 14x14, while preserving spatial relationships.")
            else:
                st.write("This is the second Conv2DTranspose layer. It continues the upscaling process by increasing the tensor size from 14x14 to 28x28 via 64 learned filters.")

        elif isinstance(model.layers[i], layers.Conv2D):
            st.write("This is a Conv2D layer - the final step. It takes all the learned features from the previous 64 filters and combines them into a single image. And voilà, there's your mushroom, popping up as the output, ready for the show!")

        # Visualization for Layer 1 (input_2) - 2D vector
        if i == 0:  # Layer 1 - Input Layer (Activation shape: (1, 2))
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.imshow(layer_activation[0, :].reshape(1, -1), cmap='gray', aspect='auto')

            ax.text(0.1, 0.0, f"{layer_activation[0, 0]:.2f}", ha='center', va='center', color='red', fontsize=12)
            ax.text(0.9, 0.0, f"{layer_activation[0, 1]:.2f}", ha='center', va='center', color='red', fontsize=12)

            ax.axis('off')  # Turn off axes
            fig.patch.set_facecolor('none')  # Set the figure background to transparent
            ax.patch.set_facecolor('none')   # Set the axes background to transparent
            st.pyplot(fig)
        
        # Visualization for Layer 2 (dense_3) - 1D vector (6272 activations)
        if i == 1:  # Layer 2 - Dense Layer (Activation shape: (1, 6272))
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.imshow(layer_activation[0, :].reshape(1, -1), cmap='gray', aspect='auto')
            ax.axis('off')  # Turn off axes
            fig.patch.set_facecolor('none')  # Set the figure background to transparent
            ax.patch.set_facecolor('none')   # Set the axes background to transparent
            st.pyplot(fig)
        
        # Plot the activations for Layer 3 (index 2), Layer 4 (index 3), and Layer 5 (index 4)
        if i == 2:  # Layer 3 - Reshape Layer (128 filters)
            num_filters = layer_activation.shape[-1]  # Number of filters
            grid_size = 12  # Grid size

            # Create the grid with sufficient size
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

            axes = axes.flatten()  # Flatten axes for easier iteration

            for j in range(num_filters):  # Loop over each filter
                ax = axes[j]
                ax.imshow(layer_activation[0, :, :, j], cmap='gray')
                ax.axis('off')

            # Hide unused subplots if there are any
            for j in range(num_filters, len(axes)):
                axes[j].axis('off')

            fig.patch.set_facecolor('none')  # Set the figure background to transparent
            ax.patch.set_facecolor('none')   # Set the axes background to transparent
            st.pyplot(fig)

        if i == 3:  # Layer 4 - Conv2DTranspose Layer 
            num_filters = layer_activation.shape[-1]  # Number of filters
            grid_size = 12  # Grid size

            # Create the grid with sufficient size
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

            axes = axes.flatten()  # Flatten axes for easier iteration

            for j in range(num_filters):  # Loop over each filter
                ax = axes[j]
                ax.imshow(layer_activation[0, :, :, j], cmap='gray')
                ax.axis('off')

            # Hide unused subplots if there are any
            for j in range(num_filters, len(axes)):
                axes[j].axis('off')
            fig.patch.set_facecolor('none')  # Set the figure background to transparent
            ax.patch.set_facecolor('none')   # Set the axes background to transparent
            st.pyplot(fig)

        if i == 4:  # Layer 5 - Conv2DTranspose Layer 
            num_filters = layer_activation.shape[-1]  # Number of filters
            grid_size = 8  # Grid size 

            # Create the grid with sufficient size
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

            axes = axes.flatten()  # Flatten axes for easier iteration

            for j in range(num_filters):  # Loop over each filter
                ax = axes[j]
                ax.imshow(layer_activation[0, :, :, j], cmap='gray')
                ax.axis('off')

            # Hide unused subplots if there are any
            for j in range(num_filters, len(axes)):
                axes[j].axis('off')
            fig.patch.set_facecolor('none')  # Set the figure background to transparent
            ax.patch.set_facecolor('none')   # Set the axes background to transparent
            st.pyplot(fig)

            # Visualization for Layer 6 (conv2d_4) - Final output layer 
        if i == 5:  # Layer 6 - Final Conv2D layer
            final_activation = layer_activation[0, :, :, 0]  # Extract the image from the 4D output
            fig, ax = plt.subplots(figsize=(0.7, 0.7))
            ax.imshow(final_activation, aspect='auto', cmap='gray')  
            ax.axis('off')  # Turn off axes
            fig.patch.set_facecolor('none')  # Set the figure background to transparent
            ax.patch.set_facecolor('none')   # Set the axes background to transparent
            st.pyplot(fig, use_container_width=False)


def contact_form():
    # Instructions
    st.markdown(
        "<p style='color: gray;'>Do you have any ideas for improving the app or even creating a better model? The author is eager to learn and would greatly appreciate your feedback! :)</p>", 
        unsafe_allow_html=True
    )

    # HTML Form to send data to Formspree with gray text
    contact_form = """
    <form action="https://formspree.io/f/xzzezrqe" method="POST">
    <label style="color: gray;">
        Your email:
        </br>
        <input type="email" name="email">
    </label>
    </br>
    <label style="color: gray;">
        Your message:
        </br>
        <textarea name="message"></textarea>
    </label>
    </br>
    <button type="submit">Send</button>
    </form>
    """

    # Display the form in Streamlit
    st.markdown(contact_form, unsafe_allow_html=True)

def link_to_other_apps():
        # Links with gray text
    st.markdown(
        "<p style='color: gray;'>Want to play with CNNs? <a href='https://applepear.streamlit.app' style='color: gray;'>Check out this app</a>. </p>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color: gray;'>Interested in exploring TensorFlow optimizers? <a href='https://minimize-me.streamlit.app' style='color: gray;'>Check out this app</a>.</p>", 
        unsafe_allow_html=True
    )