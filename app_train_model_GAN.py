import streamlit as st
import tensorflow as tf
import numpy as np
from src.data_preprocessing import load_data, normalize_images
from src.gan_model import build_generator, build_discriminator, compile_gan, train_gan
import matplotlib.pyplot as plt
from src.utils import extract_last_word_from_filename

# Streamlit UI setup
st.title("Quickdraw GAN")
st.write("Generate sketches using a simple GAN")

# Load and preprocess data
data_file = st.file_uploader("Upload your dataset (.npz)", type=["npz"])

if data_file is not None:
    
    # Split the file name by underscore and get the last part
    sketch_type = extract_last_word_from_filename(data_file)

    data = np.load(data_file)
    images = data['images']
    images = normalize_images(images)

    st.write("Dataset loaded, training model...")

    image_placeholder = st.empty() 
    image_placeholder_loss = st.empty() 

    # GAN setup
    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    gan = compile_gan(generator, discriminator)

    # GAN Training
    epochs = 5000
    batch_size = 64

    train_gan(sketch_type, generator, discriminator, gan, images, image_placeholder,image_placeholder_loss, epochs=epochs, batch_size=64, latent_dim=100)

    st.write("Training complete!")

