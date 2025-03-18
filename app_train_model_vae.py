import streamlit as st
import tensorflow as tf
import numpy as np
from src.data_preprocessing import load_data, normalize_images, create_dataset
import matplotlib.pyplot as plt
from src.utils import extract_last_word_from_filename
import tensorflow as tf
from src.data_preprocessing import create_dataset
from src.vae_model import train_vae, encoder, decoder, simple_decoder

st.title("Quickdraw GAN")
st.write("Generate sketches using a simple GAN")

strategy = "VAE"

# Load and preprocess data
# data_file = st.file_uploader("Upload your dataset (.npz)", type=["npz"])

data_file ="./data/filtered_dataset_mushroom.npz"  

if data_file is not None:
    
    # Split the file name by underscore and get the last part
    sketch_type = extract_last_word_from_filename(data_file)

    data = np.load(data_file)
    images = data['images']
    images = normalize_images(images)

    # Get the number of images
    num_images = images.shape[0]

    print(f"Number of images: {num_images}")

    st.write("Dataset loaded, training model...")

    image_placeholder = st.empty() 
    image_placeholder_loss = st.empty() 



    def vae_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss):
                
                latent_dim = 2
            
                encoder_model = encoder(latent_dim)
                decoder_model = decoder(latent_dim)

                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96, staircase=True)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

                #parameter for KL annealing
                p_update_rate = 0.002

                # GAN Training
                n_epochs = 10
                batch_size = 32  

                dataset = create_dataset(images, batch_size,limit =100000)

                # how often are results saved and displayed
                n_freq_show = 1
                n_freq_save = 1

                # Start training
                train_vae( strategy,
                        sketch_type,
                        optimizer,
                        dataset,
                        encoder_model,
                        decoder_model,
                        image_placeholder,
                        n_freq_show,
                        n_freq_save,
                        n_epochs,
                        latent_dim,
                        r = 0.999,
                        update_rate = p_update_rate )
            
    vae_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss)

