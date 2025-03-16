import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from src.utils import save_generated_images, show_images_in_streamlit
from src.data_preprocessing import denormalize_images
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def encoder(latent_dim=64):
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Add Conv2D layers with Batch Normalization
    x = layers.Conv2D(32, 3, activation=None,  padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(64, 3, activation=None, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(64, 3, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Flatten and create the latent space representation
    x = layers.Flatten()(x)

    x = layers.Dense(32, activation='relu')(x)  # Reduced dense layer size
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    return models.Model(inputs, [z_mean, z_log_var])


def decoder(latent_dim=64):
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    # Dense layer to reshape the latent vector
    x = layers.Dense(7 * 7 * 128, activation='relu')(latent_inputs)  # Adjusted size to match 28x28 output
    x = layers.Reshape((7, 7, 128))(x)  # Reshape to 7x7x128
    
    # Decoder part with Conv2DTranspose layers
    x = layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)  # 14x14
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)   # 28x28
    
    # Final Conv2D layer to output the grayscale image
    decoded = layers.Conv2D(1, 3, activation='tanh', padding='same')(x)  # Output shape (28, 28, 1)
    
    return models.Model(latent_inputs, decoded)

def simple_decoder(latent_dim=64):
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    # Dense layer to reshape the latent vector, but with fewer units
    x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)  # Smaller size
    x = layers.Reshape((7, 7, 64))(x)  # Reshape to 7x7x64
    
    # Decoder part with Conv2DTranspose layers
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)  # 14x14
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)   # 28x28
    
    # Final Conv2D layer to output the grayscale image
    decoded = layers.Conv2D(1, 3, activation='tanh', padding='same')(x)  # Output shape (28, 28, 1)
    
    return models.Model(latent_inputs, decoded)

# KL Annealing: Function to update beta
def update_beta(epoch, max_beta=1.0, anneal_rate=0.002):
    """
    Update beta value (KL term weight) for annealing. Starts at 0 and increases
    to `max_beta` over time. 
    """
    beta = min(max_beta, anneal_rate * epoch)
    return beta


@tf.function
def train_step(encoder, decoder, images, optimizer,r,beta):
    with tf.GradientTape() as tape:
        # Forward pass through the encoder
        z_mean, z_log_var = encoder(images)
        
        # Sampling from the latent space
        z = sampling([z_mean, z_log_var])
        
        # Forward pass through the decoder
        reconstructed = decoder(z)
        
        # Calculate VAE loss (reconstruction + KL divergence)
        reconstruction_loss = tf.reduce_mean(tf.square(images - reconstructed))

        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )

        # vae_loss = r * reconstruction_loss + (1-r) * kl_loss
        vae_loss = reconstruction_loss + beta * kl_loss
    # Compute gradients and update the model
    grads = tape.gradient(vae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))

    return vae_loss
# , reconstruction_loss, kl_loss

# Training loop
def train_vae( strategy, sketch_type,optimizer, dataset, encoder,decoder, image_placeholder, freq_show=10, freq_save=100, epochs=100,latent_dim=100,r=0.5, update_rate = 0.002):
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        # Update beta for the current epoch
        beta = update_beta(epoch,anneal_rate= update_rate)

        for batch_images in dataset:
            batch_loss = train_step(encoder,decoder, batch_images, optimizer,r,beta)
            epoch_loss += batch_loss
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss.numpy()}')

        # Show generated images every 'freq_show' epochs
        if epoch % freq_show == 0:

            random_latent_vectors = tf.random.normal(shape=(64, latent_dim))  # Generate 20 random latent vectors
            generated_images = decoder(random_latent_vectors)  # Decode them to generate images

            denormalized_generated_images = denormalize_images(generated_images) 

            show_images_in_streamlit(batch_images, denormalized_generated_images, epoch, image_placeholder)

        # Save generated images every 'freq_save' epochs
        if epoch % freq_save == 0:

            # Generate random latent vectors
            random_latent_vectors = tf.random.normal(shape=(64, latent_dim))  # Generate 20 random latent vectors
            # print("saving before")
            generated_images = decoder(random_latent_vectors)  # Decode them to generate images
            denormalized_generated_images = denormalize_images(generated_images) 
            save_generated_images(sketch_type, denormalized_generated_images, epoch, path=f"./generated_images_{strategy}_{sketch_type}")

    
    # Save both the encoder and decoder models after training
    encoder.save(f"trained_encoder_{strategy}_{sketch_type}_final.h5")
    decoder.save(f"trained_decoder_{strategy}_{sketch_type}_final.h5")
    print(f"Encoder and Decoder models saved successfully.")

def generate_image(decoder_model, latent_dim):
    random_latent_vectors = np.random.normal(size=(1, latent_dim))
    generated_image = decoder_model.predict(random_latent_vectors)
    return generated_image[0]


