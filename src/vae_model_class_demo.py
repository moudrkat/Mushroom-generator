import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from src.utils import save_generated_images, show_images_in_streamlit
from src.data_preprocessing import denormalize_images
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class VAE:
    def __init__(self, latent_dim=64, r=0.5, update_rate=0.002):
        # Set parameters
        self.latent_dim = latent_dim
        self.r = r
        self.update_rate = update_rate
        
        # Initialize encoder and decoder
        self.encoder = self.build_encoder(latent_dim)
        self.decoder = self.build_decoder(latent_dim)
        
        # Initialize optimizer
        self.optimizer = Adam(learning_rate=1e-4)

    def build_encoder(self, latent_dim):
        inputs = layers.Input(shape=(28, 28, 1))

        # Encoder convolutional layers with Batch Normalization
        x = layers.Conv2D(32, 3, activation=None, padding='same')(inputs)
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

        # Flatten and create latent space representation (mean and log-variance)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)

        return models.Model(inputs, [z_mean, z_log_var])

    def build_decoder(self, latent_dim):
        latent_inputs = layers.Input(shape=(latent_dim,))

        # Decoder dense layer to reshape latent vector
        x = layers.Dense(7 * 7 * 128, activation='relu')(latent_inputs)
        x = layers.Reshape((7, 7, 128))(x)

        # Decoder with Conv2DTranspose layers
        x = layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)

        # Final Conv2D layer to output the grayscale image
        decoded = layers.Conv2D(1, 3, activation='tanh', padding='same')(x)

        return models.Model(latent_inputs, decoded)

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @staticmethod
    def update_beta(epoch, max_beta=1.0, anneal_rate=0.002):
        """Update beta value for KL annealing."""
        return min(max_beta, anneal_rate * epoch)

    @tf.function
    def train_step(self, images, beta):
        with tf.GradientTape() as tape:
            # Forward pass through the encoder
            z_mean, z_log_var = self.encoder(images)

            # Sampling from the latent space
            z = self.sampling([z_mean, z_log_var])

            # Forward pass through the decoder
            reconstructed = self.decoder(z)

            # VAE loss (reconstruction + KL divergence)
            reconstruction_loss = tf.reduce_mean(tf.square(images - reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )

            vae_loss = reconstruction_loss + beta * kl_loss

        # Compute gradients and update the model
        grads = tape.gradient(vae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return vae_loss

    def train_vae(self, strategy, sketch_type, dataset, image_placeholder, freq_show=10, freq_save=100, epochs=100):
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            # Update beta for the current epoch
            beta = self.update_beta(epoch, anneal_rate=self.update_rate)

            for batch_images in dataset:
                batch_loss = self.train_step(batch_images, beta)
                epoch_loss += batch_loss
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss.numpy()}')

            # Show generated images every 'freq_show' epochs
            if epoch % freq_show == 0:
                random_latent_vectors = tf.random.normal(shape=(64, self.latent_dim))
                generated_images = self.decoder(random_latent_vectors)
                denormalized_generated_images = denormalize_images(generated_images)
                show_images_in_streamlit(batch_images, denormalized_generated_images, epoch, image_placeholder)

            # Save generated images every 'freq_save' epochs
            if epoch % freq_save == 0:
                random_latent_vectors = tf.random.normal(shape=(64, self.latent_dim))
                generated_images = self.decoder(random_latent_vectors)
                denormalized_generated_images = denormalize_images(generated_images)
                save_generated_images(sketch_type, denormalized_generated_images, epoch, path=f"./generated_images_{strategy}_{sketch_type}")

        # Save the encoder and decoder models after training
        self.encoder.save(f"trained_encoder_{strategy}_{sketch_type}_final.h5")
        self.decoder.save(f"trained_decoder_{strategy}_{sketch_type}_final.h5")
        print(f"Encoder and Decoder models saved successfully.")

    def generate_image(self):
        random_latent_vectors = np.random.normal(size=(1, self.latent_dim))
        generated_image = self.decoder.predict(random_latent_vectors)
        return generated_image[0]
