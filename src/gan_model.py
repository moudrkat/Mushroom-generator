import tensorflow as tf
import numpy as np
from src.utils import save_generated_images, show_images_in_streamlit, show_loss_acc_in_streamlit
from src.data_preprocessing import denormalize_images

    
def build_generator(latent_dim=100, width=64, drop=0.4):
    model = tf.keras.Sequential()
    
    # First Dense Layer: Fully connected to generate the initial feature map
    model.add(tf.keras.layers.Dense(7 * 7 * width, input_dim=latent_dim))  # Reshape to 7x7 feature map with 'width' channels
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Reshape((7, 7, width)))  # Reshape to 7x7x64
    model.add(tf.keras.layers.Dropout(drop))
    
    # First Upsampling Layer
    model.add(tf.keras.layers.UpSampling2D())  # Upsample by factor of 2
    model.add(tf.keras.layers.Conv2DTranspose(int(width / 2), kernel_size=5, padding='same', activation=None))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Activation('relu'))
    
    # Second Upsampling Layer
    model.add(tf.keras.layers.UpSampling2D())  # Upsample by factor of 2
    model.add(tf.keras.layers.Conv2DTranspose(int(width / 4), kernel_size=5, padding='same', activation=None))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Activation('relu'))
    
    # Third Upsampling Layer (no need to use UpSampling2D since the final output is 28x28)
    model.add(tf.keras.layers.Conv2DTranspose(int(width / 8), kernel_size=5, padding='same', activation=None))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
    model.add(tf.keras.layers.Activation('relu'))
    
    # Output Layer: Generate final 28x28 grayscale image
    model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=7, padding='same', activation='tanh'))  # Image output in range [-1, 1]
    
    return model


def build_discriminator(img_width=28, img_height=28, width=64, p=0.4):
    model = tf.keras.Sequential()
    
    # First convolutional layer
    model.add(tf.keras.layers.Conv2D(width*1, kernel_size=5, strides=2, padding='same', input_shape=(img_width, img_height, 1)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Second convolutional layer
    model.add(tf.keras.layers.Conv2D(width*2, kernel_size=5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Third convolutional layer
    model.add(tf.keras.layers.Conv2D(width*4, kernel_size=5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Fourth convolutional layer
    model.add(tf.keras.layers.Conv2D(width*8, kernel_size=5, strides=1, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Flatten layer to feed into a dense output layer
    model.add(tf.keras.layers.Flatten())
    
    # Output layer: Single neuron with sigmoid activation
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Use 0 to 1 output range
    
    # compile model
    #opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    opt = tf.keras.optimizers.RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def compile_gan(generator, discriminator):
    
    # # Freeze the discriminator's weights during the GAN training
    discriminator.trainable = False
    
    # # Optimizer for the generator
    lr_gen = 0.0004
    optimizer_gen = tf.keras.optimizers.RMSprop(lr=lr_gen, decay=3e-8, clipvalue=1.0)

    # GAN is a combined model of generator and discriminator
    gan = tf.keras.Sequential([generator, discriminator])
    
    # Compile the GAN model with the optimizer for the generator
    gan.compile(loss='binary_crossentropy', optimizer=optimizer_gen)
    
    return gan


def train_gan(sketch_type,generator, discriminator, gan, images, image_placeholder,image_placeholder_loss, epochs=100, batch_size=64, latent_dim=100):
    # Initialize lists to store losses
    g_losses = []
    d_losses = []
    d_accuracies = []

    half_batch = batch_size // 2  # Half batch size for real/fake images
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Latent noise for generator

        generated_images = generator.predict(noise)  # Generate fake images

        # 1. Train the discriminator with real and fake images
        discriminator.trainable = True  # Unfreeze the discriminator to train it
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]  # Select real images from the dataset
        real_labels = np.ones((half_batch, 1))  # Label for real images: 1

        fake_images = generated_images[:half_batch]  # Use first half of generated images
        fake_labels = np.zeros((half_batch, 1))  # Label for fake images: 0

        d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # Average of both losses
        d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)  # Average of both accuracies
        
        # 2. Train the generator (through the GAN model) every epoch
        discriminator.trainable = False  # Freeze the discriminator (now only the generator will be trained)
        valid_labels = np.ones((batch_size, 1))  # Fake images are labeled as real for the generator
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Append losses for this epoch to the respective lists
        g_losses.append(g_loss)
        d_losses.append(d_loss)  
        d_accuracies.append(d_acc)  

        # Every few epochs, print the progress and save the model
        if epoch % 100 == 0:  # Save model and show images every 100 epochs
            generator.save(f"./trained_generators_{sketch_type}/trained_generator_epoch_{epoch}.h5")  # Save model
            #st.write(f"Saved generator model at epoch {epoch}")
            #st.write(f"Epoch {epoch}/{epochs} | D Loss: {d_loss} | G Loss: {g_loss}")
            save_generated_images(sketch_type,generated_images, epoch, path=f"./generated_images_{sketch_type}")

            denormalized_fake_images = denormalize_images(fake_images)  # Denormalize for display
            show_images_in_streamlit(real_images, denormalized_fake_images, epoch, image_placeholder)  # Show images in Streamlit
            
            show_loss_acc_in_streamlit(g_losses, d_losses, d_accuracies, epoch,epochs, image_placeholder_loss)

    # After training completes, save the final generator model
    generator.save(f"trained_generator_{sketch_type}_final.h5")


