import numpy as np
import tensorflow as tf
import os


def create_dataset(images_array, batch_size, limit = 5000, shuffle=True):
    # Ensure the images_array is a NumPy array and has the shape (num_images, height, width, channels)
    images_array = np.array(images_array)

    # Shuffle the images array
   
    np.random.shuffle(images_array)
    
    # Limit the dataset size if limit is provided
    if limit is not None:
        images_array = images_array[:limit]
    
    # Create TensorFlow dataset from the images array
    dataset = tf.data.Dataset.from_tensor_slices(images_array)
    
    # Shuffle the dataset if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

def normalize_images(images):
    # Normalizes images from 0-255 to -1 to 1
    images = (images.astype(np.float32) - 127.5) / 127.5
    return images

def load_data(file_path):
    data = np.load(file_path)
    images = data['images']
    return images

def denormalize_images(images):
    # Convert from [-1, 1] to [0, 255]
    images = (images + 1) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images