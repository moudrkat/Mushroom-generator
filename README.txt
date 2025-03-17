

mushroom_generator/
├── app_train_model_GAN.py              # streamlit app for model developer (me)
├──app_train_model_VAE                  # streamlit app for model developer (me). I actually used this VAE model
├──app_latent_mycelium                  # streamlit app for model developer (me). This app is used to explore latent space
├── app_generate_mushroom.py            # streamlit app for users (you)
├── src/                                 
│   ├── __init__.py                      
│   ├── data_preprocessing.py           # Contains data processing functions
│   ├── gan_model.py                    # Contains your GAN model functions
│   ├── utils.py                        # Any utility functions (like image loading)
|
|-- streamlit frontend
│   ├── __init__.py                      
│   ├── streamlit_frontend.py           # Contains function used in streamlit (mostly dataviz)
├──
├── requirements.txt                # Python dependencies 
├── README.txt                       # Project description
└── .gitignore                      # 


Welcome to the Mushroom Generator App, where you can explore the fascinating world of latent space to generate mushroom images! 🌱🍄

This app is built using a Variational Autoencoder (VAE), which allows you to experiment with latent vectors and see how they affect the resulting mushroom images. You can also visualize the inner workings of each decoder layer, making it easier to understand how the model generates these mushrooms from scratch.

Features
Generate Mushrooms: The app uses a VAE model to generate realistic bitmap mushroom images from latent vectors.
Explore Latent Space: Modify the latent vector and watch how the generated mushrooms change. This allows you to see the relationship between the latent space and the final mushroom image.
Visualize Decoder Layers: View the intermediate steps of the VAE decoder, from reshaping the latent vector to progressively refining the image.


---------------------------------------------

Initially, I experimented with a Generative Adversarial Network (GAN) model, but I found that the latent space it generated wasn’t interpretable. This made it difficult to explore and manipulate the generated images in a meaningful way. As a result, I switched to the Variational Autoencoder (VAE), which provides a more interpretable and continuous latent space.

I also tried experimenting with various sketches from the Quickdraw dataset to generate different objects, but ultimately, mushrooms proved to be the best choice. Their distinct characteristics and variety made them an excellent subject for exploring latent space and generating visually interesting outputs.

--------------------------------------------

resources >

https://www.kaggle.com/code/vincentman0403/vae-with-convolution-on-mnist