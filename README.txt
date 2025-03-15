It is possible to train quickdraw sketches generator for any sketch type. It is also possible to explore latent space of generated images intearctively. I didnt deploy the app, as I didnt find the latent sopace exploration giving interpretable results.

mushroom_generator/
├── app_train_model.py              # streamlit app for model developer (me)
├── app_generate_mushroom.py        # streamlit app for future users 
├── src/                            # 
│   ├── __init__.py                 # 
│   ├── data_preprocessing.py       # Contains data processing functions
│   ├── gan_model.py                # Contains your GAN model functions
│   ├── utils.py                    # Any utility functions (like image loading)
├── requirements.txt                # Python dependencies 
├── README.txt                       # Project description
└── .gitignore                      # 


resources >

https://www.kaggle.com/code/vincentman0403/vae-with-convolution-on-mnist