# magenta-torch
Pytorch Implementation of MusicVAE with LSTM and GRU architectures, allowing for integrating danceability as an additional feature to condition upon. 

Code belonging to the MSc. Thesis "Creating meaningful and controllable latent representations of music using VAEs."  - S.A.J. Wijtsma
See abstract below. 

## Usage
### Setup
    git clone https://github.com/st33f/magenta-torch.git
    cd magenta-torch
    pip install -r requirements.txt

### Logging
This project makes use of Weights & Biases for online logging and experiment tracking. Having an account with their platform is mandatory. Before running the scrips, you to login from you local machine by running:
    
    wandb login

### Dataset
To train a model on you own dataset, you must first use the preprocessing script on your MIDI files. Make sure to use the format of the Lakh dataset (https://colinraffel.com/projects/lmd/), which has subfolders per artist with the MIDI filenames containing the trackname. The script will preprocess all MIDI files, collect additional musical features on the track from the Spotify API and create a dataset in the desired format:

    python /scripts/preprocess.py --import_dir=<YOUR_MIDI_DIRECTORY> 

__Note that to retrieve the features from the Spotify API, your own valid credentials for the Spotify API are assumed to be available as environment variables. See `src/spotify.py`__

### Training a model
To train a model, first define the desired hyperparameters in the configuration file `conf.yml`, then run:

     python /scripts/train.py --conf=conf.yml --model_type=gru --epochs=10 
     
Results will be logged to Weights and Biases. 

# Abstract
Deep generative models are increasingly being used to exploit 'machine intelligence' for creative purposes. The Variational autoencoder (VAE) has proven to be an effective model for capturing (and generating) the dynamics of musical compositions. The VAEs latent representation can be used in all kinds of creative applications such as controlled music generation or interpolation between two musical sequences.
From a practical or creative standpoint it would be very useful if a user could manipulate meaningful semantic aspects of musical features (like changing danceability, mood or energy) directly via the latent variables. By including additional features explicitly in the latent variables, higher-level semantic knowledge can be integrated as a support of the raw symbolic representations. More importantly, having access to these semantically meaningful features in the latent variables potentially enables creative operations further down the line, such as sampling, interpolation and controlled generation based on these exact features.

This work proposes a recurrent neural network architecture based on VAEs that explicitly embeds high-level musical features (danceability) to learn a latent representation of music and subsequently generate music through decoding this representation. 

Specifically, the aim is to enrich the raw symbolic note events with a high-level feature that is meaningful to a user, and integrate these in the latent variables with the aim of performing creative operations using musical features, such as increasing the danceability of an existing song. Hereby, creating an interactive, controllable model enhancing the user with high-level creative power over the music generation process. A quantitative analysis is carried out to evaluate the robustness of the proposed system and compare the performance of VAEs as generative architectures with or without explicitly encoded additional features. 
