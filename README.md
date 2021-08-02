# magenta-torch
Code belonging to the MSc. Thesis "Creating meaningful and controllable latent representations of music using VAEs." 
S.A.J. Wijtsma

Pytorch Implementation of MusicVAE with LSTM and GRU architectures, allowing for integrating danceability as an additional feature to condition upon. 
Code includes a script to retrieve the danceability feature from the Spotify API. 

Abstract;
Deep generative models are increasingly being used to exploit 'machine intelligence' for creative purposes. The Variational autoencoder (VAE) has proven to be an effective model for capturing (and generating) the dynamics of musical compositions. The VAEs latent representation can be used in all kinds of creative applications such as controlled music generation or interpolation between two musical sequences.
From a practical or creative standpoint it would be very useful if a user could manipulate meaningful semantic aspects of musical features (like changing danceability, mood or energy) directly via the latent variables. By including additional features explicitly in the latent variables, higher-level semantic knowledge can be integrated as a support of the raw symbolic representations. More importantly, having access to these semantically meaningful features in the latent variables potentially enables creative operations further down the line, such as sampling, interpolation and controlled generation based on these exact features.

This work proposes a recurrent neural network architecture based on VAEs that explicitly embeds high-level musical features (danceability) to learn a latent representation of music and subsequently generate music through decoding this representation. 

Specifically, the aim is to enrich the raw symbolic note events with a high-level feature that is meaningful to a user, and integrate these in the latent variables with the aim of performing creative operations using musical features, such as increasing the danceability of an existing song. Hereby, creating an interactive, controllable model enhancing the user with high-level creative power over the music generation process. A quantitative analysis is carried out to evaluate the robustness of the proposed system and compare the performance of VAEs as generative architectures with or without explicitly encoded additional features. 
