import torch
import torch.nn as nn

from src.layers import *

"""
VAE models
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        print(x.size())
        return x




class MusicLSTMVAE(nn.Module):
    """
    Inputs
    - encoder_input_size: Size of input representation (i.e 60 music pitches + 1 silence)
    - num_subsequences: Number of subsequences to partition input (corresponds to U)
    - sequence_length: Length of sequences (T=32 for 2-bar and T=256 for 16-bar data)
    - encoder_hidden_size: dimension of encoder hidden state
    - decoder_hidden_size: dimension of decoder hidden state
    - latent_dim: dimension of latent variable z
    - encoder_num_layers: Number of encoder lstm layers
    - decoder_num_layers: Number of decoder lstm layers
    """
    def __init__(self, 
                 num_subsequences=16, 
                 max_sequence_length=256,
                 sequence_length=16, 
                 encoder_input_size=61, 
                 decoder_input_size=61, 
                 encoder_hidden_size=2048, 
                 decoder_hidden_size=1024, 
                 latent_dim=512, 
                 encoder_num_layers=2, 
                 decoder_num_layers=2):
        super(MusicLSTMVAE, self).__init__()
        self.input_size = decoder_input_size
        self.max_sequence_length = max_sequence_length
        self.encoder = BiLSTMEncoder(encoder_input_size, 
                                     encoder_hidden_size, 
                                     latent_dim, 
                                     encoder_num_layers)
        self.z_embedding = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.Tanh()
        )
        self.decoder = HierarchicalLSTMDecoder(num_embeddings=num_subsequences, 
                                           input_size=decoder_input_size, 
                                           hidden_size=decoder_hidden_size, 
                                           latent_size=latent_dim, 
                                           num_layers=decoder_num_layers, 
                                           max_seq_length=max_sequence_length,
                                           seq_length=sequence_length)

    def forward(self, x, use_teacher_forcing):
        """
        Input
        - x: input sequence x = x_1, ... ,x_T
        """
        x = x.to(device)
        batch_size = x.size(1)
        h_enc, c_enc = self.encoder.init_hidden(batch_size)
        mu, sigma = self.encoder(x, h_enc, c_enc)

        # Sample latent variable
        with torch.no_grad():
            epsilon = torch.randn_like(mu, device=device)

        z = self.z_embedding(mu + sigma*epsilon)
        h_dec, c_dec = self.decoder.init_hidden(batch_size)
        out = self.decoder(x, z, h_dec, c_dec, use_teacher_forcing)
        return out, mu, sigma, z
    
    def reconstruct(self, x, temperature):
        batch_size = x.size(1)
        h_enc, c_enc = self.encoder.init_hidden(batch_size)
        mu, sigma = self.encoder(x, h_enc, c_enc)
        with torch.no_grad():
            epsilon = torch.randn_like(mu, device=device)
        z = self.z_embedding(mu + sigma*epsilon)
        h_dec, c_dec = self.decoder.init_hidden(batch_size)
        out = self.decoder.reconstruct(z, h_dec, c_dec, temperature)
        return out

class DanceabilityGRUVAE(nn.Module):
    """
    Inputs
    - encoder_input_size: Size of input representation (i.e 60 music pitches + 1 silence)
    - num_subsequences: Number of subsequences to partition input (corresponds to U)        INCREASE
    - sequence_length: Length of sequences (T=32 for 2-bar and T=256 for 16-bar data)       INCREASE
    - encoder_hidden_size: dimension of encoder hidden state
    - decoder_hidden_size: dimension of decoder hidden state
    - latent_dim: dimension of latent variable z
    - encoder_num_layers: Number of encoder lstm layers
    - decoder_num_layers: Numnber of decoder lstm layers
    """
    def __init__(self, 
                 num_subsequences=16, 
                 max_sequence_length=256,
                 sequence_length=16, 
                 encoder_input_size=61, 
                 decoder_input_size=61, 
                 encoder_hidden_size=2048, 
                 decoder_hidden_size=1024, 
                 latent_dim=512, 
                 encoder_num_layers=2, 
                 decoder_num_layers=2,
                 use_danceability=True):
        super(DanceabilityGRUVAE, self).__init__()
        self.use_danceability = use_danceability
        self.input_size = decoder_input_size
        self.max_sequence_length = max_sequence_length
        self.encoder = Danceability_BiGRUEncoder(encoder_input_size,
                                     encoder_hidden_size, 
                                     latent_dim, 
                                     encoder_num_layers)
        self.z_embedding = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.Tanh()#,
            #PrintLayer()
        )
        self.decoder = HierarchicalGRUDecoder(num_embeddings=num_subsequences, 
                                           input_size=decoder_input_size, 
                                           hidden_size=decoder_hidden_size, 
                                           latent_size=latent_dim, 
                                           num_layers=decoder_num_layers, 
                                           max_seq_length=max_sequence_length,
                                           seq_length=sequence_length)

    def forward(self, data, use_teacher_forcing, da=None):
        """
        Input
        - x: input sequence x = x_1, ... ,x_T
        """
        #print(x.size())
        # for i in x:
        #     print(i.size())
        #data, danceability = x
        #print(f"da: {da}")
        #print(f"data model: {data}")
        #data, dance = x
        #print(f"data: {data.size()}")
        batch_size = data.size(1)
        h_enc = self.encoder.init_hidden(batch_size)
        mu, sigma = self.encoder(data, h_enc, da)

        # Sample latent variable
        with torch.no_grad():
            epsilon = torch.randn_like(mu, device=device)

        z = self.z_embedding(mu + sigma*epsilon)
        # print(z)

        # print(z.size())
        h_dec = self.decoder.init_hidden(batch_size)
        out = self.decoder(data, z, h_dec, use_teacher_forcing)
        return out, mu, sigma, z
    
    def reconstruct(self, x, temperature):
        x = x.to(device)
        print(f"X: {x}")
        batch_size = x.size(1)
        h_enc = self.encoder.init_hidden(batch_size)
        mu, sigma = self.encoder(x, h_enc)
        with torch.no_grad():
            epsilon = torch.randn_like(mu, device=device)
        z = self.z_embedding(mu + sigma*epsilon)
        h_dec = self.decoder.init_hidden(batch_size)
        out = self.decoder.reconstruct(z, h_dec, temperature)
        return out


class MusicGRUVAE(nn.Module):
    """
    Inputs
    - encoder_input_size: Size of input representation (i.e 60 music pitches + 1 silence)
    - num_subsequences: Number of subsequences to partition input (corresponds to U)
    - sequence_length: Length of sequences (T=32 for 2-bar and T=256 for 16-bar data)
    - encoder_hidden_size: dimension of encoder hidden state
    - decoder_hidden_size: dimension of decoder hidden state
    - latent_dim: dimension of latent variable z
    - encoder_num_layers: Number of encoder lstm layers
    - decoder_num_layers: Numnber of decoder lstm layers
    """

    def __init__(self,
                 num_subsequences=16,
                 max_sequence_length=256,
                 sequence_length=16,
                 encoder_input_size=61,
                 decoder_input_size=61,
                 encoder_hidden_size=2048,
                 decoder_hidden_size=1024,
                 latent_dim=512,
                 encoder_num_layers=2,
                 decoder_num_layers=2,
                 use_danceability=False):
        super(MusicGRUVAE, self).__init__()
        self.input_size = decoder_input_size
        self.max_sequence_length = max_sequence_length
        self.encoder = BiGRUEncoder(encoder_input_size,
                                    encoder_hidden_size,
                                    latent_dim,
                                    encoder_num_layers)
        self.z_embedding = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.Tanh()
        )
        self.decoder = HierarchicalGRUDecoder(num_embeddings=num_subsequences,
                                              input_size=decoder_input_size,
                                              hidden_size=decoder_hidden_size,
                                              latent_size=latent_dim,
                                              num_layers=decoder_num_layers,
                                              max_seq_length=max_sequence_length,
                                              seq_length=sequence_length)

    def forward(self, x, use_teacher_forcing, da=None):
        """
        Input
        - x: input sequence x = x_1, ... ,x_T
        """
        batch_size = x.size(1)
        h_enc = self.encoder.init_hidden(batch_size)
        mu, sigma = self.encoder(x, h_enc)

        # Sample latent variable
        with torch.no_grad():
            epsilon = torch.randn_like(mu, device=device)

        z = self.z_embedding(mu + sigma * epsilon)
        h_dec = self.decoder.init_hidden(batch_size)
        out = self.decoder(x, z, h_dec, use_teacher_forcing)
        torch.set_printoptions(profile="full")
        #print("Printing Z....")
        #print(z.size())
        #print(z)
        #print("Printing OUT.....")
        #print(out.size())
        #print(out)
        torch.set_printoptions(profile="default")

        return out, mu, sigma, z

    def reconstruct(self, x, temperature):
        batch_size = x.size(1)
        h_enc = self.encoder.init_hidden(batch_size)
        mu, sigma = self.encoder(x, h_enc)
        with torch.no_grad():
            epsilon = torch.randn_like(mu, device=device)
        z = self.z_embedding(mu + sigma * epsilon)
        h_dec = self.decoder.init_hidden(batch_size)
        
        out = self.decoder.reconstruct(z, h_dec, temperature)
        return out
