import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMEncoder(nn.Module):
    """
    Bi-directional LSTM encoder from MusicVAE
    Inputs:
    - input_size: Dimension of one-hot representation of input notes
    - hidden_size: hidden size of bidirectional lstm
    - num_layers: Number of layers for bidirectional lstm
    """
    def __init__(self,
                 input_size=61,
                 hidden_size=2048,
                 latent_size=512,
                 num_layers=2):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.softplus = nn.Softplus()

    def forward(self, input, h0, c0):
        batch_size = input.size(1)
        _, (h_n, c_n) = self.bilstm(input, (h0, c0))
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
        mu = self.mu(h_n)
        sigma = self.softplus(self.sigma(h_n))
        return mu, sigma

    def init_hidden(self, batch_size=1):
        # Bidirectional lstm so num_layers*2
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device), \
               torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device)


class HierarchicalLSTMDecoder(nn.Module):
    """
    Hierarchical decoder from MusicVAE
    """

    def __init__(self,
                 num_embeddings,
                 input_size=61,
                 hidden_size=1024,
                 latent_size=512,
                 num_layers=2,
                 max_seq_length=256,
                 seq_length=16):
        super(HierarchicalLSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings
        self.max_seq_length = max_seq_length
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.tanh = nn.Tanh()
        self.conductor = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.conductor_embeddings = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=latent_size),
            nn.Tanh())
        self.lstm = nn.LSTM(input_size=input_size + latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=input_size),
            nn.Softmax(dim=2)
        )

    def forward(self, target, latent, h0, c0, use_teacher_forcing=True, temperature=1.0):
        batch_size = target.size(1)

        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        # Initialie start note
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)

        # Conductor produces an embedding vector for each subsequence
        for embedding_idx in range(self.num_embeddings):
            embedding, (h0, c0) = self.conductor(latent.unsqueeze(0), (h0, c0))
            embedding = self.conductor_embeddings(embedding)

            # Initialize lower decoder hidden state
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))

            # Decoder produces sequence of distributions over output tokens
            # for each subsequence where at each step the current
            # conductor embedding is concatenated with the previous output
            # token to be used as input
            if use_teacher_forcing:
                embedding = embedding.expand(self.seq_length, batch_size, embedding.size(2))
                idx = range(embedding_idx * self.seq_length, embedding_idx * self.seq_length + self.seq_length)
                e = torch.cat((target[idx, :, :], embedding), dim=2).to(device)
                prev_note, h0_dec = self.lstm(e, h0_dec)
                prev_note = self.out(prev_note)
                out[idx, :, :] = prev_note
                prev_note = prev_note[-1, ::].unsqueeze(0)
            else:
                for note_idx in range(self.seq_length):
                    e = torch.cat((prev_note, embedding), -1)
                    prev_note, h0_dec = self.lstm(e, h0_dec)
                    prev_note = self.out(prev_note)

                    idx = embedding_idx * self.seq_length + note_idx
                    out[idx, :, :] = prev_note.squeeze()
        return out
    
    def reconstruct(self, latent, h0, c0, temperature):
        """
        Reconstruct the actual midi using categorical distribution
        """
        one_hot = torch.eye(self.input_size).to(device)
        batch_size = 1
        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)
        for embedding_idx in range(self.num_embeddings):
            embedding, (h0, c0) = self.conductor(latent.unsqueeze(0), (h0, c0))
            embedding = self.conductor_embeddings(embedding)
            h0_dec = (torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                      torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))
            for note_idx in range(self.seq_length):
                print("Prev Note = {prev_note.size()}")
                print("Embedding = {embedding.size()}")
                e = torch.cat((prev_note, embedding), -1)
                prev_note, h0_dec = self.lstm(e, h0_dec)
                prev_note = self.out(prev_note)
                prev_note = Categorical(prev_note / temperature).sample()
                prev_note = self.one_hot(prev_note)
                out[idx, :, :] = prev_note.squeeze()
        return out
                

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device))
        #return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)


class Danceability_BiGRUEncoder(nn.Module):
    """
    Bi-directional GRU encoder from MusicVAE
    Inputs:
    - input_size:
    - hidden_size: hidden size of bidirectional gru
    - num_layers: Number of layers for bidirectional gru
    """

    def __init__(self,
                 input_size=61,
                 hidden_size=2048,
                 latent_size=512,
                 num_layers=2):
        super(Danceability_BiGRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size + 1, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size + 1, out_features=latent_size)
        self.softplus = nn.Softplus()

    def forward(self, input, h0, da=None):
        print(f"input:  {input.size()}")
        #input = input.to(device)
        #print(f" This is my data: {da.size()}")
        #print(da)
        #data, _danceability = input
        #print(f'datashape: {data.size()}')
        batch_size = input.size(1)

        # print(f"batch size layers.py: {batch_size}")
        # print(f"dance before: {da.size()}")
        if da is not None:
            try:
                danceability = da.view(batch_size, 1)
                danceability = danceability.to(device)
                # print(f"dance after: {danceability.size()}")
                print(f"\n  Danceability size: {danceability.size()}")
                print(f"  Danceability: {danceability}")
            except:
                danceability = torch.zeros(batch_size, 1)
                print(f"ERROR da: {da}")


        _, h_n = self.bigru(input, h0)
        # print(f"___ = {_.size()}")
        # print(h_n.size())
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
        # print(f"h_n: {h_n.size()}")
        # print(h_n[:, -1])

        # start comcatenation of danceability scores
        if da is not None:
            new_h_n = torch.cat((h_n, danceability), 1)
            # print(f" New h_n size: {new_h_n.size()}")
            # print(f" New h_n da dimension: {new_h_n[:, -1]}")
            mu = self.mu(new_h_n)
            sigma = self.softplus(self.sigma(new_h_n))
            # print(f" Mu size: {mu.size()}")
            # print(self.mu.weight.data.size())
            # print(len(self.mu.weight.data[:,-1]))
            print()
            print("Mean weight for danceability:")
            print(self.mu.weight.data[:,-1].abs().mean())
            print("all weights to MU")
            print(self.mu.weight.data.abs().mean(dim=0))
            print(self.mu.weight.data.abs().mean())
            print()

            #print(sigma.size())
        else:
            mu = self.mu(h_n)
            sigma = self.softplus(self.sigma(h_n))

        return mu, sigma

    def init_hidden(self, batch_size=1):
        # Bidirectional gru so num_layers*2
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device)



class MuDanceVAE_Encoder(nn.Module):
    """
    Bi-directional GRU encoder from MusicVAE
    Inputs:
    - input_size:
    - hidden_size: hidden size of bidirectional gru
    - num_layers: Number of layers for bidirectional gru
    """

    def __init__(self,
                 input_size=61,
                 hidden_size=2048,
                 latent_size=512,
                 num_layers=2):
        super(MuDanceVAE_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.softplus = nn.Softplus()

    def forward(self, input, h0, da=None):
        print(f"input:  {input.size()}")
        batch_size = input.size(1)

        if da is not None:
            try:
                danceability = da.view(batch_size, 1)
                danceability = danceability.to(device)
                print(f"\n  Danceability size: {danceability.size()}")
                print(f"  Danceability: {danceability}")
            except:
                danceability = torch.zeros(batch_size, 1)
                print(f"DANCEABILITY ERROR\n DA: {da}")


        _, h_n = self.bigru(input, h0)
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)


        # start comcatenation of danceability scores to MU
        if da is not None:
            mu = self.mu(h_n)
            new_mu = torch.cat((mu, danceability), 1)
            print(f" New mu size: {new_mu.size()}")
            print(f" New mu da dimension: {new_mu[:, -1]}")


            sigma = self.softplus(self.sigma(h_n))
            new_sigma = torch.cat((sigma, torch.ones([batch_size, 1], device=device)), 1)
            print(f" New sigma size: {new_sigma.size()}")
            print(f" New sigma da dimension: {new_sigma[:, -1]}")
        else:
            new_mu = self.mu(h_n)
            new_sigma = self.softplus(self.sigma(h_n))

        return new_mu, new_sigma

    def init_hidden(self, batch_size=1):
        # Bidirectional gru so num_layers*2
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device)


class HierarchicalGRUDecoder(nn.Module):
    """
    Hierarchical decoder from MusicVAE
    """

    def __init__(self,
                 num_embeddings,
                 input_size=61,
                 hidden_size=1024,
                 latent_size=512,
                 num_layers=2,
                 max_seq_length=256,
                 seq_length=16):
        super(HierarchicalGRUDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings
        self.max_seq_length = max_seq_length
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.tanh = nn.Tanh()
        self.conductor = nn.GRU(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.conductor_embeddings = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=latent_size),
            nn.Tanh())
        self.gru = nn.GRU(input_size=input_size + latent_size, hidden_size=hidden_size, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=input_size),
            nn.Softmax(dim=2)
        )

    def forward(self, target, latent, h0, use_teacher_forcing=False, temperature=1.0):
        batch_size = target.size(1)
        target = target.to(device)

        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        # Initialie start note
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)

        # Conductor produces an embedding vector for each subsequence, where each
        # subsequence consists of a bar of 16th notes
        for embedding_idx in range(self.num_embeddings):
            embedding, h0 = self.conductor(latent.unsqueeze(0), h0)
            embedding = self.conductor_embeddings(embedding)

            # Initialize lower decoder hidden state
            h0_dec = torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)

            # Decoder produces sequence of distributions over output tokens
            # for each subsequence where at each step the current
            # conductor embedding is concatenated with the previous output
            # token to be used as input
            if use_teacher_forcing:
                embedding = embedding.expand(self.seq_length, batch_size, embedding.size(2)).to(device)
                idx = range(embedding_idx * self.seq_length, embedding_idx * self.seq_length + self.seq_length)
                e = torch.cat((target[idx, :, :], embedding), dim=2)
                prev_note, h0_dec = self.gru(e, h0_dec)
                prev_note = self.out(prev_note)
                out[idx, :, :] = prev_note
                prev_note = prev_note[-1, :, :].unsqueeze(0)
            else:
                for note_idx in range(self.seq_length):
                    e = torch.cat((prev_note, embedding), -1)
                    prev_note, h0_dec = self.gru(e, h0_dec)
                    prev_note = self.out(prev_note)

                    idx = embedding_idx * self.seq_length + note_idx
                    out[idx, :, :] = prev_note.squeeze()
        return out
    
    def reconstruct(self, latent, h0, temperature):
        """
        Reconstruct the actual midi using categorical distribution
        """
        one_hot = torch.eye(self.input_size).to(device)
        batch_size = h0.size(1)
        out = torch.zeros(self.max_seq_length, batch_size, self.input_size, dtype=torch.float, device=device)
        prev_note = torch.zeros(1, batch_size, self.input_size, dtype=torch.float, device=device)
        for embedding_idx in range(self.num_embeddings):
            embedding, h0 = self.conductor(latent.unsqueeze(0), h0)
            embedding = self.conductor_embeddings(embedding)
            h0_dec = torch.randn(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)
            for note_idx in range(self.seq_length):
                e = torch.cat((prev_note, embedding), -1)
                e = e.to(device)
                prev_note, h0_dec = self.gru(e, h0_dec)
                prev_note = self.out(prev_note)
                prev_note = prev_note.to(device)
                prev_note = Categorical(prev_note / temperature).sample()
                prev_note = one_hot[prev_note]
                out[note_idx, :, :] = prev_note.squeeze()
        return out

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=device)



class BiGRUEncoder(nn.Module):
    """
    Bi-directional GRU encoder from MusicVAE
    Inputs:
    - input_size:
    - hidden_size: hidden size of bidirectional gru
    - num_layers: Number of layers for bidirectional gru
    """

    def __init__(self,
                 input_size=61,
                 hidden_size=2048,
                 latent_size=512,
                 num_layers=2):
        super(BiGRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
        self.softplus = nn.Softplus()

    def forward(self, input, h0):
        batch_size = input.size(1)
        _, h_n = self.bigru(input, h0)
        h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
        print("printing h_n:.......")
        print(h_n)
        #wandb.log({"h_n Hidden layer weights": wandb.Histogram(h_n.cpu().detach().numpy())})
        mu = self.mu(h_n)
        sigma = self.softplus(self.sigma(h_n))
        #torch.set_printoptions(profile="full")
        #print(h_n)
        #print(sigma)
        #torch.set_printoptions(profile="default")
        return mu, sigma

    def init_hidden(self, batch_size=1):
        # Bidirectional gru so num_layers*2
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device)

class BiGRU_fixed_sigma_Encoder(nn.Module):
        """
        Bi-directional GRU encoder from MusicVAE
        Inputs:
        - input_size:
        - hidden_size: hidden size of bidirectional gru
        - num_layers: Number of layers for bidirectional gru
        """

        def __init__(self,
                     input_size=61,
                     hidden_size=2048,
                     latent_size=512,
                     num_layers=2):
            super(BiGRU_fixed_sigma_Encoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.latent_size = latent_size
            self.num_layers = num_layers

            self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=True)
            self.mu = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
            self.sigma = nn.Linear(in_features=2 * hidden_size, out_features=latent_size)
            self.softplus = nn.Softplus()

        def forward(self, input, h0):
            batch_size = input.size(1)
            print("----- X size layers.py ---", input.size())
            _, h_n = self.bigru(input, h0)
            h_n = h_n.view(self.num_layers, 2, batch_size, -1)[-1].view(batch_size, -1)
            print("printing h_n:.......")
            print(h_n)
            #wandb.log({"h_n Hidden layer weights": wandb.Histogram(h_n.cpu().detach().numpy())})
            mu = self.mu(h_n)
            # sigma = self.softplus(self.sigma(h_n))
            sigma = torch.tensor([1], dtype=torch.float, device=device)
            # torch.set_printoptions(profile="full")
            # print(h_n)
            # print(sigma)
            # torch.set_printoptions(profile="default")
            return mu, sigma

        def init_hidden(self, batch_size=1):
            # Bidirectional gru so num_layers*2
            return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float, device=device)
