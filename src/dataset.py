import os

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MidiDataset(Dataset):
    """
    Inputs:
    - song_tensor: input pitches of shape (num_samples, input_length, different_pitches)
    - song_paths: list of paths to input midi files
    
    Attributes:
    - midi_paths: list of paths to midi files
    - song_names: List of song names in dataset
    - song_name_to_idx: Dictionary mapping song name to idx in song_names and midi paths
    - index_mapper: List of tuple (song_idx, bar_idx) for each song
    """
    def __init__(self, input_tensor, song_paths=None, instruments=None, tempos=None, danceability=None):
        self.song_tensor = [x.astype(float) for x in input_tensor]
        self.midi_paths = song_paths
        self.song_names = None
        self.danceabilities = danceability

        if song_paths is not None:
            self.song_names = [os.path.basename(x).split('.')[0] for x in song_paths]
        self.index_mapper, self.song_to_bar_idx = self._initialize()
        if self.song_names is not None:
            #self.song_to_idx = {v:k for (k,v) in enumerate(self.song_names)}
            self.song_to_idx = {v: k for (k, v) in enumerate(self.midi_paths)}
        self.instruments = instruments
        self.tempos = tempos
    
    def _initialize(self):
        index_mapper = []
        song_to_idx = dict()
        
        for song_idx in range(len(self.song_tensor)):
            song_tuples = []
            split_count = self.song_tensor[song_idx].shape[0] 
            for bar_idx in range(0, split_count):
                song_tuples.append((song_idx, bar_idx))
                index_mapper.append((song_idx, bar_idx))
                
            if self.song_names is not None:
                song_name = self.song_names[song_idx]
                song_to_idx[song_name] = song_tuples
        return index_mapper, song_to_idx
        
    def __len__(self):
        return len(self.index_mapper)
        
    def __getitem__(self, idx):
        """
        A sample is B consecutive bars. In MusicVAE this would be 16 consecutive
        bars. For MidiVAE a sample consists of a single bar.
        """
        song_idx, section_idx = self.index_mapper[idx]
        #print(f"IDX: {idx}")
        #print(f"song_idx: {song_idx}")
        sample = self.song_tensor[song_idx]
        #print(f"song name: {self.song_names[song_idx]}")

        # make danceability tensor
        # danceability = torch.tensor(self.danceabilities[idx], dtype=torch.float32)
        if self.danceabilities is not None:
            danceability = torch.tensor(self.danceabilities[song_idx], dtype=torch.float32)
        else:
            danceability = []


        sample = sample[section_idx,:,:]
        x = torch.tensor(sample, dtype=torch.float32)

        # return both x and DA as tuple
        result = (x, danceability)
        # return x.to(device)
        return result
    
    def get_tensor_by_name(self, song_name):
        """
        Return tensor for specified song partitioned by bars
        """
        #print(self.midi_paths)
        song_idx = self.song_to_idx[song_name]
#         song_idx, bar_idx = self.song_to_bar_idx[song_name]
#         print(bar_idx)
        samples = self.song_tensor[song_idx]
        samples = torch.tensor(samples, dtype=torch.float)
        return samples 
    
    def get_aux_by_names(self, song_name):
        """
        Return aux information such as instruments and tempo
        """
        if self.song_to_idx is not None:
            idx = self.song_to_idx[song_name]
            if self.instruments is not None:
                return self.instruments[idx], self.tempos[idx]
        return None
