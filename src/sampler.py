from src.checkpoint import Checkpoint

from src.loss import ELBO

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from math import exp
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spherical_interpolation(p0, p1, t):
    omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                            np.squeeze(p1/np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

class Sampler:
    def __init__(self,
                 free_bits=256,
                 output_dir='samples'):
        self.free_bits = free_bits
        self.output_dir = output_dir
        
    def reconstruction_loss(self, model, batch, da=None):
        """
        Return reconstruction loss with and witout teacher forcing
        """
        pred_tf, _, _, _ = model(batch, True, da=da)
        pred, _, _, _ = model(batch, False, da=da)
        loss_tf = torch.nn.functional.binary_cross_entropy_with_logits(pred_tf, batch, reduction='mean')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, batch, reduction='mean')
        return loss_tf, loss
    
    def evaluate(self, model, input_data):
        """
        Evaluate test test data directly using logits
        """
        model.eval()
        loss_acc_tf = 0
        loss_acc = 0
        print(f"input_data: {input_data}")
        with torch.no_grad():
            for idx, batch in enumerate(input_data):
                print(f"batch: {batch[0].size()}")
                data, da = batch
                print(data.size())
                #print(da)
                da.to(device)
                data = data.transpose(0, 1)
                batch_size = data.size(1)
                data = data.view(model.max_sequence_length, batch_size, model.decoder.input_size)
                data.to(device)
                batch_loss_tf, batch_loss = self.reconstruction_loss(model, data, da)
                loss_acc_tf += batch_loss_tf
                loss_acc += batch_loss
                print('idx: %d, loss_tf: %.4f, loss: %.4f' % (idx, batch_loss_tf, batch_loss))
        return loss_acc_tf / len(input_data), loss_acc / len(input_data)
    
    def reconstruct(self, model, song, temperature):
        """
        Reconstruct song
        """
        model.eval()
        with torch.no_grad():
            print(song)
            song = song.transpose(0, 1)
            batch_size = song.size(1)
            song.view(model.max_sequence_length, batch_size, model.decoder.input_size)
            song.to(device)
            sample = model.reconstruct(song, temperature)
            # Samples are currently (seq_len, batch_size, num_notes) where 
            # batch_size is the number of segments of 16 bars. These belong to the 
            # same song so we want to return the concatenation of the entire song
            sample = sample.view(-1, model.input_size)
            return sample
    
#     def interpolate(self, model, song_A, song_B, num_steps,
#                    length=None, temperature=1.0, assert_same_length=True)
#         """
#         Args:
#           model: Trained model
#           start_sequence: The NoteSequence to interpolate from.
#           end_sequence: The NoteSequence to interpolate to.
#           num_steps: Number of NoteSequences to be generated, including the
#             reconstructions of the start and end sequences.
#           length: The maximum length of a sample in decoder iterations. Required
#             if end tokens are not being used.
#           temperature: The softmax temperature to use (if applicable).
#           assert_same_length: Whether to raise an AssertionError if all of the
#             extracted sequences are not the same length.
#         Returns:
#           A list of interpolated NoteSequences.
#         """
#         model.eval()
#         model.to(device)
#         batch_size = 2
        
#         # Load songs
#         input = torch.randn(256, batch_size, 61)
#         reconstructed, _, _, latent = model(input, use_teacher_forcing=True)
#         # Interpolate between latent spaces
#         z_interpolated = np.array([spherical_interpolation(latent[0], latent[1], t)
#                                   for t in np.linspace(0, 1, num_steps)])
#         # Decode interpolations
#         decoded = [model.decoder(step, use_teacher_forcing=True, temperature=temperature)
#                   for step in z_interpolated]
#         # Reconstruct decoded interpolations
    