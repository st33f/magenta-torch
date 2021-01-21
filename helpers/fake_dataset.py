import numpy
import pickle
import torch
import numpy as np
from src.plot import plot_spectogram
'''
train_data = "/Users/stefanwijtsma/code/magenta-torch/pickle_data/clean_midi_2_small/X_train.pickle"
X_train = pickle.load(open(train_data, 'rb'))

train_paths = "/Users/stefanwijtsma/code/magenta-torch/pickle_data/clean_midi_2_small/train_paths.pickle"
paths = pickle.load(open(train_paths, 'rb'))

#print(paths)
print(X_train)
print(len(X_train))
for i, values in enumerate(X_train):
    if i < 1:
        print(len(values))
        print(len(values[0]))
'''

timestep = [0.] * 61

class_count = 61



def generate_fake_songs(num_batches, num_songs):

    song = []
    class_count = 61
    note_count = 0

    for number in range(num_batches):
        section = []
        for i in range(256):
            steplist = [0.] * 61
            if note_count < class_count:
                steplist[note_count] += 1
                section.append(steplist)
                note_count += 1
            else:
                note_count = 0
                steplist[note_count] += 1
                section.append(steplist)
                note_count += 1

        song.append(section)
    return [np.array(song)] * num_songs

# song = generate_fake_songs(38, 172)
#
# print(song)
# plot_spectogram(song, song)
# print()
#
# print(len(song))
# for i, values in enumerate(song):
#     if i < 1:
#         print(len(values))
#         print(len(values[0]))