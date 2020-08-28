#import pypianoroll as pr
#from src.reconstruction import *
#from matplotlib import pyplot as plt
#from music21 import *
import torch
import pickle
import numpy as np

def load_filepaths(picklefile):
    filepaths = pickle.load(open(picklefile, 'rb'))
    return filepaths

def load_danceability(picklefile):
    da = pickle.load(open(picklefile, 'rb'))
    return da

def generate_fake_songs(num_batches, num_songs):

    song = []
    class_count = 61

    for number in range(num_batches):
        note_count = 0
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

#pick = pickle.load(open("/Users/stefanwijtsma/code/magenta-torch/pickle_data/clean_midi_1/train_paths.pickle", 'rb'))
#print(pick)
#print(len(pick))

#pick = pickle.load(open("/Users/stefanwijtsma/code/magenta-torch/pickle_data/clean_midi_1/test_paths.pickle", 'rb'))
#print(pick)
#print(len(pick))

# 119547037146038801333356
# 119547037146038801333356
#
# da = load_danceability()
# print(len(da))
#
#
# train_paths = pickle.load(open("/Users/stefanwijtsma/code/magenta-torch/pickles_small/train_paths.pickle", 'rb'))
# print(train_paths)
# print(len(train_paths))
