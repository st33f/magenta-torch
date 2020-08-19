#import pypianoroll as pr
#from src.reconstruction import *
#from matplotlib import pyplot as plt
#from music21 import *
import torch
import pickle

def load_filepaths(picklefile):
    filepaths = pickle.load(open(picklefile, 'rb'))
    return filepaths

def load_danceability(picklefile):
    da = pickle.load(open(picklefile, 'rb'))
    return da


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
