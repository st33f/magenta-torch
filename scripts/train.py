#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import yaml
import pandas as pd
sys.path.append(".")

from torch.utils.data import DataLoader
import torch.nn as nn

from src.model import *
from src.trainer import *
from src.dataset import MidiDataset
from src.helpers import load_danceability, load_filepaths

from helpers.fake_dataset import generate_fake_songs

import wandb


# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='conf.yml')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument("--use_da", type=bool, default=True)
parser.add_argument("--on_cluster", type=bool, default=False)



# def load_extra_features(path):
#     extra_features_df = pd.read_csv(path, header=0)
#     filenames = extra_features_df['file']
#     danceability = extra_features_df['danceability']
#     energy = extra_features_df['energy']
#     #print('filenames: \n')
#     #print(filenames)
#     return filenames.values, danceability.values




def load_model(model_type, params):
    print(params)
    if model_type == 'lstm':
        model = MusicLSTMVAE(**params)
    elif model_type == 'gru':
        model = MusicGRUVAE(**params)
        print(params)
        if params['use_danceability']:
            model = DanceabilityGRUVAE(**params)
    else:
        raise Exception("Invalid model type. Expected lstm or gru")
    return model


def load_data(train_data, val_data, batch_size, validation_split=0.2, random_seed=874, song_paths=None, danceability=None, use_fake_data=False):
    train_loader = None
    val_loader = None

    # for testing purposes
    if use_fake_data:
        X_train = generate_fake_songs(32, 10)
        train_data = MidiDataset(X_train, song_paths=song_paths)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        X_val = generate_fake_songs(32, 4)
        val_data = MidiDataset(X_val, song_paths=song_paths)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader

    # the real loader
    if train_data != '':
        X_train = pickle.load(open(train_data, 'rb'))
        train_data = MidiDataset(X_train, song_paths=song_paths, danceability=danceability)
        print(' --- train data summary --- ')
        print(len(train_data.danceabilities))
        print(len(song_paths))
        last_batch_index = train_data.index_mapper[-1]
        print(f"last_batch: {last_batch_index}")
        last_song_id = last_batch_index[0]
        print(f"last_song_id: {last_song_id}")

        # print(train_data.song_to_idx)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        print(f'len train loader: {len(train_loader)}')
        print(f"da slice training: {len(train_data.danceabilities[:last_song_id])}")
        print(f"da slice val: {len(train_data.danceabilities[last_song_id:])}")
        danceability = train_data.danceabilities[last_song_id:]
        song_paths = song_paths[last_song_id:]
        # REDEFINE DA WITH SLICE, SO VAL DATA DA DOESNT CONTAIN TRAIN DATA AND STARTS AT 0
    if val_data != '':
        X_val = pickle.load(open(val_data, 'rb'))
        val_data = MidiDataset(X_val, song_paths=song_paths, danceability=danceability)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        print(' --- val data summary --- ')
        print(len(val_data.danceabilities))
        print(len(song_paths))
        print(f'len val loader: {len(val_loader)}')

    return train_loader, val_loader


def train(model, trainer, train_data, val_data, epochs, resume):
    """
    Train a model
    """
    trainer.train(model, train_data, None, epochs, resume, val_data)


def main(args):
    model_params = None
    trainer_params = None
    data_params = None
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
        model_params = config['model']
        trainer_params = config['trainer']
        if args.on_cluster == True:
            data_params = config['das-data']
        else:
            data_params = config['data']

    # init weights and biases
    wandb.init(project="master-thesis", config=config)
    wandb.config.update({"epochs": args.epochs, "batch_size": trainer_params['batch_size']})


    #filepaths, danceability = load_extra_features(args.extra_features)
    #filepaths = load_filepaths()
    #print(danceability)
    filepaths = load_filepaths(data_params['filepaths'])
    danceability = load_danceability(data_params['danceability_paths'])

    train_data, val_data = load_data(data_params['train_data'], 
                                     data_params['val_data'], 
                                     trainer_params['batch_size'],
                                     song_paths=filepaths,
                                     danceability=danceability,
                                     use_fake_data=trainer_params['use_fake_data'])
    # print(f"len train data: {len(train_data)}")
    # print(f"len Val data: {len(val_data)}")
    # print(f"len filepaths: {len(filepaths)}")
    model = load_model(args.model_type, model_params)
    #model = model.to(device)
    #model = torch.nn.DataParallel(model, device_ids=[0,1])
    #print(f"device count: {torch.cuda.device_count()}")
    
    # Watch the model with weights and biases
    wandb.watch(model)

    trainer = Trainer(**trainer_params)

    train(model, trainer, train_data, val_data, args.epochs, args.resume)

    # save model to wandbb
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
