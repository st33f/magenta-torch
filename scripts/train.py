#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
sys.path.append(".")



from src.model import *
from src.trainer import *
from src.dataset import MidiDataset
from src.helpers import load_danceability, load_filepaths

from helpers.fake_dataset import generate_fake_songs

import wandb

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='conf.yml')
parser.add_argument('--model_type', type=str, default='lstm')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument("--use_da", type=bool, default=True)
parser.add_argument("--on_cluster", type=str2bool, nargs='?',
                        const=True, default=False)



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
    elif model_type == 'fixed-sigma':
        model = Fixed_sigma_MusicGRUVAE(**params)
    elif model_type == 'mudance':
        model = MuDanceVAE(**params)
    elif model_type == 'zdance':
        model = ZDanceVAE(**params)
    elif model_type == 'gru':
        model = MusicGRUVAE(**params)
        if params['use_danceability']:
            model = DanceabilityGRUVAE(**params)
    else:
        raise Exception("Invalid model type. Expected lstm, gru or fixed-sigma")
    return model

def unpickle(p):
    data = pickle.load(open(p, 'rb'))
    return data


def load_data(train_data, val_data, batch_size, validation_split=0.2, random_seed=874, train_paths=None, val_paths=None, train_ef=None, val_ef=None, use_fake_data=False):
    train_paths = unpickle(train_paths)
    val_paths = unpickle(val_paths)
    train_ef = unpickle(train_ef)
    val_ef = unpickle(val_ef)

    train_loader = None
    val_loader = None

    # for testing purposes
    if use_fake_data:
        X_train = generate_fake_songs(32, 100)
        train_data = MidiDataset(X_train, song_paths=train_paths)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        X_val = generate_fake_songs(32, 10)
        val_data = MidiDataset(X_val, song_paths=val_paths)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader


    # the real loader
    if train_data != '':
        X_train = pickle.load(open(train_data, 'rb'))
        train_data = MidiDataset(X_train, song_paths=train_paths, danceability=train_ef)
        print(' --- train data summary --- ')
        print(f"len danceability {len(train_data.danceabilities)}")
        print(f"Len song paths: {len(train_paths)}")
        #last_batch_index = train_data.index_mapper[-1]
        #print(f"last_batch: {last_batch_index}")
        #last_song_id = last_batch_index[0]
        #print(f"last_song_id: {last_song_id}")

        # print(train_data.song_to_idx)
        train_data_subset = Subset(train_data, list(range(7000)))
        train_loader = DataLoader(train_data_subset, batch_size=batch_size, shuffle=False)
        print(f"Len train loader: {len(train_loader)}")
        #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        # print(f'len train loader: {len(train_loader)}')
        # print(f"da slice training: {len(train_data.danceabilities[:last_song_id])}")
        # print(f"da slice val: {len(train_data.danceabilities[last_song_id:])}")
        # danceability = train_data.danceabilities[last_song_id:]
        # song_paths = song_paths[last_song_id:]
        # REDEFINE DA WITH SLICE, SO VAL DATA DA DOESNT CONTAIN TRAIN DATA AND STARTS AT 0
    if val_data != '':
        X_val = pickle.load(open(val_data, 'rb'))
        val_data = MidiDataset(X_val, song_paths=val_paths, danceability=val_ef)
        val_data_subset = Subset(train_data, list(range(3000)))
        val_loader = DataLoader(val_data_subset, batch_size=batch_size, shuffle=False)
        #val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        print(' --- val data summary --- ')
        print(len(val_data.danceabilities))
        print(len(val_paths))
        print(f'len val loader: {len(val_loader)}')

    return train_loader, val_loader


def train(model, trainer, train_data, val_data, epochs, resume):
    """
    Train a model
    """
    trainer.train(model, train_data, None, epochs, resume, val_data)


def main(args):

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_params = None
    trainer_params = None
    data_params = None
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)
        model_params = config['model']
        trainer_params = config['trainer']
        print(f"On_cluster: {args.on_cluster}")
        if args.on_cluster == True:
            data_params = config['das-data']
        else:
            data_params = config['data']

    # init weights and biases
    wandb.init(project="master-thesis", config=config)
    wandb.config.update({"epochs": args.epochs, "batch_size": trainer_params['batch_size']})
    wandb.run.summary["On_cluster"] = args.on_cluster
    run_name = wandb.run.id

    #filepaths, danceability = load_extra_features(args.extra_features)
    #filepaths = load_filepaths()
    #print(danceability)
    #filepaths = load_filepaths(data_params['filepaths'])
    # danceability = load_danceability(data_params['danceability_paths'])

    train_data, val_data = load_data(data_params['train_data'],
                                     data_params['val_data'], 
                                     trainer_params['batch_size'],
                                     train_paths=data_params['train_song_paths'],
                                     val_paths=data_params['val_song_paths'],
                                     train_ef=data_params['train_extra_features'],
                                     val_ef=data_params['val_extra_features'],
                                     use_fake_data=trainer_params['use_fake_data'])
    # print(f"len train data: {len(train_data)}")
    # print(f"len Val data: {len(val_data)}")
    # print(f"len filepaths: {len(filepaths)}")
    model = load_model(args.model_type, model_params)
    print("Memory after model load \n", torch.cuda.memory_allocated())
    print(torch.cuda.memory_cached())


    #model = model.to(device)
    #model = torch.nn.DataParallel(model, device_ids=[0,1])
    #print(f"device count: {torch.cuda.device_count()}")
    
    # Watch the model with weights and biases
    wandb.watch(model)

    trainer = Trainer(**trainer_params, run_name=run_name)

    train(model, trainer, train_data, val_data, args.epochs, args.resume)

    # save model to wandb
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
