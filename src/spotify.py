import glob
import sys
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fuzzywuzzy import fuzz
import pandas as pd

#sys.setrecursionlimit(100000)
sys.setrecursionlimit(10**9)

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def match_artist(query, artist):
    ratio = fuzz.token_set_ratio(query, artist)
    if ratio > 50:
        return True
    else:
        return False


def execute_query(trackname, artist):
    try:
        response = sp.search(q=trackname, type='track', limit=50)['tracks']
    except:
        return False
    tracks = response['items']
    found_match = False
    for i in tracks:
        artists = i['artists']
        track_id = i['id']
        for j in artists:
            if match_artist(artist, j['name']):
                found_match = True
                print(f"Track ID found for {i['name']}  -  {artist}")
                return track_id
    if not found_match:
        print(f"No ID found for {trackname} - {artist}")
        return False

def get_filenames(dir):
    filenames = []
    for file in glob.glob(f'{dir}/*/*'):
        filenames.append(file)
    return filenames

def make_query(filepath, dir):
    filename = filepath.replace(f'{dir}/', '')
    artist = filename.split('/')[0]
    track = filename.split('/')[1].split('.')[0]
    #print(artist, '\n', track)
    return artist, track

def get_features(track_id):
    response = sp.audio_features(tracks=[track_id])
    items = response[0]
    if items:
        danceability = items['danceability']
        mode = items['mode']
        energy = items['energy']
        return danceability, mode, energy
    else:
        return False


def main():
    #files = get_filenames("/Users/stefanwijtsma/code/mt/data/clean_midi_1")
    dir = "/Users/stefanwijtsma/code/mt/data/very_small_dataset"
    #dir = "/Users/stefanwijtsma/code/mt/data/clean_midi"
    files = get_filenames(dir)

    files_processed = 0
    matches = 0
    track_ids = []
    scores_tuples = []


    for i in files:
        files_processed += 1
        artist, track = make_query(i, dir)
        track_id = execute_query(track, artist)
        if track_id:
            matches += 1
            track_ids.append((i, track_id))

    for track in track_ids:
        filepath, track_id = track[0], track[1]
        #print(track_id)
        try:
            danceability, mode, energy = get_features(track_id)
        except:
            continue
        if danceability:
            scores_tuples.append((filepath, danceability, mode, energy))

    scores_df = pd.DataFrame(scores_tuples, columns=['file', 'danceability', 'mood', 'energy'])
    print(scores_df.head())
    scores_df.to_csv('/Users/stefanwijtsma/code/mt/data/scores_clean_midi.csv')

    #print(track_ids)
    #print(scores_tuples)
    print(f"Files Processed: {files_processed}")
    print(f"Matches: {matches}")
    print(f"Percentage matched: {(matches / files_processed) * 100} %")

#main()
query = "circles and squares"
artist = "Phuture noize"

#response = execute_query(query, artist)

