import pretty_midi as pretty_midi
import src.midi_functions as mf
import os
import sys
import numpy as np
import pickle
import math
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from pretty_midi import note_number_to_name

np.set_printoptions(threshold=1000)#np.inf)

t = str(int(round(time.time())))

class MidiPreprocessor:
    """
    Modifying MidiVAE (Brunner et al) preprocessing to remove global variables
    
    Class to proprocess midi files
    - low_crop: Low note cutoff
    - high_crop: high crop cutoff
    - num_notes: Number of midi notes represented 
    - smallest_note: Smallest note representation (i.e. 16th note)
    - max_velocity: Midi velocity is represented in range (0, 127)
    """
    def __init__(self, 
                 classes,
                 pickle_store_folder,
                 include_unknown=False,
                 only_unknown=False,
                 low_crop=24, 
                 high_crop=84, 
                 num_notes=128, 
                 smallest_note=16, 
                 max_velocity=127,
                 include_only_monophonic_instruments=False,
                 max_voices_per_track=1,
                 max_voices=4,
                 include_silent_note=True,
                 velocity_threshold=0.5,
                 instrument_attach_method='1hot-category',
                 attach_instruments=False,
                 input_length=16,
                 output_length=16,
                 test_fraction=0.1,
                 include_held_note=True):
        self.classes = classes
        self.pickle_store_folder = pickle_store_folder
        self.include_unknown = include_unknown
        self.only_unknown=only_unknown
        self.low_crop = low_crop
        self.high_crop = high_crop
        self.num_notes = num_notes
        self.smallest_note = smallest_note
        self.max_velocity = max_velocity
        self.note_columns = [pretty_midi.note_number_to_name(n) for n in range(0, num_notes)]
        self.include_only_monophonic_instruments = include_only_monophonic_instruments
        self.max_voices_per_track = max_voices_per_track
        self.max_voices = max_voices
        self.velocity_threshold = velocity_threshold
        self.instrument_attach_method = instrument_attach_method
        self.attach_instruments = attach_instruments
        self.input_length = input_length
        self.output_length = output_length
        self.test_fraction = test_fraction
        self.include_held_note = include_held_note
        
        if include_unknown:
            self.num_classes = len(classes) + 1
        else:
            self.num_classes = len(classes)
            
        self.include_silent_note = include_silent_note
        if include_silent_note:
            self.silent_dim = 1
        else:
            self.silent_dim = 0
            
        if instrument_attach_method == '1hot-category':
            self.instrument_dim = 16
        elif instrument_attach_method == 'khot-category':
            self.instrument_dim = 4
        elif instrument_attach_method == '1hot-instrument':
            self.instrument_dim = 128
        elif instrument_attach_method == 'khot-instrument':
            self.instrument_dim = 7


    def drop_empty_seqs(self, X, Y, V, D, silent_threshold=16):
        # print(X.shape)
        # print(Y.shape)
        # print(V.shape)
        # print(D.shape)
        to_drop = []
        for k, section in enumerate(X):
            section_silents = 0
            # print(section.shape)
            # print(k)
            for tick in section:
                if tick[-1] == 1:
                    section_silents += tick[-1]
                    if section_silents > silent_threshold:
                        to_drop.append(k)
                        break
                else:
                    section_silents = 0
                # if tick[-1] == 1:
                #     print(tick)

        # print(to_drop)
        clean_X = np.delete(X, to_drop, axis=0)

        clean_Y = np.delete(Y, to_drop, axis=0)
        clean_V = np.delete(V, to_drop, axis=0)
        clean_D = np.delete(D, to_drop, axis=0)

        # print(clean_X.shape)
        # print(clean_Y.shape)
        # print(clean_V.shape)
        # print(clean_D.shape)
        return clean_X, clean_Y, clean_V, clean_D, len(to_drop)

    def plot_dataset_metrics(self, data):
        # print(data)
        note_counts = np.zeros(data[0][0].shape[-1])
        silent_lengths = np.zeros(data[0][0].shape[0])

        print(data[0].shape)
        print(data[0].shape[2])

        for song in data:
            for example in song:
                # count note frequencies
                example_sum = np.sum(example, axis=0)
                note_counts += example_sum
                # count the silence
                c = 0
                for i in example[:, -1]:
                    if i == 1:
                        c += 1
                    else:
                        if c != 0:
                            silent_lengths[c - 1] += 1
                            c = 0

        note_counts = note_counts / sum(note_counts) * 100

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        # freq, bins = np.histogram(note_counts, bins=data[0][0].shape[-1])
        ax.bar(range(data[0].shape[2]), note_counts)
        ax.set_xticks(np.arange(0, data[0].shape[2]))
        title = "x"
        ax.set_title(f"Note frequencies for dataset: {self.pickle_store_folder.split('/')[0]}")
        tick_range_x_min = self.low_crop
        tick_range_x_max = self.high_crop
        if self.include_silent_note:
            tick_range_x_max += 1
        if self.include_held_note:
            tick_range_x_max += 1

        ax.set_xticklabels([note_number_to_name(i) for i in range(tick_range_x_min, tick_range_x_max)], fontdict={"fontsize": 10})
        # ax.set_xticklabels(["C{}".format(i - 2) for i in range(11)])

        fname = self.pickle_store_folder + '/note-freqs.png'
        plt.savefig(fname)
        # plt.show()
        plt.close()

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        # freq, bins = np.histogram(note_counts, bins=data[0][0].shape[-1])
        ax.bar(range(128), silent_lengths[:128])
        ax.set_xticks(np.arange(1, 129))
        ax.set_title('Consecutive silence in ticks')
        fname = self.pickle_store_folder + '/silence.png'
        plt.savefig(fname)
        # plt.show()
        plt.close()



    def load_rolls(self, path, name, save_preprocessed_midi):

        #try loading the midi file
        #if it fails, return all None objects
        try:
            mid = pretty_midi.PrettyMIDI(path + name)
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
            exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
            print(exception_str)
            return None, None, None, None, None, None

        #determine start and end of the song
        #if there are tempo changes in the song, only take the longest part where the tempo is steady
        #this cuts of silent starts and extended ends
        #this also makes sure that the start of the bars are aligned through the song
        tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
        song_start = 0
        song_end = mid.get_end_time()
        #there will always be at least one tempo change to set the first tempo
        #but if there are more than one tempo changes, that means that the tempos are changed
        if len(tempo_change_times) > 1:
            longest_part = 0
            longest_part_start_time = 0
            longest_part_end_time = song_end
            longest_part_tempo = 0
            for i, tempo_change_time in enumerate(tempo_change_times):
                if i == len(tempo_change_times) - 1:
                    end_time = song_end
                else:
                    end_time = tempo_change_times[i+1]
                current_part_length = end_time - tempo_change_time
                if current_part_length > longest_part:
                    longest_part = current_part_length
                    longest_part_start_time = tempo_change_time
                    longest_part_end_time = end_time
                    longest_part_tempo = tempo_change_bpm[i]
            song_start = longest_part_start_time
            song_end = longest_part_end_time
            tempo = longest_part_tempo
        else:
            tempo = tempo_change_bpm[0]

        # exclude the drum tracks by redefining mid.instruments
        mid.instruments = [i for i in mid.instruments if i.is_drum == False]


        #cut off the notes that are not in the longest part where the tempo is steady
        for instrument in mid.instruments:
            # exclude drum tracks
            if instrument.is_drum == True:
                print("We have a drum track")
                continue
            new_notes = [] #list for the notes that survive the cutting
            for note in instrument.notes:
                #check if it is in the given range of the longest part where the tempo is steady
                if note.start >= song_start and note.end <= song_end:
                    #adjust to new times
                    note.start -= song_start
                    note.end -= song_start
                    new_notes.append(note)
            instrument.notes = new_notes

        #(descending) order the piano_rolls according to the number of notes per track
        number_of_notes = []
        piano_rolls = [i.get_piano_roll(fs=100) for i in mid.instruments if i.is_drum is False]
        for piano_roll in piano_rolls:
            number_of_notes.append(np.count_nonzero(piano_roll))
        permutation = np.argsort(number_of_notes)[::-1]
        # print(permutation)

        mid.instruments = [mid.instruments[i] for i in permutation]


        quarter_note_length = 1. / (tempo/60.)
        #fs is is the frequency for the song at what rate notes are picked
        #the song will by sampled by (0, song_length_in_seconds, 1./fs)
        #fs should be the inverse of the length of the note, that is to be sampled
        #the value should be in beats per seconds, where beats can be quarter notes or whatever...
        fs = 1. / (quarter_note_length * 4. / self.smallest_note)

        total_ticks = math.ceil(song_end * fs)

        #assemble piano_rolls, velocity_rolls and held_note_rolls
        piano_rolls = []
        velocity_rolls = []
        held_note_rolls = []
        max_concurrent_notes_per_track_list = []
        for instrument in mid.instruments:
            piano_roll = np.zeros((total_ticks, 128))

            pr_start_times = np.zeros((total_ticks, 128))

            #counts how many notes are played at maximum for this instrument at any given tick
            #this is used to determine the depth of the velocity_roll and held_note_roll
            concurrent_notes_count = np.zeros((total_ticks,))

            #keys is a tuple of the form (tick_start_of_the_note, pitch)
            #this uniquely identifies a note since there can be no two notes
            # playing on the same pitch for the same instrument
            note_to_velocity_dict = dict()

            #keys is a tuple of the form (tick_start_of_the_note, pitch)
            #this uniquely identifies a note since there can be no two notes playing
            # on the same pitch for the same instrument
            note_to_duration_dict = dict()

            # this part is basically quantization
            for note in instrument.notes:
                note_tick_start = note.start * fs
                note_tick_end = note.end * fs
                absolute_start = int(round(note_tick_start))
                absolute_end = int(round(note_tick_end))
                print("absolute note times ---")
                print(absolute_start, absolute_end)

                decimal = note_tick_start - absolute_start
                #see if it starts at a tick or not
                #if it doesn't start at a tick (decimal > 10e-3) but is longer than one tick, include it anyways
                if decimal < 10e-3 or absolute_end-absolute_start >= 1:
                    piano_roll[absolute_start:absolute_end, note.pitch] = 1
                    if absolute_start < total_ticks:
                        pr_start_times[absolute_start, note.pitch] = 1
                    concurrent_notes_count[absolute_start:absolute_end] += 1

                    #save information of velocity and duration for later use
                    #this can not be done right now because there might be no ordering in the notes
                    note_to_velocity_dict[(absolute_start, note.pitch)] = note.velocity
                    note_to_duration_dict[(absolute_start, note.pitch)] = absolute_end - absolute_start

            max_concurrent_notes = int(np.max(concurrent_notes_count))
            max_concurrent_notes_per_track_list.append(max_concurrent_notes)

            velocity_roll = np.zeros((total_ticks, max_concurrent_notes))
            held_note_roll = np.zeros((total_ticks, max_concurrent_notes))

            # print(f"len piano_roll: {len(piano_roll)}")
            # print(piano_roll.shape)
            # print(piano_roll)

            for step, note_vector in enumerate(piano_roll):
                pitches = list(note_vector.nonzero()[0])
                sorted_pitches_from_highest_to_lowest = sorted(pitches)[::-1]
                for voice_number, pitch in enumerate(sorted_pitches_from_highest_to_lowest):
                    if (step, pitch) in note_to_velocity_dict.keys():
                        velocity_roll[step, voice_number] = note_to_velocity_dict[(step, pitch)]
                    if (step, pitch) not in note_to_duration_dict.keys():
                        #if the note is in the dictionary, it means that it is the start of the note
                        #if its not the start of a note, it means it is held
                        held_note_roll[step, voice_number] = 1



            if self.include_held_note:
                piano_rolls.append(pr_start_times)
            else:
                piano_rolls.append(piano_roll)
            # piano_rolls.append(piano_roll)
            velocity_rolls.append(velocity_roll)
            held_note_rolls.append(held_note_roll)

        # count which one hast most held notes

        #get the program numbers for each instrument
        #program numbers are between 0 and 127 and have a 1:1 mapping to the instruments described in settings file
        programs = [i.program for i in mid.instruments]

        # we may want to override the maximal_number_of_voices_per_track
        # if the following tracks are all silent it makes no sense to exclude
        # voices from the first instrument and then just have a song with 1 voice
        override_max_notes_per_track_list = [self.max_voices_per_track
                                             for _ in max_concurrent_notes_per_track_list]
        silent_tracks_if_we_dont_override = self.max_voices - \
        sum([min(self.max_voices_per_track, x) if x > 0 else 0
             for x in max_concurrent_notes_per_track_list[:self.max_voices]])

        for voice in range(min(self.max_voices, len(max_concurrent_notes_per_track_list))):
            if silent_tracks_if_we_dont_override > 0 and \
                max_concurrent_notes_per_track_list[voice] > self.max_voices:
                additional_voices = min(silent_tracks_if_we_dont_override,
                                        max_concurrent_notes_per_track_list[voice] - \
                                        self.max_voices)
                override_max_notes_per_track_list[voice] += additional_voices
                silent_tracks_if_we_dont_override -= additional_voices

        #chose the most important piano_rolls
        #each of them will be monophonic
        chosen_piano_rolls = []
        chosen_velocity_rolls = []
        chosen_held_note_rolls = []
        chosen_programs = []
        max_song_length = 0

        #go through all pianorolls in the descending order of the total notes they have
        for batch in zip(piano_rolls,
                         velocity_rolls,
                         held_note_rolls,
                         programs,
                         max_concurrent_notes_per_track_list,
                         override_max_notes_per_track_list):
            piano_roll = batch[0]
            velocity_roll = batch[1]
            held_note_roll = batch[2]
            program = batch[3]
            max_concurrent_notes = batch[4]
            override_max_notes_per_track = batch[5]
            #see if there is actually a note played in that pianoroll
            if max_concurrent_notes > 0:

                #skip if you only want monophonic instruments and there are more than 1 notes played at the same time
                if self.include_only_monophonic_instruments:
                    if max_concurrent_notes > 1:
                        continue
                    monophonic_piano_roll = piano_roll
                    #append them to the chosen ones
                    if len(chosen_piano_rolls) < self.max_voices:
                        chosen_piano_rolls.append(monophonic_piano_roll)
                        chosen_velocity_rolls.append(velocity_roll)
                        chosen_held_note_rolls.append(held_note_roll)
                        chosen_programs.append(program)
                        if monophonic_piano_roll.shape[0] > max_song_length:
                            max_song_length = monophonic_piano_roll.shape[0]
                    else:
                        break

                else:
                    #limit the number of voices per track by the minimum of the actual
                    # concurrent voices per track or the maximal allowed in the settings file
                    for voice in range(min(max_concurrent_notes, max(self.max_voices_per_track,
                                                                     override_max_notes_per_track))):
                        #Take the highest note for voice 0, second highest for voice 1 and so on...
                        monophonic_piano_roll = np.zeros(piano_roll.shape)
                        for step in range(piano_roll.shape[0]):
                            #sort all the notes from highest to lowest
                            notes = np.nonzero(piano_roll[step,:])[0][::-1]
                            if len(notes) > voice:
                                monophonic_piano_roll[step, notes[voice]] = 1

                        #append them to the chosen ones
                        if len(chosen_piano_rolls) < self.max_voices:
                            chosen_piano_rolls.append(monophonic_piano_roll)
                            # print("monophonic PR")
                            # print(monophonic_piano_roll.shape)
                            # print(sum(monophonic_piano_roll[0,:]))

                            chosen_velocity_rolls.append(velocity_roll[:, voice])
                            chosen_held_note_rolls.append(held_note_roll[:, voice])
                            chosen_programs.append(program)
                            if monophonic_piano_roll.shape[0] > max_song_length:
                                max_song_length = monophonic_piano_roll.shape[0]
                        else:
                            break
                    if len(chosen_piano_rolls) == self.max_voices:
                        break

        assert(len(chosen_piano_rolls) == len(chosen_velocity_rolls))
        assert(len(chosen_piano_rolls) == len(chosen_held_note_rolls))
        assert(len(chosen_piano_rolls) == len(chosen_programs))

        if len(chosen_piano_rolls) > 0:
            print("Held notes ----")
            print(len(chosen_held_note_rolls))
            print(chosen_held_note_rolls[0].shape)
            #print(chosen_held_note_rolls[0])

            print("Piano rolls ----")
            print(len(chosen_piano_rolls))
            print(chosen_piano_rolls[0].shape)
            # print(chosen_piano_rolls[0])
            #print(piano_rolls[0])

        #do the unrolling and prepare for model input
        if len(chosen_piano_rolls) > 0:

            song_length = max_song_length * self.max_voices

            #prepare Y
            #Y will be the target notes
            Y = np.zeros((song_length, chosen_piano_rolls[0].shape[1]))
            #unroll the pianoroll into one matrix
            for i, piano_roll in enumerate(chosen_piano_rolls):
                for step in range(piano_roll.shape[0]):
                    Y[i + step*self.max_voices,:] += piano_roll[step,:]
            #assert that there is always at most one note played
            for step in range(Y.shape[0]):
                assert(np.sum(Y[step,:]) <= 1)
            #cut off pitch values which are very uncommon
            #this reduces the feature space significantly
            Y = Y[:,self.low_crop:self.high_crop]

            print("Y ------")
            print(Y.shape)
            # print(Y)
            # Here we append now the HELD NOTES ROLLL
            if self.include_held_note:
                Y = np.append(Y,  np.zeros((Y.shape[0], 1)), axis=1)
                for i, pr in enumerate(chosen_piano_rolls):
                    for step in range(pr.shape[0]):
                        if np.sum(Y[i + step*self.max_voices,:]) == 1:
                            #print("note played")
                            #print(Y[i + step * self.max_voices, :])
                            pass
                        else:
                            #print("no")
                            Y[i + step*self.max_voices, -1] += chosen_held_note_rolls[i][step]


            #append silent note if desired
            #the silent note will always be at the last note
            if self.include_silent_note:
                Y = np.append(Y, np.zeros((Y.shape[0], 1)), axis=1)
                for step in range(Y.shape[0]):
                    if np.sum(Y[step]) == 0:
                        Y[step, -1] = 1
                #assert that there is now a 1 at every step
                for step in range(Y.shape[0]):
                    assert(np.sum(Y[step,:]) == 1)

            #unroll the velocity roll
            #V will only have shape (song_length,) and it's values will be between 0 and 1 (divide by MAX_VELOCITY)
            V = np.zeros((song_length,))
            for i, velocity_roll in enumerate(chosen_velocity_rolls):
                for step in range(velocity_roll.shape[0]):
                    if velocity_roll[step] > 0:
                        velocity = self.velocity_threshold + \
                        (velocity_roll[step] / self.max_velocity) * (1.0 - self.velocity_threshold)
                        # a note is therefore at least 0.1*max_velocity loud
                        # but this is good, since we can now more clearly distinguish between silent or played notes
                        assert(velocity <= 1.0)
                        V[i + step*self.max_voices] = velocity


            #unroll the held_note_rolls
            #D will only have shape (song_length,) and it's values will be  0 or 1 (1 if held)
            #it's name is D for Duration to not have a name clash with the history (H)
            D = np.zeros((song_length,))
            for i, held_note_roll in enumerate(chosen_held_note_rolls):
                for step in range(held_note_roll.shape[0]):
                    D[i + step*self.max_voices] = held_note_roll[step]

            instrument_feature_matrix = mf.programs_to_instrument_matrix(chosen_programs,
                                                                         self.instrument_attach_method,
                                                                         self.max_voices)

            if self.attach_instruments:
                instrument_feature_matrix = np.transpose(np.tile(np.transpose(instrument_feature_matrix), song_length//self.max_voices))
                Y = np.append(Y, instrument_feature_matrix, axis=1)
            X = Y

            if save_preprocessed_midi: mf.rolls_to_midi(Y,
                                                        chosen_programs,
                                                        'preprocess_midi_data/' + t + '/',
                                                        name,
                                                        tempo,
                                                        self.low_crop,
                                                        self.high_crop,
                                                        self.num_notes,
                                                        self.velocity_threshold,
                                                        V,
                                                        D)


            #split the song into chunks of size output_length or input_length
            #pad them with silent notes if necessary
            if self.input_length > 0:

                #split X
                padding_length = self.input_length - (X.shape[0] % self.input_length)
                if self.input_length == padding_length:
                    padding_length = 0
                else:
                    #pad to the right..
                    X = np.pad(X, ((0,padding_length),(0, 0)), 'constant', constant_values=(0, 0))
                    if self.include_silent_note:
                        X[-padding_length:,-1] = 1
                number_of_splits = X.shape[0] // self.input_length
                X = np.split(X, number_of_splits)
                X = np.asarray(X)

            if self.output_length > 0:
                #split Y
                padding_length = self.output_length - (Y.shape[0] % self.output_length)

                if self.output_length == padding_length:
                    padding_length = 0
                else:
                    print("Padding Length", padding_length)
                    #pad to the right..
                    Y = np.pad(Y, ((0,padding_length),(0, 0)), 'constant', constant_values=(0, 0))
                    print(Y)
                    if self.include_silent_note:
                        Y[-padding_length:,-1] = 1
                number_of_splits = Y.shape[0] // self.output_length
                Y = np.split(Y, number_of_splits)
                Y = np.asarray(Y)

                #split V
                #pad to the right with zeros..
                V = np.pad(V, (0,padding_length), 'constant', constant_values=0)
                number_of_splits = V.shape[0] // self.output_length
                V = np.split(V, number_of_splits)
                V = np.asarray(V)

                #split D
                #pad to the right with zeros..
                D = np.pad(D, (0,padding_length), 'constant', constant_values=0)
                number_of_splits = D.shape[0] // self.output_length
                D = np.split(D, number_of_splits)
                D = np.asarray(D)

            # print(X)
            # print(X[0])
            # print(sum(X[0][0]))
            return X, Y, instrument_feature_matrix, tempo, V, D
        else:
            return None, None, None, None, None, None
        
    def import_midi_from_folder(self, 
                                folder, 
                                save_imported_midi_as_pickle, 
                                save_preprocessed_midi):
        X_list = []
        Y_list = []
        paths = []
        c_classes = []
        I_list = []
        T_list = []
        V_list = []
        D_list = []
        no_imported = 0
        n_input_seqs = 0
        n_dropped = 0

        for path, subdirs, files in os.walk(folder):
            for name in files:
                _path = path.replace('\\', '/') + '/'
                _name = name.replace('\\', '/')

                if _name.endswith('.mid') or _name.endswith('.midi'):

                    shortpath = _path[len(folder):]
                    found = False
                    for i, c in enumerate(self.classes):
                        if c.lower() in shortpath.lower():
                            found = True
                            print("Importing " + c + " song called " + _name)
                            C = i
                            if not self.only_unknown:

                                X, Y, I, T, V, D = self.load_rolls(_path, _name, save_preprocessed_midi)

                                if X is not None and Y is not None:
                                    X_list.append(X)
                                    Y_list.append(Y)
                                    I_list.append(I)
                                    T_list.append(T)
                                    V_list.append(V)
                                    D_list.append(D)
                                    paths.append(_path + _name)
                                    c_classes.append(C)
                                    no_imported += 1
                            break
                    if not found:
                        #assign new category for all the files with no proper title
                        if self.include_unknown:
                            C = self.num_classes -1
                            print("Importing unknown song ", _name)

                            X, Y, I, T, V, D = self.load_rolls(_path, _name, save_preprocessed_midi)



                            if X is not None and Y is not None:
                                n_input_seqs += X.shape[0]
                                print("X-------")
                                print(X)
                                # print(X[0])
                                clean_x, clean_y, clean_v, clean_d, dropped = self.drop_empty_seqs(X, Y, V, D)
                                n_dropped += dropped

                                X_list.append(clean_x)
                                Y_list.append(clean_y)
                                V_list.append(clean_v)
                                D_list.append(clean_d)
                                # X_list.append(X)
                                # Y_list.append(Y)
                                I_list.append(I)
                                T_list.append(T)
                                # V_list.append(V)
                                # D_list.append(D)
                                paths.append(_path + _name)
                                c_classes.append(C)
                                no_imported += 1


        assert(len(X_list) == len(paths))
        assert(len(X_list) == len(c_classes))
        assert(len(X_list) == len(I_list))
        assert(len(X_list) == len(T_list))
        assert(len(X_list) == len(D_list))
        assert(len(X_list) == len(V_list))

        print(f"Sequences (16 bars) imported: {n_input_seqs}")
        print(f"Sequences (silent) dropped: {n_dropped}")
        print(f"Percentage silent sequences: {round(n_dropped / n_input_seqs *100, 2)} %")

        unique, counts = np.unique(c_classes, return_counts=True)

        data = train_test_split(V_list, 
                                D_list, 
                                T_list, 
                                I_list, 
                                Y_list, 
                                X_list, 
                                c_classes,
                                paths, 
                                test_size=self.test_fraction, 
                                random_state=42, 
                                stratify=c_classes)
        
        V_train = data[0]
        V_test = data[1] 
        D_train = data[2]
        D_test = data[3]
        T_train = data[4]
        T_test = data[5]
        I_train = data[6]
        I_test = data[7]
        Y_train = data[8]
        Y_test = data[9]
        X_train = data[10]
        X_test = data[11]
        c_train = data[12]
        c_test = data[13]
        train_paths = data[14]
        test_paths = data[15]

        train_set_size = len(X_train)
        test_set_size = len(X_test)  

        if save_imported_midi_as_pickle:
            if not os.path.exists(self.pickle_store_folder):
                os.makedirs(self.pickle_store_folder)
                
            pickle.dump(V_train,open(self.pickle_store_folder+'/V_train.pickle', 'wb'))
            pickle.dump(V_test,open(self.pickle_store_folder+'/V_test.pickle', 'wb'))

            pickle.dump(D_train,open(self.pickle_store_folder+'/D_train.pickle', 'wb'))
            pickle.dump(D_test,open(self.pickle_store_folder+'/D_test.pickle', 'wb'))

            pickle.dump(T_train,open(self.pickle_store_folder+'/T_train.pickle', 'wb'))
            pickle.dump(T_test,open(self.pickle_store_folder+'/T_test.pickle', 'wb'))

            pickle.dump(I_train,open(self.pickle_store_folder+'/I_train.pickle', 'wb'))
            pickle.dump(I_test,open(self.pickle_store_folder+'/I_test.pickle', 'wb'))

            pickle.dump(Y_train,open(self.pickle_store_folder+'/Y_train.pickle', 'wb'))
            pickle.dump(Y_test,open(self.pickle_store_folder+'/Y_test.pickle', 'wb'))

            pickle.dump(X_train,open(self.pickle_store_folder+'/X_train.pickle', 'wb'))
            pickle.dump(X_test,open(self.pickle_store_folder+'/X_test.pickle', 'wb'))

            pickle.dump(c_train,open(self.pickle_store_folder+'/c_train.pickle', 'wb'))
            pickle.dump(c_test,open(self.pickle_store_folder+'/c_test.pickle', 'wb'))

            pickle.dump(train_paths,open(self.pickle_store_folder+'/train_paths.pickle', 'wb'))
            pickle.dump(test_paths,open(self.pickle_store_folder+'/test_paths.pickle', 'wb'))

        self.plot_dataset_metrics(X_train)
        return data


    