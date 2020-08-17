# Taken from MidiVAE (Brunner et al. 2018)
import numpy as np
import _pickle as pickle
import os
import sys
import pretty_midi as pm
import mido
import operator


def programs_to_instrument_matrix(programs, instrument_attach_method, max_voices):

    if instrument_attach_method == '1hot-instrument':
        #very large, not recommended
        instrument_feature_matrix = np.zeros((max_voices, 128))
        for i, program in enumerate(programs):
            instrument_feature_matrix[i, program] = 1

    elif instrument_attach_method == '1hot-category':
        #categories according to midi declaration, https://en.wikipedia.org/wiki/General_MIDI
        #8 consecutive instruments make 1 category
        instrument_feature_matrix = np.zeros((max_voices, 16))
        for i, program in enumerate(programs):
            instrument_feature_matrix[i, program//8] = 1
        
    elif instrument_attach_method == 'khot-instrument':
        #make a khot vector in log2 base for the instrument
        #log2(128) = 7
        instrument_feature_matrix = np.zeros((max_voices, 7))
        for i, program in enumerate(programs):
            p = program
            for exponent in range(7):
                if p % 2 == 0:
                    instrument_feature_matrix[i, exponent] = 1
                p = p // 2
    elif instrument_attach_method == 'khot-category':
        #categories according to midi declaration, https://en.wikipedia.org/wiki/General_MIDI
        #8 consecutive instruments make 1 category
        #make a khot vector in log2 base for the category
        #log2(16) = 4
        instrument_feature_matrix = np.zeros((max_voices, 4))
        for i, program in enumerate(programs):
            p = program//8
            for exponent in range(4):
                if p % 2 == 1:
                    instrument_feature_matrix[i, exponent] = 1
                p = p // 2
    else:
        print("Not implemented!")

    return instrument_feature_matrix


def rolls_to_midi(pianoroll, 
                  programs, 
                  save_folder, 
                  filename, 
                  bpm, 
                  low_crop,
                  high_crop,
                  num_notes,
                  velocity_threshold,
                  velocity_roll=None, 
                  held_notes_roll=None, 
                  smallest_note=16,
                  max_velocity=127):

    #bpm is in quarter notes, so scale accordingly
#     bpm = bpm * (smallest_note / 4)

    pianoroll = np.pad(np.copy(pianoroll), ((0,0),(low_crop,num_notes-high_crop)), mode='constant', constant_values=0)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    midi = pm.PrettyMIDI(initial_tempo=bpm, resolution=1000)    # TODO change resolution to sensible PPC: between 24 and 960
    midi.time_signature_changes.append(pm.TimeSignature(4, 4, 0))

    for voice, program in enumerate(programs):
    
        current_instrument = pm.Instrument(program=program)
        current_pianoroll = pianoroll[voice::len(programs),:]

        if velocity_roll is not None:
            current_velocity_roll = np.copy(velocity_roll[voice::len(programs)])
            #during the training, the velocities were scaled to be in the range 0,1
            #scale it back to the actual velocity numbers
            current_velocity_roll[np.where(current_velocity_roll < velocity_threshold)] = 0
            current_velocity_roll[np.where(current_velocity_roll >= velocity_threshold)] -= 0.5
            current_velocity_roll /= (1.0 - velocity_threshold)
            current_velocity_roll *= max_velocity

        if held_notes_roll is not None:
            current_held_notes_roll = np.copy(held_notes_roll[voice::len(programs)])


            
        tracker = []
        start_times  = dict()
        velocities = dict()
        for i, note_vector in enumerate(current_pianoroll):
            notes = list(note_vector.nonzero()[0])
    #       
            #notes that were just played and need to be removed from the tracker
            removal_list = []
            for note in tracker:

                #determine if you still hold this note or not
                hold_this_note = True
                if held_notes_roll is not None:
                    hold_this_note = current_held_notes_roll[i] > 0.5

                    #it may happen that a note seems to be held but has switched to another channel
                    #in that case, play the note anyways
                    if note not in notes:
                        hold_this_note = False

                else:
                    hold_this_note = note in notes and (i)% smallest_note is not 0

                if hold_this_note:
                    #held note, don't play a new note
                    notes.remove(note)
                else:
                    if velocity_roll is not None:
                        velocity = velocities[note]
                        if velocity > max_velocity:
                            velocity = int(max_velocity)
                    else:
                        velocity = 80

                    midi_note = pm.Note(velocity=velocity, pitch=note, start=(60/bpm)*start_times[note], end=(60/bpm)*i)
                    current_instrument.notes.append(midi_note)

                    removal_list.append(note)
            for note in removal_list:
                tracker.remove(note)


            for note in notes:
                tracker.append(note)
                start_times[note]=i
                if velocity_roll is not None:
                    velocities[note] = int(current_velocity_roll[i])

        midi.instruments.append(current_instrument)
    midi.write(os.path.join(save_folder,filename+'.mid'))