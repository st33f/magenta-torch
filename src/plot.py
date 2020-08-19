import pypianoroll as pr
#from src.reconstruction import *
from matplotlib import pyplot as plt
from music21 import *

output_midi = "/Users/stefanwijtsma/code/magenta-torch/midi_reconstruction/Petite Fleur.mid"
#output_midi = "/Users/stefanwijtsma/code/mt/data/clean_midi/Chris Barber's Jazz Band/Petite Fleur.mid"
example_midi = "/Users/stefanwijtsma/code/magenta-torch/preprocess_midi_data/1595508226/Petite Fleur.mid.mid"
model_midi = pr.Multitrack(output_midi)
true_midi = pr.Multitrack(example_midi)

# fakePiece = converter.parse(output_midi)
# fakePiece.plot()
#
# #piece = converter.parse(example_midi)
# # set ticks
# # piece.plot()
#
# print(analysis.correlate.ActivityMatch(fakePiece))
#
# plt.show()


# midi = pretty_midi.PrettyMIDI(midi_file=example_midi)
#
#
# mb = MidiBuilder()

# mb.plot_midi(midi)
# plt.show()






# Music 21 shizzle
# seq = converter.parse(output_midi)
# seq.plot('pianoroll', titile="")

# print("Model Output: \n")
# print(model_midi.get_active_length())
# print(model_midi.tempo)
# print(model_midi.beat_resolution)
# print(model_midi.count_downbeat())
# for instrument in model_midi.tracks:
#     #print(f"track: {instrument}")
#     n_pitches = pr.metrics.n_pitches_used(instrument.pianoroll)
#     pitch_classes = pr.metrics.n_pitch_classes_used(instrument.pianoroll)
#     empty_beats = pr.metrics.empty_beat_rate(instrument.pianoroll, beat_resolution=4)
#     print(f"n_pitches: {n_pitches}, pitch_classes: {pitch_classes}, empty_beats: {empty_beats}")
# print("\n")
#
#
# print("True input: \n")
# print(true_midi.get_active_length())
# print(true_midi.tempo)
# print(true_midi.beat_resolution)
# print(true_midi.count_downbeat())
# for instrument in true_midi.tracks:
#     #print(f"track: {instrument}")
#     n_pitches = pr.metrics.n_pitches_used(instrument.pianoroll)
#     pitch_classes = pr.metrics.n_pitch_classes_used(instrument.pianoroll)
#     empty_beats = pr.metrics.empty_beat_rate(instrument.pianoroll, beat_resolution=4)
#     print(f"n_pitches: {n_pitches}, pitch_classes: {pitch_classes}, empty_beats: {empty_beats}")
# print("\n")

# fig, ax = pr.plot_pianoroll(model_midi.tracks[0], beat_resolution=4, grid='y')


# Plot the multitrack pianoroll
# fig, axs = midi.plot()
# fig, ax = pr.plot_multitrack(model_midi, grid='off')
# fig1, ax1 = pr.plot_multitrack(true_midi, grid='off')
#
#
# plt.show()

#pypianoroll.pad_to_same(midi)


