#!/usr/bin/env python3
import sys
import pypianoroll as pr
from matplotlib import pyplot as plt
from music21 import *
import pretty_midi
sys.path.append(".")

import numpy as np
#from helpers.fake_dataset import generate_fake_songs
from src.helpers import generate_fake_songs
#from src.reconstruction import *
import wandb

def plot_pianoroll(
    ax,
    pianoroll,
    is_drum=False,
    beat_resolution=None,
    downbeats=None,
    preset="default",
    cmap="Greens",
    xtick="auto",
    ytick="octave",
    xticklabel=True,
    yticklabel="auto",
    tick_loc=None,
    tick_direction="in",
    label="both",
    grid="both",
    grid_linestyle=":",
    grid_linewidth=0.5,
):
    """
    Plot a pianoroll given as a numpy array.

    Parameters
    ----------
    ax : matplotlib.axes.Axes object
        A :class:`matplotlib.axes.Axes` object where the pianoroll will be
        plotted on.
    pianoroll : np.ndarray
        A pianoroll to be plotted. The values should be in [0, 1] when data type
        is float, and in [0, 127] when data type is integer.

        - For a 2D array, shape=(num_time_step, num_pitch).
        - For a 3D array, shape=(num_time_step, num_pitch, num_channel), where
          channels can be either RGB or RGBA.

    is_drum : bool
        A boolean number that indicates whether it is a percussion track.
        Defaults to False.
    beat_resolution : int
        The number of time steps used to represent a beat. Required and only
        effective when `xtick` is 'beat'.
    downbeats : list
        An array that indicates whether the time step contains a downbeat (i.e.,
        the first time step of a bar).

    preset : {'default', 'plain', 'frame'}
        A string that indicates the preset theme to use.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmap :  `matplotlib.colors.Colormap`
        The colormap to use in :func:`matplotlib.pyplot.imshow`. Defaults to
        'Blues'. Only effective when `pianoroll` is 2D.
    xtick : {'auto', 'beat', 'step', 'off'}
        A string that indicates what to use as ticks along the x-axis. If 'auto'
        is given, automatically set to 'beat' if `beat_resolution` is also given
        and set to 'step', otherwise. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        A string that indicates what to use as ticks along the y-axis.
        Defaults to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis. Only effective when `xtick`
        is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum` is
        True) as tick labels along the y-axis. If 'number', use pitch number. If
        'auto', set to 'name' when `ytick` is 'octave' and 'number' when `ytick`
        is 'pitch'. Defaults to 'auto'. Only effective when `ytick` is not
        'off'.
    tick_loc : tuple or list
        The locations to put the ticks. Availables elements are 'bottom', 'top',
        'left' and 'right'. Defaults to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        A string that indicates where to put the ticks. Defaults to 'in'. Only
        effective when one of `xtick` and `ytick` is on.
    label : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add labels to the x-axis and y-axis.
        Defaults to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add grids to the x-axis, y-axis, both
        or neither. Defaults to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.

    """

    if pianoroll.ndim not in (2, 3):
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")
    if pianoroll.shape[1] != 128:
        raise ValueError("The length of the second axis of `pianoroll` must be 128.")
    if xtick not in ("auto", "beat", "step", "off"):
        raise ValueError("`xtick` must be one of {'auto', 'beat', 'step', 'none'}.")
    if xtick == "beat" and beat_resolution is None:
        raise ValueError("`beat_resolution` must be specified when `xtick` is 'beat'.")
    if ytick not in ("octave", "pitch", "off"):
        raise ValueError("`ytick` must be one of {octave', 'pitch', 'off'}.")
    if not isinstance(xticklabel, bool):
        raise TypeError("`xticklabel` must be bool.")
    if yticklabel not in ("auto", "name", "number", "off"):
        raise ValueError(
            "`yticklabel` must be one of {'auto', 'name', 'number', 'off'}."
        )
    if tick_direction not in ("in", "out", "inout"):
        raise ValueError("`tick_direction` must be one of {'in', 'out', 'inout'}.")
    if label not in ("x", "y", "both", "off"):
        raise ValueError("`label` must be one of {'x', 'y', 'both', 'off'}.")
    if grid not in ("x", "y", "both", "off"):
        raise ValueError("`grid` must be one of {'x', 'y', 'both', 'off'}.")

    # plotting
    if pianoroll.ndim > 2:
        to_plot = pianoroll.transpose(1, 0, 2)
    else:
        to_plot = pianoroll.T
    if np.issubdtype(pianoroll.dtype, np.bool_) or np.issubdtype(
        pianoroll.dtype, np.floating
    ):
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="none",
            alpha=0.5
        )
    elif np.issubdtype(pianoroll.dtype, np.integer):
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=127,
            origin="lower",
            interpolation="none",
        )
    else:
        raise TypeError("Unsupported data type for `pianoroll`.")

    # tick setting
    if tick_loc is None:
        tick_loc = ("bottom", "left")
    if xtick == "auto":
        xtick = "beat" if beat_resolution is not None else "step"
    if yticklabel == "auto":
        yticklabel = "name" if ytick == "octave" else "number"

    if preset == "plain":
        ax.axis("off")
    elif preset == "frame":
        ax.tick_params(
            direction=tick_direction,
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
    else:
        ax.tick_params(
            direction=tick_direction,
            bottom=("bottom" in tick_loc),
            top=("top" in tick_loc),
            left=("left" in tick_loc),
            right=("right" in tick_loc),
            labelbottom=(xticklabel != "off"),
            labelleft=(yticklabel != "off"),
            labeltop=False,
            labelright=False,
        )

    # x-axis
    if xtick == "beat" and preset != "frame":
        num_beat = pianoroll.shape[0] // beat_resolution
        ax.set_xticks(beat_resolution * np.arange(num_beat) - 0.5)
        ax.set_xticklabels("")
        ax.set_xticks(beat_resolution * (np.arange(num_beat) + 0.5) - 0.5, minor=True)
        ax.set_xticklabels(np.arange(1, num_beat + 1), minor=True, size=6)
        ax.tick_params(axis="x", which="minor", width=0)

    # y-axis
    if ytick == "octave":
        ax.set_yticks(np.arange(0, 128, 12))
        if yticklabel == "name":
            ax.set_yticklabels(["C{}".format(i - 2) for i in range(11)])
    elif ytick == "step":
        ax.set_yticks(np.arange(0, 128))
        if yticklabel == "name":
            if is_drum:
                ax.set_yticklabels(
                    [pretty_midi.note_number_to_drum_name(i) for i in range(128)]
                )
            else:
                ax.set_yticklabels(
                    [pretty_midi.note_number_to_name(i) for i in range(128)]
                )

    # axis labels
    if label in ("x", "both"):
        if xtick == "step" or not xticklabel:
            ax.set_xlabel("time (step)")
        else:
            ax.set_xlabel("time (beat)")

    if label in ("y", "both"):
        if is_drum:
            ax.set_ylabel("key name")
        else:
            ax.set_ylabel("pitch")

    # grid
    if grid != "off":
        ax.grid(
            axis=grid, color="k", linestyle=grid_linestyle, linewidth=grid_linewidth
        )

    # downbeat boarder
    if downbeats is not None and preset != "plain":
        for step in downbeats:
            ax.axvline(x=step, color="k", linewidth=1)

def plot_multitrack(
    multitrack,
    filename=None,
    mode="separate",
    track_label="name",
    preset="default",
    cmaps=None,
    xtick="auto",
    ytick="octave",
    xticklabel=True,
    yticklabel="auto",
    tick_loc=None,
    tick_direction="in",
    label="both",
    grid="both",
    grid_linestyle=":",
    grid_linewidth=0.5,
):
    """
    Plot the pianorolls or save a plot of them.

    Parameters
    ----------
    filename : str
        The filename to which the plot is saved. If None, save nothing.
    mode : {'separate', 'stacked', 'hybrid'}
        A string that indicate the plotting mode to use. Defaults to 'separate'.

        - In 'separate' mode, all the tracks are plotted separately.
        - In 'stacked' mode, a color is assigned based on `cmaps` to the
          pianoroll of each track and the pianorolls are stacked and plotted as
          a colored image with RGB channels.
        - In 'hybrid' mode, the drum tracks are merged into a 'Drums' track,
          while the other tracks are merged into an 'Others' track, and the two
          merged tracks are then plotted separately.

    track_label : {'name', 'program', 'family', 'off'}
        A sting that indicates what to use as labels to the track. When `mode`
        is 'hybrid', all options other than 'off' will label the two track with
        'Drums' and 'Others'.
    preset : {'default', 'plain', 'frame'}
        A string that indicates the preset theme to use.

        - In 'default' preset, the ticks, grid and labels are on.
        - In 'frame' preset, the ticks and grid are both off.
        - In 'plain' preset, the x- and y-axis are both off.

    cmaps :  tuple or list
        The `matplotlib.colors.Colormap` instances or colormap codes to use.

        - When `mode` is 'separate', each element will be passed to each call of
          :func:`matplotlib.pyplot.imshow`. Defaults to ('Blues', 'Oranges',
          'Greens', 'Reds', 'Purples', 'Greys').
        - When `mode` is stacked, a color is assigned based on `cmaps` to the
          pianoroll of each track. Defaults to ('hsv').
        - When `mode` is 'hybrid', the first (second) element is used in the
          'Drums' ('Others') track. Defaults to ('Blues', 'Greens').

    xtick : {'auto', 'beat', 'step', 'off'}
        A string that indicates what to use as ticks along the x-axis. If 'auto'
        is given, automatically set to 'beat' if `beat_resolution` is also given
        and set to 'step', otherwise. Defaults to 'auto'.
    ytick : {'octave', 'pitch', 'off'}
        A string that indicates what to use as ticks along the y-axis. Defaults
        to 'octave'.
    xticklabel : bool
        Whether to add tick labels along the x-axis. Only effective when `xtick`
        is not 'off'.
    yticklabel : {'auto', 'name', 'number', 'off'}
        If 'name', use octave name and pitch name (key name when `is_drum` is
        True) as tick labels along the y-axis. If 'number', use pitch number. If
        'auto', set to 'name' when `ytick` is 'octave' and 'number' when `ytick`
        is 'pitch'. Defaults to 'auto'. Only effective when `ytick` is not
        'off'.
    tick_loc : tuple or list
        The locations to put the ticks. Availables elements are 'bottom', 'top',
        'left' and 'right'. Defaults to ('bottom', 'left').
    tick_direction : {'in', 'out', 'inout'}
        A string that indicates where to put the ticks. Defaults to 'in'. Only
        effective when one of `xtick` and `ytick` is on.
    label : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add labels to the x-axis and y-axis.
        Defaults to 'both'.
    grid : {'x', 'y', 'both', 'off'}
        A string that indicates whether to add grids to the x-axis, y-axis, both
        or neither. Defaults to 'both'.
    grid_linestyle : str
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linestyle'
        argument.
    grid_linewidth : float
        Will be passed to :meth:`matplotlib.axes.Axes.grid` as 'linewidth'
        argument.

    Returns
    -------
    fig : `matplotlib.figure.Figure` object
        A :class:`matplotlib.figure.Figure` object.
    axs : list
        List of :class:`matplotlib.axes.Axes` object.

    """

    def get_track_label(track_label, track=None):
        """Return corresponding track labels."""
        if track_label == "name":
            return track.name
        if track_label == "program":
            return pretty_midi.program_to_instrument_name(track.program)
        if track_label == "family":
            return pretty_midi.program_to_instrument_class(track.program)
        return track_label

    def add_tracklabel(ax, track_label, track=None):
        """Add a track label to an axis."""
        if not ax.get_ylabel():
            return
        ax.set_ylabel(get_track_label(track_label, track) + "\n\n" + ax.get_ylabel())

    multitrack.check_validity()
    if not multitrack.tracks:
        raise ValueError("There is no track to plot.")
    if mode not in ("separate", "stacked", "hybrid"):
        raise ValueError("`mode` must be one of {'separate', 'stacked', 'hybrid'}.")
    if track_label not in ("name", "program", "family", "off"):
        raise ValueError("`track_label` must be one of {'name', 'program', 'family'}.")

    if cmaps is None:
        if mode == "separate":
            cmaps = ("Blues", "Oranges", "Greens", "Reds", "Purples", "Greys")
        elif mode == "stacked":
            cmaps = "hsv"
            #cmaps = ("Blues", "Oranges", "Greens", "Reds", "Purples", "Greys")
        else:
            cmaps = ("Blues", "Greens")

    num_track = len(multitrack.tracks)
    print(f"num track {num_track}")
    downbeats = multitrack.get_downbeat_steps()

    if mode == "separate":
        if num_track > 1:
            fig, axs = plt.subplots(num_track, sharex=True)
            fig.patch.set_facecolor('red')
            plt.rcParams['figure.facecolor'] = 'red'
        else:
            fig, ax = plt.subplots()
            axs = [ax]


        for idx, track in enumerate(multitrack.tracks):
            now_xticklabel = xticklabel if idx < num_track else False
            plot_pianoroll(
                axs[idx],
                track.pianoroll,
                False,
                multitrack.beat_resolution,
                downbeats,
                preset=preset,
                cmap=cmaps,
                xtick=xtick,
                ytick=ytick,
                xticklabel=now_xticklabel,
                yticklabel=yticklabel,
                tick_loc=tick_loc,
                tick_direction=tick_direction,
                label=label,
                grid=grid,
                grid_linestyle=grid_linestyle,
                grid_linewidth=grid_linewidth,
            )
            # if track_label != "none":
            #     add_tracklabel(axs[idx], track_label, track)

        if num_track > 1:
            fig.subplots_adjust(hspace=0)

        if filename is not None:
            plt.savefig(filename)

        return (fig, axs)

    if mode == "stacked":
        is_all_drum = True
        for track in multitrack.tracks:
            if not track.is_drum:
                is_all_drum = False

        fig, ax = plt.subplots()
        ax.set_ylim(bottom=36, top=116)
        stacked = multitrack.get_stacked_pianoroll()

        colormap = plt.cm.get_cmap(cmaps)
        colormatrix = colormap(np.arange(0, 1, 1 / num_track))[:, :3]
        recolored = np.matmul(stacked.reshape(-1, num_track), colormatrix)
        stacked = recolored.reshape(stacked.shape[:2] + (3,))
        plot_pianoroll(
            ax,
            stacked,
            is_all_drum,
            multitrack.beat_resolution,
            downbeats,
            preset=preset,
            xtick=xtick,
            ytick=ytick,
            xticklabel=xticklabel,
            yticklabel=yticklabel,
            tick_loc=tick_loc,
            tick_direction=tick_direction,
            label=label,
            grid=grid,
            grid_linestyle=grid_linestyle,
            grid_linewidth=grid_linewidth,
        )


        if filename is not None:
            plt.savefig(filename)

        return (fig, [ax])



# output_midi = "/Users/stefanwijtsma/code/magenta-torch/midi_reconstruction/Petite Fleur.mid"
# output_midi = pr.parse("/Users/stefanwijtsma/code/mt/data/clean_midi/Chris Barber's Jazz Band/Petite Fleur.mid")
# example_midi = "/Users/stefanwijtsma/code/magenta-torch/preprocess_midi_data/1595508226/Petite Fleur.mid.mid"
# model_midi = pr.Multitrack(output_midi)
# true_midi = pr.Multitrack(example_midi)



# This is for testing the plotting function

fake_data = generate_fake_songs(1, 1)

midi_start = 24
midi_end = 85

data = fake_data[0][0]
mask = data == 1
notes = np.where(mask)[0]

#n = np.reshape(notes, (-1, 256))

columns = [pretty_midi.note_number_to_name(n) for n in range(midi_start, midi_end)]

num_array = np.zeros([256, 128])
print(num_array.shape)
print(data.shape)
#np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
npad = ((0, 0), (48, 19))
data_padded = np.pad(data, pad_width=npad, mode='constant', constant_values=0)

numpy_midi = pr.Track(data_padded)
np2 = pr.Track(data_padded[:128, :])

first_section = [numpy_midi, np2]
data = [numpy_midi, numpy_midi]

first_half = pr.Multitrack(tracks=[numpy_midi, np2], beat_resolution=4)
true_track = pr.Track(data_padded[30:158, :])


def plot_pred_and_target(pred, target, is_eval=True, include_silent_note=False):
    if include_silent_note:
        npad = ((0, 0), (49, 19))
    else:
        npad = ((0, 0), (48, 19))

    # first pad to full 128 midi notes
    pred_padded = np.pad(pred, pad_width=npad, mode='constant', constant_values=0)
    target_padded = np.pad(target, pad_width=npad, mode='constant', constant_values=0)

    # convert into pypianoroll tracks
    pred_track = pr.Track(pred_padded)
    target_track = pr.Track(target_padded)

    # plot both as multitrack
    to_plot = pr.Multitrack(tracks=[pred_track, target_track], beat_resolution=4)
    fig, (ax1) = plot_multitrack(to_plot, mode="stacked", grid="both")
    ax1[0].set_ylim(bottom=36, top=116)
    if is_eval:
        wandb.log({"Eval Pianorolls": fig})
    else:
        wandb.log({"Training Pianorolls": fig})
    plt.close('all')

    #plt.show()

def plot_spectogram(pred, target, num_plots=1, is_eval=False):
    print("--- PLOT SPECTORGRAM ---")
    first_target = target[:, 0, :]
    first_pred = pred[:, 0, :]
    print(first_target.size())
    print(target.size())

    # Plot Spectorgram for pred and target
    plt.subplot(211)
    plt.title('Spectrogram of pred and target')
    plt.plot(first_target)

    plt.xlabel('Time')
    plt.ylabel('Note')

    plt.subplot(212)


    # print(pred.size())
    print(len(pred))
    print(len(pred[0]))
    print(pred)

    pred_viz = [item for sublist in first_pred for item in sublist]

    fig = plt.specgram(pred_viz, NFFT=len(pred[0]), noverlap=0)
    plt.xlabel('Time')
    plt.ylabel('Note')

    if is_eval:
        wandb.log({"Eval Spectogram": fig})
    else:
        wandb.log({"Training Spectogram": fig})
    plt.close('all')










# Music 21 shizzle
# seq = converter.parse(output_midi)
# seq.plot('pianoroll', titile="")
#
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

#fig, ax = pr.plot_pianoroll(model_midi.tracks[0], beat_resolution=4, grid='y')



#Plot the multitrack pianoroll
# fig, axs = midi.plot()
# fig, ax = pr.plot_multitrack(model_midi, grid='off')
# fig1, ax1 = pr.plot_multitrack(true_midi, grid='off')


#plt.show()

#pypianoroll.pad_to_same(midi)


# fig = plt.figure(figsize=(9, 13))
# columns = 1
# rows = 2
#
# # prep (x,y) for extra plotting
# xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
# ys = np.abs(np.sin(xs))           # absolute of sine
#
# # ax enables access to manipulate each of subplots
# ax = []
#
# for i in range(columns*rows):
#     #img = np.random.randint(10, size=(128, 256))
#     #img = pr.plot_track(true_track, cmap=cmaps[1], beat_resolution=4)
#     # create subplot and append to ax
#     ax.append( pr.plot_track(first_section[i], cmap=cmaps[i], beat_resolution=4))
#     ax[-1][-1].set_title("ax:"+str(i))  # set title
#     pr.plot_track(numpy_midi, cmap=cmaps[1], beat_resolution=4)
#     #plt.imshow(img, alpha=0.25)
#
# # do extra plots on selected axes/subplots
# # note: index starts with 0
# #ax[1][1].plot(true_track)
#
# plt.show()  # finally, render the plot


# Create two subplots sharing y axis
#fig, (ax1, ax2) = plt.subplots(2, sharey=True, sharex=False)
