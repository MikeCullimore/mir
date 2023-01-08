import os.path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def main():
    # todo: replace with regex search for all files in data folder.
    folder = 'data'
    # filenames = ['03_automatic_stop.mp3']
    # filenames = ['Four Out Of Five.wav']
    filenames = ['c-major-scale.ogg']
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        analyse_audio(filepath)

def analyse_audio(filepath):
    # Load audio from file (snippet).
    y, sr = librosa.load(filepath, duration=16)

    # Estimate tempo.
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # For display purposes, let's zoom in on a 15-second chunk from the middle of the song
    # https://librosa.org/doc/main/auto_examples/plot_chroma.html
    # todo: wrap as method.
    # todo: how to truncate original signal rather than chroma?
    start = 4
    # finish = start + 1.5
    finish = start + 5
    # finish = 16
    idx = tuple([slice(None), slice(*list(librosa.time_to_frames([start, finish])))])

    # Plot waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    plt.show()

    # # todo: harmonic/percussive separation.
    # # harmonic, percussive = librosa.decompose.hpss()
    # y_harm = librosa.effects.harmonic(y=y, margin=8)
    # chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)[idx]
    # chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)[idx]

    # # Plot chroma.
    # # todo: filter following this page: https://librosa.org/doc/main/auto_examples/plot_chroma.html
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time', ax=ax)
    # # img = librosa.display.specshow(chroma_harm, y_axis='chroma', x_axis='time', ax=ax)
    # ax.set_title('Chroma CQT')
    # fig.colorbar(img, ax=ax)
    # plt.show()

    # # Plot chroma to high frequency.
    # # todo: combine with plot above.
    # # todo: drop bins below C3.
    # C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))
    # data = librosa.amplitude_to_db(C, ref=np.max)[idx]
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(data,
    #                                y_axis='cqt_note', x_axis='time', bins_per_octave=12*3,
    #                                ax=ax)
    # fig.colorbar(img, ax=[ax], format="%+2.f dB")
    # ax.label_outer()
    # plt.show()

    # NMF
    # todo: STFT as input instead of CQT?
    # todo: configure so that the components are equal temperament, A4 = 440Hz (think piano keys).
    # todo: shift tuning (A4) to optimise template matching. But are glitchy activations down to vibrato? Also tuning is automatically estimated from the signal, see https://librosa.org/doc/latest/generated/librosa.cqt.html
    # todo: axis labels (note scientific names).
    # for n_steps in np.linspace(-1, 1, 7):
        # y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.1) # todo: optimise n_steps for clean activations.
    # S = np.abs(librosa.stft(y))[idx]
    S = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))[idx]
    comps, acts = librosa.decompose.decompose(S, n_components=16, sort=True)
    fig, ax = plt.subplots(1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(comps, ref=np.max), ax=ax[0], y_axis='cqt_note')
    librosa.display.specshow(acts, ax=ax[1], cmap='gray_r', x_axis='time')
    ax[0].set_title('Components')
    # ax[0].set_xlabel('Note')
    # ax[0].set_ylabel('Frequency [Hz]')
    ax[1].set_title('Activations')
    fig.suptitle(f'Identiyfing note activations via NMF. Input: {filepath}')
    plt.show()

if __name__ == '__main__':
    main()
