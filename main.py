import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def main():
    # todo: replace with regex search for all files in data folder.
    filepaths = ['data/03_automatic_stop.mp3']
    for filepath in filepaths:
        analyse_audio(filepath)

def analyse_audio(filepath):
    # Load audio from file (snippet).
    y, sr = librosa.load(filepath, duration=8.7)

    # Plot waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    plt.show()

    # todo: harmonic/percussive separation.
    # harmonic, percussive = librosa.decompose.hpss()
    y_harm = librosa.effects.harmonic(y=y, margin=8)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

    # todo: filter following this page: https://librosa.org/doc/main/auto_examples/plot_chroma.html
    fig, ax = plt.subplots()
    # img = librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time', ax=ax)
    img = librosa.display.specshow(chroma_harm, y_axis='chroma', x_axis='time', ax=ax)
    ax.set_title('Chroma CQT')
    fig.colorbar(img, ax=ax)
    plt.show()

    # todo: combine with plot above.
    C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                   y_axis='cqt_note', x_axis='time', bins_per_octave=12*3,
                                   ax=ax)
    fig.colorbar(img, ax=[ax], format="%+2.f dB")
    ax.label_outer()
    plt.show()

if __name__ == '__main__':
    main()
