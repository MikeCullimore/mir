# mir

Music Information Retrieval

## Setup

Command line (on Linux):
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txtpip 
```

## todo

* Use Jupyter notebooks (like [this example](https://librosa.org/doc/main/generated/librosa.decompose.decompose.html) from FMP website.)
* Analyse keyboard intro to Four out of Five:
    * Time 4-16s
    * Apply chord recognition from FMP p256.
    * Infer tuning (centre frequency) and align chroma on that.
    * Extract melody: see FMP chapter 8 and https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S2_MelodyExtractSep.html
    * Nonnegative matrix factorisation (NMF) -based audio decomposition: template matrix of MIDI note numbers and their frequencies (fundamental and harmonics) and activation matrix (note numbers vs time).
    * NMF for a track with known score (C major scale piano?), use to tune parameters to reconstruct it then apply to tracks with unknown score.
    * Apply to Salieri's march of welcome.
* Save plots to file.
* Analyse songs in data folder:
    * Chromagram
    * Audio thumbnail
    * Chord detection
    * Self-similarity matrix
    * Tempo analysis
    * Separate harmonic and percussive, save to files, listen (and adjust params if need be).
* Add songs to data folder: Beatles, White Stripes.
* Get PySoundFile working for load?
* Use [Viterbi](https://librosa.org/doc/main/auto_examples/plot_viterbi.html#sphx-glr-auto-examples-plot-viterbi-py) to truncate silence at beginning and end (only).
* Use [Laplacian segmentation](https://librosa.org/doc/main/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py) to identify and label song sections (verse, chorus etc.).
* Play audio with vertical line sweeping through plot (e.g. chroma) in sync. See FMP toolbox in [YouTube demo](https://youtu.be/g1H-7773gpo).

## Useful links

* FMP: Fundamentals of Music Processing by Meinard Mueller. See [website](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html).
* [Constant-Q transform](https://en.wikipedia.org/wiki/Constant-Q_transform)
* [Non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
* [Short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)