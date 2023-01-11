"""
Non-negative matrix factorisation (NMF) to identift notes in audio (pitch and onset time).

Aim is to generate a piano roll plot for the C major scale (notes/pitches/frequencies vs time).

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFSpecFac.html

todo:
Convert this to a Jupyter notebook.
Compare CQT instead of STFT.
Compare wavelets instead of STFT.
Pre-processing to clean up audio, especially recordings from phone.
Apply to other audio files:
    Use A4 estimate.
    Optimise tol_pitch (width of bands in pitch templates).
Improve template_pitch:
    Frequency band widths should be monotonically increasing with MIDI note number.
    Gaussian envelope for each band not rectangular.
Manual post-processing:
    Shuffle the order of the matrices so the activations look like the desired output.
    Matrices are different each time so save and reload!
    Remove broadband columns.
    Reconstruct V with the wideband signal in W discarded (white noise at onset).
    Identify columns in W which correspond to the same note and combine them (in H also).
Initialise pitches with slices from spectrogram.
    Try direct division again.
Understand how to get clean activations for C major scale, then apply to more complex audio files.
    Sort pitches by fundamental frequency (then apply same permutation to activations).
Capture tone of instrument in recording by taking slices from spectrogram in given note onset.
    Use MIDI input from keyboard to trigger samples.
    How to capture vibrato, onset noise?
Mask onsets to ignore broadband signal.
"""

import os.path

import libfmp.b
import libfmp.c8
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import soundfile as sf

def plt_show_max():
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    plt.show()

def plot_W(W):
    plt.figure()
    for i in range(W.shape[1]):
        plt.plot(W[:, i], label=str(i))
    plt.legend()
    plt_show_max()

# todo: Gaussian envelope not rectangular.
# todo: pass fundamental frequency not MIDI number.
# todo: zero harmonics above say 10 but more prominent second, third etc.
def template_pitch_custom(K, pitch, freq_res, tol_pitch=0.05, A4=440):
    """Defines spectral template for a given pitch

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        K (int): Number of frequency points
        pitch (float): Fundamental pitch
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        template (np.ndarray): Nonnegative template vector of size K
    """
    max_freq = K * freq_res
    pitch_freq = 2**((pitch - 69) / 12) * A4 # todo: lift out as own function, pass frequency as arg to this one.
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    template = np.zeros(K)
    w = 0
    for m in range(1, max_order + 1):
        min_idx = max(0, int((1 - tol_pitch) * m * pitch_freq / freq_res))
        max_idx = min(K-1, int((1 + tol_pitch) * m * pitch_freq / freq_res))
        idx = (max_idx + min_idx)//2
        if m == 1:
            w += max_idx - min_idx
        template[idx-w:idx+w] = 1 / m
        # template[min_idx:max_idx+1] = 1 / m
    return template

def template_pitch(K, pitch, freq_res, tol_pitch=0.05, A4=440):
    """Defines spectral template for a given pitch

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        K (int): Number of frequency points
        pitch (float): Fundamental pitch
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        template (np.ndarray): Nonnegative template vector of size K
    """
    max_freq = K * freq_res
    pitch_freq = 2**((pitch - 69) / 12) * A4 # todo: lift out as own function, pass frequency as arg to this one.
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    template = np.zeros(K)
    for m in range(1, max_order + 1):
        min_idx = max(0, int((1 - tol_pitch) * m * pitch_freq / freq_res))
        max_idx = min(K-1, int((1 + tol_pitch) * m * pitch_freq / freq_res))
        template[min_idx:max_idx+1] = 1 / m
    return template
        
def init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05, A4=440):
    """Initializes template matrix for a given set of pitches

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        K (int): Number of frequency points
        pitch_set (np.ndarray): Set of fundamental pitches
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        W (np.ndarray): Nonnegative matrix of size K x R with R = len(pitch_set)
    """
    R = len(pitch_set)
    W = np.zeros((K, R))
    for r in range(R):
        W[:, r] = template_pitch(K, pitch_set[r], freq_res, tol_pitch=tol_pitch, A4=A4)
        # W[:, r] = template_pitch_custom(K, pitch_set[r], freq_res, tol_pitch=tol_pitch, A4=A4)
    return W

def main():
    # folder = 'data'
    # filename = 'c-major-scale.ogg'
    # filename = 'Four Out Of Five.wav'
    
    folder = 'data/sight_reading_phone'
    # filename = 'Sight reading ex 45 RH 1.m4a'
    # filename = 'Sight reading ex 45 LH 1.m4a'
    # filename = 'Sight reading ex 45 together 1.m4a'
    filename = "Gigue à l'Angloise right hand.m4a"
    
    filepath = os.path.join(folder, filename)
    
    # Load audio from file (snippet).
    kw = {}
    # kw = {'duration': 16}
    x, Fs = librosa.load(filepath, **kw)
    # print(f'Sampling frequency [Hz]: {Fs:,}')

    # Plot waveform
    # fig, ax = plt.subplots()
    # librosa.display.waveshow(x, sr=Fs, ax=ax)
    # plt.show()

    # Estimate tuning.
    tuning = librosa.estimate_tuning(y=x, sr=Fs)
    A4 = librosa.tuning_to_A4(tuning)
    print(f'A4 estimate [Hz]: {A4}')

    # Isolate harmonic component.
    # See https://librosa.org/doc/main/auto_examples/plot_chroma.html
    x = librosa.effects.harmonic(y=x, margin=4)

    # STFT = Short-time Fourier transform.
    N_fft = 2048
    H_fft = 1024
    freq_max = 3000 # (Also needed for NMF plots.)
    X = librosa.stft(x, n_fft=N_fft, hop_length=H_fft)
    V = np.log(1 + np.abs(X))
    
    # todo: make CQT work as alternative to STFT.
    # X = librosa.cqt(y=x, sr=Fs, bins_per_octave=12*3, n_bins=7*12*3)
    # V = np.abs(X)
    # V = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    
    # Plot spectrogram.
    libfmp.b.plot_matrix(V, Fs=Fs/H_fft, Fs_F=N_fft/Fs)
    plt.ylim([0, freq_max])
    # plt.show()
    plt_show_max()

    # Initialise pitch templates with piano key frequencies and their harmonics.
    # If the tuning in the recording is not A4 = 440Hz, the results will be worse.
    # pitch_set = np.arange(21, 109) # 88 piano keys
    pitch_set = np.arange(50, 96) # Omit low keys.
    # pitch_set = np.arange(60, 73) # Pitch set: MIDI note numbers.
    # pitch_set = np.array([60, 62, 64, 65, 67, 69, 71, 72]) # MIDI note numbers for C major scale.
    K = V.shape[0]
    N = V.shape[1]
    R = pitch_set.shape[0]
    freq_res = Fs / N_fft
    # W_init = libfmp.c8.init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05)
    W_init = init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.0215, A4=A4)
    H_init = np.random.rand(R,N)

    # Plot one pitch template vector.
    # todo: compare with slice through spectrogram.
    # todo: use to define more representative pitch templates.
    # plt.figure()
    # tmp = V[:, 20]
    # tmp /= np.max(tmp)
    # plt.plot(tmp, label=f'measured')
    # plt.plot(W_init[:, 1], label='template')
    # plt.xlim([0, 400])
    # plt.legend()
    # plt_show_max()

    # plot_W(W_init)
    
    # todo: sort W by fundamental frequency, apply same permutation to H.
    # np.savetxt("W_init.csv", W_init, delimiter=",")
    
    # Simpler alternative: keep pitch templates fixed, solve V=W*H by least squares.
    # todo: improve pitch templates: Gaussian spread centred on fundamental frequency, width proportional to frequency.
    # todo: alternative pitch templates: slices from spectrogram.
    # todo: plot overlay pitch templates with corresponding note onsets in original FT.
    # todo: plot fit residuals.
    # todo: change axis labels from MIDI note numbers to note names.
    # H_lstsq, resid, rank, s = np.linalg.lstsq(W_init, V, rcond=None)
    # libfmp.c8.plot_nmf_factors(W_init, H_lstsq, W_init.dot(H_lstsq), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set, title_W='pitch templates', title_H='onsets/activations', title_V='Reconstructed spectrogram')
    # error = np.linalg.norm(resid)
    # print(f'Fit error: {error}')
    # print(H_lstsq.shape)

    # return # tmp
    
    # NMF
    W, H, V_approx, V_approx_err, H_W_error = libfmp.c8.nmf(V, R, W=W_init, H=H_init, L=100, norm=True)
    # plot_W(W)

    # Save W and H to file then reload, so that matrices are fixed for manual post-processing.
    # csvw = 'W.csv'
    # csvh = 'H.csv'
    # np.savetxt(csvw, W, delimiter=",")
    # np.savetxt(csvh, H, delimiter=",")
    # W = np.loadtxt(csvw, delimiter=',')
    # H = np.loadtxt(csvh, delimiter=',')

    # Manually post-process matrices.
    # todo: save to file (name to include source audio file name).
    # print(W.shape) # 1025 x 13
    # print(H.shape) # 13 x 217
    # W[:, 0] = np.zeros(np.shape(W[:, 0])) # todo: function to zero a vector.
    # H[0, :] = np.zeros(np.shape(H[0, :]))
    # sort_order = [0, 8, 1, 10, 12, 5, 6, 3, 2, 9, 11, 7, 4]
    # W = W[:, sort_order]
    # H = H[sort_order, :]

    # # Plot one pitch template vector.
    # plt.figure()
    # plt.plot(W[:, 1])
    # plt_show_max()

    # # Plot activation of one vector as time series.
    # plt.figure()
    # plt.plot(H[1, :])
    # plt_show_max()

    # # Combine vectors that represent the same note.
    # def combine_vectors(i, j):
    #     W[:, i] = 0.5*(W[:, i] + W[:, j])
    #     W[:, j] = np.zeros(np.shape(W[:, j]))
    #     H[i, :] += H[j, :]
    #     H[j, :] = np.zeros(np.shape(H[j, :]))
    
    # combine_vectors(1, 2)
    # combine_vectors(4, 5)
    # combine_vectors(8, 9)
    # combine_vectors(10, 11)

    # # Plot one pitch template vector.
    # plt.figure()
    # plt.plot(W[:, 1])
    # plt_show_max()
    
    # libfmp.c8.plot_nmf_factors(W_init, H_init, W_init.dot(H_init), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)
    libfmp.c8.plot_nmf_factors(W, H, V_approx, Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)
    
    # Inverse Fourier transform to reconstruct audio from spectrogram.
    # todo: use to improve fit quality.
    # todo: plot diff with original audio.
    # reconstructed = librosa.istft(V_approx, n_fft=N_fft, hop_length=H_fft)
    
    # Write out audio as 24bit PCM WAV.
    # sf.write('reconstructed.wav', reconstructed, Fs, subtype='PCM_24')

if __name__ == '__main__':
    main()
