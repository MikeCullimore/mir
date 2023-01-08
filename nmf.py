"""
Non-negative matrix factorisation (NMF) to identift notes in audio (pitch and onset time).

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFSpecFac.html

todo:
Understand how to get clean activations for C major scale, then apply to more complex audio files.
"""

import os.path

import libfmp.b
import libfmp.c8
import librosa
import matplotlib.pyplot as plt
import numpy as np

def template_pitch(K, pitch, freq_res, tol_pitch=0.05):
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
    pitch_freq = 2**((pitch - 69) / 12) * 440
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    template = np.zeros(K)
    for m in range(1, max_order + 1):
        min_idx = max(0, int((1 - tol_pitch) * m * pitch_freq / freq_res))
        max_idx = min(K-1, int((1 + tol_pitch) * m * pitch_freq / freq_res))
        template[min_idx:max_idx+1] = 1 / m
    return template
        
def init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05):
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
        W[:, r] = template_pitch(K, pitch_set[r], freq_res, tol_pitch=tol_pitch)
    return W

def plot_nmf_factors(W, H, V, Fs, N_fft, H_fft, freq_max, label_pitch=None,
                     title_W='W', title_H='H', title_V='V', figsize=(13, 3)):
    """Plots the factore of an NMF-based spectral decomposition

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        W: Template matrix
        H: Activation matrix
        V: Reconstructed input matrix
        Fs: Sampling frequency
        N_fft: FFT length
        H_fft: Hopsize
        freq_max: Maximum frequency to be plotted
        label_pitch: Labels for the different pitches (Default value = None)
        title_W: Title for imshow of matrix W (Default value = 'W')
        title_H: Title for imshow of matrix H (Default value = 'H')
        title_V: Title for imshow of matrix V (Default value = 'V')
        figsize: Size of the figure (Default value = (13, 3))
    """
    R = W.shape[1]
    N = H.shape[1]
    # cmap = libfmp.b.compressed_gray_cmap(alpha=5)
    cmap = 'gray_r'
    dur_sec = (N-1) * H_fft / Fs
    if label_pitch is None:
        label_pitch = np.arange(R)

    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(W, cmap=cmap, origin='lower', aspect='auto', extent=[0, R, 0, Fs/2], interpolation='nearest')
    plt.ylim([0, freq_max])
    plt.colorbar()
    plt.xticks(np.arange(R) + 0.5, label_pitch)
    plt.xlabel('Pitch')
    plt.ylabel('Frequency (Hz)')
    plt.title(title_W)

    plt.subplot(132)
    plt.imshow(H, cmap=cmap, origin='lower', aspect='auto', extent=[0, dur_sec, 0, R], interpolation='nearest')
    plt.colorbar()
    plt.yticks(np.arange(R) + 0.5, label_pitch)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pitch')
    plt.title(title_H)

    plt.subplot(133)
    plt.imshow(V, cmap=cmap, origin='lower', aspect='auto', extent=[0, dur_sec, 0, Fs/2], interpolation='nearest')
    plt.ylim([0, freq_max])
    plt.colorbar()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title_V)

    plt.tight_layout()
    plt.show()

def main():
    folder = 'data'
    filename = 'c-major-scale.ogg'
    filepath = os.path.join(folder, filename)
    
    # Load audio from file (snippet).
    x, Fs = librosa.load(filepath, duration=16)
    print(f'Sampling frequency [Hz]: {Fs}')

    N_fft = 2048
    H_fft = 1024
    X = librosa.stft(x, n_fft=N_fft, hop_length=H_fft)
    V = np.log(1 + np.abs(X))
    #V = np.abs(X)
    freq_max = 2000

    libfmp.b.plot_matrix(V, Fs=Fs/H_fft, Fs_F=N_fft/Fs)
    plt.ylim([0, freq_max])
    plt.show()

    # Pitch set: MIDI note numbers.
    pitch_set = np.arange(40, 53)

    K = V.shape[0]
    N = V.shape[1]
    R = pitch_set.shape[0]
    freq_res = Fs / N_fft
    W_init = init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05)
    H_init = np.random.rand(R,N)

    W, H, V_approx, V_approx_err, H_W_error = libfmp.c8.nmf(V, R, W=W_init, H=H_init, L=100, norm=True)
    plot_nmf_factors(W_init, H_init, W_init.dot(H_init), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)
    plot_nmf_factors(W, H, W.dot(H), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)

if __name__ == '__main__':
    main()
