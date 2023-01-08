"""
Non-negative matrix factorisation (NMF) to identift notes in audio (pitch and onset time).

Aim is to generate a piano roll plot for the C major scale (notes/pitches/frequencies vs time).

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S3_NMFSpecFac.html

todo:
Understand how to get clean activations for C major scale, then apply to more complex audio files.
    Sort pitches by fundamental frequency (then apply same permutation to activations).
Possible to keep pitch templates fixed and just optimise activations?
Mask onsets to ignore broadband signal.
Reconstruct V with the wideband signal in W discarded (white noise at onset).
"""

import os.path

import libfmp.b
import libfmp.c8
import librosa
import matplotlib.pyplot as plt
import numpy as np

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
    pitch_freq = 2**((pitch - 69) / 12) * 446 # todo: optimise automatically for minimum V error.
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

def main():
    folder = 'data'
    filename = 'c-major-scale.ogg'
    filepath = os.path.join(folder, filename)
    
    # Load audio from file (snippet).
    x, Fs = librosa.load(filepath, duration=16)
    # print(f'Sampling frequency [Hz]: {Fs:,}')

    # STFT = Short-time Fourier transform.
    N_fft = 2048
    H_fft = 1024
    X = librosa.stft(x, n_fft=N_fft, hop_length=H_fft)
    V = np.log(1 + np.abs(X))
    
    # Plot Fourier transform.
    libfmp.b.plot_matrix(V, Fs=Fs/H_fft, Fs_F=N_fft/Fs)
    freq_max = 2000
    plt.ylim([0, freq_max])
    # plt.show()
    plt_show_max()

    # Initialise pitch templates with piano key frequencies and their harmonics.
    # If the tuning in the recording is not A4 = 440Hz, the results will be worse.
    pitch_set = np.arange(40, 53) # Pitch set: MIDI note numbers.
    K = V.shape[0]
    N = V.shape[1]
    R = pitch_set.shape[0]
    freq_res = Fs / N_fft
    # W_init = libfmp.c8.init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05)
    W_init = init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05) # todo: try varying A4 in template_pitch above to get better result.
    H_init = np.random.rand(R,N)

    # plot_W(W_init)
    
    # todo: sort W by fundamental frequency, apply same permutation to H.
    # np.savetxt("W_init.csv", W_init, delimiter=",")
    
    # todo: simpler alternative: keep pitch templates fixed, solve V=W*H by least squares. Doesn't work: why?
    # todo: improve pitch templates: Gaussian spread centred on fundamental frequency, width proportional to frequency.
    # todo: plot overlay pitch templates with corresponding note onsets in original FT.
    # todo: shift tuning to match frequencies (get result from libfmp method that does this automatically?).
    # todo: play back reconstructed V: any clues?
    # todo: plot residuals V - W_init.dot(H_lstsq).
    # H_lstsq, resid, rank, s = np.linalg.lstsq(W_init, V, rcond=None)
    # print(H_lstsq.shape)
    # libfmp.c8.plot_nmf_factors(W_init, H_lstsq, W_init.dot(H_lstsq), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)
    
    W, H, V_approx, V_approx_err, H_W_error = libfmp.c8.nmf(V, R, W=W_init, H=H_init, L=100, norm=True)
    # plot_W(W)
    
    # libfmp.c8.plot_nmf_factors(W_init, H_init, W_init.dot(H_init), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)
    libfmp.c8.plot_nmf_factors(W, H, W.dot(H), Fs, N_fft, H_fft, freq_max, label_pitch=pitch_set)
    
    # np.savetxt("W.csv", W, delimiter=",")

if __name__ == '__main__':
    main()
