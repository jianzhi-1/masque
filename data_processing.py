import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

EPS = 1e-6

def equals(a, b):
    return abs(a - b) < EPS

def dtw(a, b):
    n, m = a.shape[0], b.shape[0]
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])  # Euclidean distance
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Insertion
                                           dtw_matrix[i, j - 1],    # Deletion
                                           dtw_matrix[i - 1, j - 1]) # Match

    # Backtrack to find the optimal path
    i, j = n, m
    path = []

    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i > 0 and j > 0:
            if equals(dtw_matrix[i, j], dtw_matrix[i - 1, j - 1] + np.linalg.norm(a[i - 1] - b[j - 1])):
                i -= 1
                j -= 1
            elif equals(dtw_matrix[i, j], dtw_matrix[i - 1, j] + np.linalg.norm(a[i - 1] - b[j - 1])):
                i -= 1
            else:
                j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return dtw_matrix[n, m], path

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def align(signal_a, signal_b, path):
    aligned_b = np.zeros_like(signal_a)

    for idx_a, idx_b in path:
        aligned_b[idx_a] = signal_b[idx_b]

    return aligned_b

def main(audio_file_1, audio_file_2):
    
    # 0. Load audio files
    audio_a, sr_a = load_audio(audio_file_1)
    audio_b, sr_b = load_audio(audio_file_2)

    # 1. Extract MFCC features
    mfcc_a = librosa.feature.mfcc(y=audio_a, sr=sr_a, n_mfcc=13).T
    mfcc_b = librosa.feature.mfcc(y=audio_b, sr=sr_b, n_mfcc=13).T

    # 2. Normalise MFCC features
    mfcc_a_normalised = (mfcc_a - np.mean(mfcc_a, axis=0))/(np.std(mfcc_a, axis=0))
    mfcc_b_normalised = (mfcc_b - np.mean(mfcc_b, axis=0))/(np.std(mfcc_b, axis=0))

    # 3. Perform DTW
    _, path = dtw(mfcc_a_normalised, mfcc_b_normalised)

    # 4. Align audio_b using DTW path
    mfcc_b_aligned = align_mfcc(mfcc_a_normalised, mfcc_b, path)
    audio_b_aligned = librosa.feature.inverse.mfcc_to_audio(np.einsum("ij->ji", mfcc_b_aligned))

    # 5. Export
    sf.write(f'./{audio_file_2}_aligned.wav', audio_b_aligned, sr_b)
    print(f"Aligned audio saved as '{audio_file_2}_aligned.wav'.")

    return

def naive_cut(audio_file_1, audio_file_2):
    audio_a, _ = load_audio(audio_file_1)
    audio_b, sr_b = load_audio(audio_file_2)
    sf.write('./audio_b_cut.wav', audio_b[:len(audio_a)], sr_b)
    print("Aligned audio saved as 'audio_b_cut.wav'.")

def naive_speed(audio_file_1, audio_file_2):
    audio_a, sr_a = load_audio(audio_file_1)
    audio_b, _ = load_audio(audio_file_2)
    sf.write('./audio_b_speed.wav', audio_b, int(sr_a*len(audio_b)/len(audio_a)))
    print("Aligned audio saved as 'audio_b_speed.wav'.")