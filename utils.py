import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np

import soundfile as sf
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

import torch
import torchaudio

### Constants
EPS = 1e-6

### Pre-trained models
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts") # Tacotron2 as TTS
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder") # HiFi-GAN Vocoder

def equals(a, b):
    return abs(a - b) < EPS

### Research infrastructure functions

def load_audio(file_path):
    """
    Obtain audio signal y from file_path
    """
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def view_spectrogram(spectrogram, title="Mel Spectrogram", n_mels=80):
    """
    Function to plot a spectrogram
    """
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = spectrogram.numpy()
    if spectrogram.shape[0] != 80:
        spectrogram = np.einsum("ij->ji", spectrogram)
    assert spectrogram.shape[0] == n_mels, f"spectrogram shape {spectrogram.shape} != ({n_mels}, seq_length)"
    print(spectrogram.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_spectrogram(file_name):
    """
    Obtain spectrogram from a .wav file
    """
    signal, rate = torchaudio.load(file_name)
    signal = torchaudio.functional.resample(signal, orig_freq=rate, new_freq=22050)

    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=None,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    return spectrogram

def spectrogram_to_waveform(spectrogram, save_file_name=None):
    """
    Converts spectrogram to audio signal using Hi-Fi GAN
    """
    waveforms = hifi_gan.decode_batch(spectrogram) # spectrogram to waveform
    if save_file_name is not None:
        torchaudio.save(save_file_name, waveforms.squeeze(1), 22050)
    return waveforms.squeeze()

def get_spectrogram_from_waveform(signal, rate):
    """
    Obtain a spectrogram from audio signal
    """
    if isinstance(signal, np.ndarray):
        signal = torch.tensor(signal, dtype=torch.float32)
    
    signal = torchaudio.functional.resample(signal, orig_freq=rate, new_freq=22050)

    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=None,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    return spectrogram

def get_reconstructed_sample(file_name, save_file_name, new_freq=22050):
    """
    Obtain signal with different sampling rate, standard new_freq is 22050 to be consistent with Hi-Fi GAN
    """
    signal, rate = torchaudio.load(file_name)
    signal = torchaudio.functional.resample(signal, orig_freq=rate, new_freq=new_freq)

    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=None,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    waveforms = hifi_gan.decode_batch(spectrogram) # spectrogram to waveform

    torchaudio.save(save_file_name, waveforms.squeeze(1), new_freq)

def transcript_to_audio(sentence, save_file_name):
    """
    Converts a text transcript to audio signal via Tacotron2
    """
    mel_output, _, _ = tacotron2.encode_text(sentence) # mel_output, mel_length, alignment

    # 1. Mel spectrogram with properties in the Tacotron paper (or see get_reconstructed_sample)
    #    Shape = (batch_size, n_mels=80, Mel_length + 1); Mel_length proportional to length of sequence
    # 2. Mel_length = mel_output.shape[2] - 1
    # 3. Alignment
    #    Shape = (batch_size, Mel_length, Token_length) where Token_length is from tacotron2.text_to_seq(txt)

    waveforms = hifi_gan.decode_batch(mel_output) # spectrogram to waveform

    torchaudio.save(save_file_name, waveforms.squeeze(1), 22050)

def transcript_to_mel(sentence):
    """
    Converts a text transcript to Mel spectrogram
    """
    mel_output, _, _ = tacotron2.encode_text(sentence) # mel_output, mel_length, alignment
    return mel_output.squeeze() # remove the batch dimension

def mel_to_audio(mel_output, save_file_name=None, display=False, new_freq=22050):
    """
    Converts a Mel spectrogram into audio signal via Hi-Fi GAN
    """
    if isinstance(mel_output, np.ndarray):
        mel_output = torch.tensor(mel_output)
    if mel_output.shape[0] != 80:
        mel_output = torch.einsum("ij->ji", mel_output)
    waveforms = hifi_gan.decode_batch(mel_output) # spectrogram to waveform
    if save_file_name is not None: torchaudio.save(save_file_name, waveforms.squeeze(1), new_freq)
    if display: return ipd.Audio(waveforms, rate=new_freq)
    return waveforms

def sample_mel(dataset, idx:int):
    """
    A utility function to sample a Mel spectrogram from the dataset
    Prints the Mel spectrogram array and converts the monotonous and expressive Mel spectrograms into audios for playing
    """
    print(dataset[idx])
    mel_to_audio(torch.einsum("ij->ji", dataset[idx]["data_mel"]), f"sample_{idx}.wav")
    mel_to_audio(torch.einsum("ij->ji", torch.tensor(dataset[idx]["original_data_mel"])), f"sample_{idx}_original.wav")

### Algorithmic functions

def dtw(a, b):
    """
    Dynamic time warping algorithm between two 2D matrices with Euclidean norm
    """
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

def align(signal_a, signal_b, path):
    """
    Given a path from DTW algorithm, ensures signal_b eventually has the same shape as signal_a
    """
    aligned_b = np.zeros_like(signal_a)

    for idx_a, idx_b in path:
        aligned_b[idx_a] = signal_b[idx_b]

    return aligned_b

def dtw_files(audio_file_1, audio_file_2):
    """
    Given two .wav files, perform DTW on them.
    """
    
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
    mfcc_b_aligned = align(mfcc_a_normalised, mfcc_b, path)
    audio_b_aligned = librosa.feature.inverse.mfcc_to_audio(np.einsum("ij->ji", mfcc_b_aligned))

    # 5. Export
    sf.write(f'./{audio_file_2}_aligned.wav', audio_b_aligned, sr_b)
    print(f"Aligned audio saved as '{audio_file_2}_aligned.wav'.")