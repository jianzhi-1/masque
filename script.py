import numpy as np
import torch
from data_processing import dtw, align_mfcc
from data_processing_adv import get_spectrogram, view_spectrogram, spectrogram_to_waveform

spect_ai = get_spectrogram("./data/ai_sample.wav")
spect_data = get_spectrogram("./data/data_sample.wav")

dtw_cost, path = dtw(np.einsum("ij->ji", spect_ai), np.einsum("ij->ji", spect_data))
aligned_spect_data = align_mfcc(np.einsum("ij->ji", spect_ai), np.einsum("ij->ji", spect_data), path)
actual_spect_data = np.einsum("ij->ji", aligned_spect_data)
spectrogram_to_waveform(torch.tensor(actual_spect_data), "./data/dtw_sample.wav")