import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from datasets import load_dataset # Expresso dataset
import tqdm.notebook