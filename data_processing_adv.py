import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import matplotlib.pyplot as plt
import librosa
import torch

# Load a pretrained HIFIGAN Vocoder
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

def view_spectrogram(spectrogram):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.numpy()
    print(spectrogram.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram from Tacotron 2')
    plt.tight_layout()
    plt.show()

def get_spectrogram(file_name):

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

def spectrogram_to_waveform(spectrogram, save_file_name):
    waveforms = hifi_gan.decode_batch(spectrogram) # spectrogram to waveform
    torchaudio.save(save_file_name, waveforms.squeeze(1), 22050)

def get_reconstructed_sample(file_name, save_file_name):

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

    waveforms = hifi_gan.decode_batch(spectrogram) # spectrogram to waveform

    torchaudio.save(save_file_name, waveforms.squeeze(1), 22050)

def transcript_to_audio(sentence, save_file_name):
    
    mel_output, mel_length, alignment = tacotron2.encode_text(sentence)
    # 1. Mel spectrogram with properties in the Tacotron paper (or see get_reconstructed_sample)
    #    Shape = (batch_size, n_mels=80, Mel_length + 1); Mel_length proportional to length of sequence
    # 2. Mel_length = mel_output.shape[2] - 1
    # 3. Alignment
    #    Shape = (batch_size, Mel_length, Token_length) where Token_length is from tacotron2.text_to_seq(txt)

    waveforms = hifi_gan.decode_batch(mel_output) # spectrogram to waveform

    torchaudio.save(save_file_name, waveforms.squeeze(1), 22050)