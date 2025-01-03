import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import get_spectrogram_from_waveform, transcript_to_mel, dtw, align, mel_to_audio

class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, split, source, params):
        assert split in ("train", "valid", "test"), "invalid split"
        self.source = source.shuffle(seed=42) # for reproducibility
        self.base_pointer, self.limit_pointer = params[split]
        self.cache = dict()
        self.label_encoder = {
            'confused': 0,
            'default': 1,
            'emphasis': 2,
            'enunciated': 3,
            'essentials': 4,
            'happy': 5,
            'laughing': 6,
            'longform': 7,
            'sad': 8,
            'singing': 9,
            'whisper': 10
        }
        self.speaker_encoder = {
            'ex01': 0, 
            'ex02': 1,
            'ex03': 2,
            'ex04': 3
        }

    def __len__(self):
        return self.limit_pointer - self.base_pointer

    def __getitem__(self, idx):
        
        # 0. Preprocessing
        
        idx += self.base_pointer
        assert idx < self.limit_pointer, "index out of bounds"
        if idx in self.cache: return self.cache[idx] # memoisation
        
        item = self.source[idx]
        
        label = item["style"]
        speaker = item["speaker_id"]
        
        # 1. Obtain Mel Spectrograms
        
        data_mel_spectrogram = get_spectrogram_from_waveform(item["audio"]["array"], item["audio"]["sampling_rate"])
        ai_mel_spectrogram = transcript_to_mel(item["text"])
        
        data_mel_spectrogram = np.einsum("ij->ji", data_mel_spectrogram)
        ai_mel_spectrogram = np.einsum("ij->ji", ai_mel_spectrogram)
        
        assert ai_mel_spectrogram.shape[1] == 80
        assert data_mel_spectrogram.shape[1] == 80
        
        # 2. DTW
        dtw_cost, path = dtw(ai_mel_spectrogram, data_mel_spectrogram)
        aligned_to_ai_spectrogram = align(ai_mel_spectrogram, data_mel_spectrogram, path)
        
        assert aligned_to_ai_spectrogram.shape == ai_mel_spectrogram.shape, "DTW was not successful"
        assert aligned_to_ai_spectrogram.shape[1] == 80
        
        # 3. Duration modelling
        
        duration_arr = np.zeros(len(ai_mel_spectrogram))
        for i, (x, y) in enumerate(path):
            if i == 0:
                duration_arr[x] += 1
            else:
                xp, yp = path[i - 1]
                if yp == y:
                    duration_arr[xp] -= 1
                    duration_arr[x] += 1
                else:
                    duration_arr[x] += 1
        assert sum(duration_arr) == len(data_mel_spectrogram), "duration modelling not successful"
        
        # 4. Return AI Mel, aligned Data Mel, Emotion Label, Speaker Label, original Data Mel
        self.cache[idx] = {
            "ai_mel": torch.tensor(ai_mel_spectrogram), 
            "data_mel": torch.tensor(aligned_to_ai_spectrogram), 
            "label": torch.tensor([self.label_encoder[label]]), 
            "speaker": torch.tensor([self.speaker_encoder[speaker]]), 
            "original_data_mel": torch.tensor(data_mel_spectrogram),
            "sequence_length": torch.tensor([ai_mel_spectrogram.shape[0]]),
            "duration": torch.tensor(duration_arr),
            "text": item["text"],
            "data_audio": item["audio"]["array"],
            "data_sample_rate": item["audio"]["sampling_rate"],
            "ai_audio": mel_to_audio(ai_mel_spectrogram),
            "ai_sample_rate": 22050
        }
        return self.cache[idx]
    
    @staticmethod
    def collate(batch):
        
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        
        ai_mel = pad_sequence(
            [item["ai_mel"] for item in batch],
            batch_first=True, padding_value=np.nan
        )
        data_mel = pad_sequence(
            [item["data_mel"] for item in batch],
            batch_first=True, padding_value=np.nan
        )
        duration = pad_sequence(
            [item["duration"] for item in batch],
            batch_first=True, padding_value=np.nan
        )
        labels = torch.cat(tuple([item["label"] for item in batch]))
        sequence_lengths = torch.cat(tuple([item["sequence_length"] for item in batch]))
        mask = torch.all(torch.where(torch.isnan(ai_mel), torch.full(ai_mel.shape, True), torch.full(ai_mel.shape, False)), 2)
        mask_check = torch.all(torch.where(torch.isnan(data_mel), torch.full(data_mel.shape, True), torch.full(data_mel.shape, False)), 2)
        mask_double_check = torch.where(torch.isnan(duration), torch.full(duration.shape, True), torch.full(duration.shape, False))
        assert torch.equal(mask, mask_check), "mask is dubious"
        assert torch.equal(mask, mask_double_check), f"mask is dubious {mask.shape}, {mask_double_check.shape}"
        
        batch_size = len(batch)
        _, ai_mel_max_length, _ = ai_mel.shape
        assert ai_mel.shape == (batch_size, ai_mel_max_length, 80)
        assert data_mel.shape == ai_mel.shape
        assert duration.shape == ai_mel.shape[:2]
        assert sequence_lengths.shape == torch.Size([batch_size])
        assert torch.all(sequence_lengths > 0), "not all sequence lengths are positive"
        assert mask.shape == ai_mel.shape[:2]
        
        return {
            "ai_mel": ai_mel.to(device),
            "data_mel": data_mel.to(device), 
            "labels": labels.to(device),
            "sequence_length": sequence_lengths.to(device),
            "mask": mask.to(device),
            "duration": duration.to(device)
        }

class ProcessedMelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, split, data):
        assert split in ("train", "valid", "test"), "invalid split"
        m = {
            "train": "training_data",
            "valid": "validation_data",
            "test": "testing_data"
        }
        self.data = data
        self.split = m[split]
        
        import random
        random.seed(225) # reproducibility
        random.shuffle(self.data["training_data"])

    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx:int):
        assert idx >= 0 and idx < len(self), "Index error in ProcessedMelSpectrogramDataset"
        return self.data[self.split][idx]
    
    @staticmethod
    def collate(batch):
        
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        
        ai_mel = pad_sequence(
            [item["ai_mel"] for item in batch],
            batch_first=True, padding_value=np.nan
        )
        data_mel = pad_sequence(
            [item["data_mel"] for item in batch],
            batch_first=True, padding_value=np.nan
        )

        # duration = pad_sequence(
        #     [item["duration"] for item in batch],
        #     batch_first=True, padding_value=np.nan
        # )

        labels = torch.cat(tuple([item["label"] for item in batch]))
        # sequence_lengths = torch.cat(tuple([item["sequence_length"] for item in batch]))
        mask = torch.all(torch.where(torch.isnan(ai_mel), torch.full(ai_mel.shape, True), torch.full(ai_mel.shape, False)), 2)
        mask_check = torch.all(torch.where(torch.isnan(data_mel), torch.full(data_mel.shape, True), torch.full(data_mel.shape, False)), 2)
        # mask_double_check = torch.where(torch.isnan(duration), torch.full(duration.shape, True), torch.full(duration.shape, False))
        assert torch.equal(mask, mask_check), "mask is dubious"
        # assert torch.equal(mask, mask_double_check), f"mask is dubious {mask.shape}, {mask_double_check.shape}"

        ai_mel = pad_sequence(
            [(item["ai_mel"] - mu)/sig for item in batch],
            batch_first=True, padding_value=0.0
        )
        data_mel = pad_sequence(
            [(item["data_mel"] - mu_data)/sig_data for item in batch],
            batch_first=True, padding_value=0.0
        )
        # duration = pad_sequence(
        #     [item["duration"] for item in batch],
        #     batch_first=True, padding_value=0.0
        # )
        
        batch_size = len(batch)
        _, ai_mel_max_length, _ = ai_mel.shape
        assert ai_mel.shape == (batch_size, ai_mel_max_length, 80)
        assert data_mel.shape == ai_mel.shape
        # assert duration.shape == ai_mel.shape[:2]
        # assert sequence_lengths.shape == torch.Size([batch_size])
        # assert torch.all(sequence_lengths > 0), "not all sequence lengths are positive"
        assert mask.shape == ai_mel.shape[:2]
        
        return {
            "ai_mel": ai_mel.to(device),
            "data_mel": data_mel.to(device), 
            "labels": labels.to(device),
            # "sequence_length": sequence_lengths.to(device),
            "mask": mask.to(device) #,
            # "duration": duration.to(device)
        }