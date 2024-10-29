import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from data_processing_adv import get_spectrogram_from_waveform, transcript_to_mel
from data_processing import dtw, align
import tqdm.notebook

class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, split, source):
        assert split in ("train", "valid", "test"), "invalid split"
        params = {
            "train": (0, 7000),
            "valid": (7000, 10000),
            "test": (10000, 11615)
        }
        self.source = source
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
        
        if idx in self.cache: return self.cache
        idx += self.base_pointer
        assert idx < self.limit_pointer, "index out of bounds"
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
        
        # 3. Return AI Mel, aligned Data Mel, Emotion Label, Speaker Label, original Data Mel
        return {
            "ai_mel": torch.tensor(ai_mel_spectrogram), 
            "data_mel": torch.tensor(aligned_to_ai_spectrogram), 
            "label": torch.tensor([self.label_encoder[label]]), 
            "speaker": torch.tensor([self.speaker_encoder[speaker]]), 
            "original_data_mel": data_mel_spectrogram
        }
    
    @staticmethod
    def collate(batch):
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        ai_mel = pad_sequence(
            [item["ai_mel"] for item in batch],
            batch_first=True, padding_value=-np.inf)
        data_mel = pad_sequence(
            [item['data_mel'] for item in batch],
            batch_first=True, padding_value=-np.inf)
        labels = torch.cat(tuple([item['label'] for item in batch]))
        return {
            'ai_mel': ai_mel.to(device), 
            'data_mel': data_mel.to(device), 
            'labels': labels.to(device)
        }

def train(model, num_epochs, batch_size, model_file, source,
          learning_rate=8e-4, dataset_cls=MelSpectrogramDataset):
    dataset = dataset_cls('train', source)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.02,  # Warm up for 2% of the total training time
    )
    best_metric = 0.0
    for epoch in tqdm.notebook.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.notebook.tqdm(
            data_loader,
            desc="epoch {}".format(epoch + 1),
            unit="batch",
            total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
            validation_metric = model.get_validation_metric()
            batch_iterator.set_postfix(
                mean_loss=total_loss / i,
                validation_metric=validation_metric
            )
            if validation_metric > best_metric:
                print(
                    "Obtained a new best validation metric of {:.3f}, saving model "
                    "checkpoint to {}...".format(validation_metric, model_file)
                )
                torch.save(model.state_dict(), model_file)
                best_metric = validation_metric
    print("Reloading best model checkpoint from {}...".format(model_file))
    model.load_state_dict(torch.load(model_file))

if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("ylacombe/expresso")
    dataset = MelSpectrogramDataset("train", ds["train"])
    print(dataset[0]) # data visualisation
    print(dataset[1234]) # data visualisation
    print(dataset[5000]) # data visualisation
    print(dataset[11000]) # out of bound error
