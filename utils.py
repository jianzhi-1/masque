import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from data_processing_adv import get_spectrogram_from_waveform, transcript_to_mel, mel_to_audio
from data_processing import dtw, align
import tqdm.notebook
import logging
import matplotlib.pyplot as plt

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

def cache_dataset(source, params, save_file_name="alldata.pth"):
    train_dataset = MelSpectrogramDataset("train", source, params)
    train_ls = [train_dataset[i] for i in range(len(train_dataset))]
    print("finished processing train")
    validation_dataset = MelSpectrogramDataset("valid", source, params)
    valid_ls = [validation_dataset[i] for i in range(len(validation_dataset))]
    print("finished processing valid")
    test_dataset = MelSpectrogramDataset("valid", source, params)
    test_ls = [test_dataset[i] for i in range(len(test_dataset))]
    print("finished processing test")
    torch.save({
        "training_data": train_ls, 
        "validation_data": valid_ls,
        "testing_data": test_ls
    }, save_file_name)

def train(model, source, params, num_epochs, batch_size, model_file,
          learning_rate=8e-4, dataset_cls=MelSpectrogramDataset):
    dataset = dataset_cls(
        'train', 
        source,
        params
    )
    validation_dataset = dataset_cls(
        'valid', 
        source,
        params
    )
    
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
    best_metric = None
    
    validation_curve = []
    total_loss_curve = []
    for epoch in tqdm.notebook.trange(num_epochs, desc="training", unit="epoch"):
        logging.info(f"=== EPOCH {epoch + 1}")
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
                if i % 20 == 0:
                    print(f"epoch={epoch + 1}; batch={i}; loss={loss.item()}; total_loss={total_loss}")
                logging.info(f"epoch={epoch + 1}; batch={i}; loss={loss.item()}; total_loss={total_loss}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
            total_loss_curve.append(total_loss)
            validation_metric = model.get_validation_metric(validation_dataset)
            validation_curve.append(validation_metric)
            batch_iterator.set_postfix(
                mean_loss=total_loss / i,
                validation_metric=validation_metric
            )
            print(f"epoch={epoch + 1}; validation={validation_metric}")
            logging.info(f"epoch={epoch + 1}; validation={validation_metric}")
            if best_metric is None or validation_metric < best_metric:
                print(
                    "Obtained a new best validation metric of {:.3f}, saving model "
                    "checkpoint to {}...".format(validation_metric, model_file)
                )
                torch.save(model.state_dict(), model_file)
                best_metric = validation_metric
        logging.info(f"=== END OF EPOCH {epoch + 1}")
        
    print("Reloading best model checkpoint from {}...".format(model_file))
    model.load_state_dict(torch.load(model_file))
    return validation_curve, total_loss_curve

def train_processed(model, data, num_epochs, batch_size, model_file,
          learning_rate=8e-4, loss_curve=[], validation_curve=[]):
    training_dataset = ProcessedMelSpectrogramDataset("train", data)
    validation_dataset = ProcessedMelSpectrogramDataset("valid", data)
    
    data_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, collate_fn=training_dataset.collate
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
    best_metric = None
    
    for epoch in tqdm.notebook.trange(num_epochs, desc="training", unit="epoch"):
        logging.info(f"=== EPOCH {epoch + 1}")
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
                loss_curve.append(loss.item())
                if i % 10 == 0:
                    print(f"epoch={epoch + 1}; batch={i}; loss={loss.item()}; total_loss={total_loss}")
                logging.info(f"epoch={epoch + 1}; batch={i}; loss={loss.item()}; total_loss={total_loss}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
            validation_metric = model.get_validation_metric(validation_dataset)
            validation_curve.append(validation_metric.item())
            batch_iterator.set_postfix(
                mean_loss=total_loss / i,
                validation_metric=validation_metric
            )
            print(f"epoch={epoch + 1}; validation={validation_metric}")
            logging.info(f"epoch={epoch + 1}; validation={validation_metric}")
            if best_metric is None or validation_metric < best_metric:
                print(
                    "Obtained a new best validation metric of {:.3f}, saving model "
                    "checkpoint to {}...".format(validation_metric, model_file)
                )
                torch.save(model.state_dict(), model_file)
                best_metric = validation_metric
        logging.info(f"=== END OF EPOCH {epoch + 1}")
    print("Reloading best model checkpoint from {}...".format(model_file))
    model.load_state_dict(torch.load(model_file))

def predict(model, source, params, dataset_cls=MelSpectrogramDataset, num_limit=10):

    model.eval()
    
    test_dataset = dataset_cls(
        'test', 
        source,
        params
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, collate_fn=test_dataset.collate
    )
    
    with tqdm.notebook.tqdm(
        data_loader,
        total=len(data_loader)) as batch_iterator:
        model.eval()

        for i, batch in enumerate(batch_iterator, start=1):
            if i > num_limit: break
            _, seq_length, n_mels = batch["ai_mel"].shape
            assert n_mels == 80
            pred = model.transform(batch)
            assert pred.shape == (1, seq_length, n_mels)
            assert pred.squeeze().shape == (seq_length, n_mels)
            mel_to_audio(pred.squeeze(), f"test{i}_pred.wav")
            mel_to_audio(batch["data_mel"].squeeze(), f"test{i}_actual.wav")

def predict_processed(model, data, num_limit=10):

    model.eval()

    test_dataset = ProcessedMelSpectrogramDataset("test", data)

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, collate_fn=test_dataset.collate
    )
    
    with tqdm.notebook.tqdm(
        data_loader,
        total=len(data_loader)) as batch_iterator:
        model.eval()

        for i, batch in enumerate(batch_iterator, start=1):
            if i > num_limit: break
            _, seq_length, n_mels = batch["ai_mel"].shape
            assert n_mels == 80
            pred = model.transform(batch)
            assert pred.shape == (1, seq_length, n_mels)
            assert pred.squeeze().shape == (seq_length, n_mels)
            mel_to_audio(pred.squeeze(), f"test{i}_pred.wav")
            mel_to_audio(batch["data_mel"].squeeze(), f"test{i}_actual.wav")

if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("ylacombe/expresso")
    dataset = MelSpectrogramDataset("train", ds["train"])
    print(dataset[0]) # data visualisation
    print(dataset[1234]) # data visualisation
    print(dataset[5000]) # data visualisation
    print(dataset[11000]) # out of bound error

    # prediction
    transformer_predict_model = TransformerEmotionModel()
    transformer_predict_model.load_state_dict(torch.load("/kaggle/working/transformer_encoder_model.pt"))
    transformer_predict_model.to(device)
    predict(
        transformer_predict_model, 
        filtered_ds, 
        params = {
            "train": (0, 2000),
            "valid": (2000, 2450),
            "test": (2450, 2903)
        },
        num_limit=10
    )

    # Caching
    cache_dataset(ds["train"], {
        "train": (0, 7000),
        "valid": (7000, 10000),
        "test": (10000, 11615)
    }, save_file_name="alldata.pth")

    filtered_ds = ds["train"].filter(lambda x: x['speaker_id'] == "ex01")
    cache_dataset(filtered_ds, {
        "train": (0, 2000),
        "valid": (2000, 2450),
        "test": (2450, 2903)
    }, save_file_name="speaker1.pth")

    # Processed dataset
    processed_dataset = torch.load("/kaggle/input/speaker1-processed/speaker1.pth")

    # Training
    loss_curve = []
    validation_curve = []
    train_processed(
        transformer_encoder_model, 
        processed_dataset, 
        num_epochs=5, 
        batch_size=64,
        model_file="transformer_encoder_model_speaker_one.pt", 
        learning_rate=0.1, 
        loss_curve=loss_curve, 
        validation_curve=validation_curve
    )

    # Visualisation
    plt.plot(np.arange(len(loss_curve)), np.log(np.array(loss_curve)))
    plt.plot(np.arange(len(validation_curve)), np.log(np.array([x.item() for x in validation_curve])))

    # Prediction
    predict_processed(
        transformer_predict_model, 
        processed_dataset,
        num_limit=10
    )

