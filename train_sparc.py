import torch
import torch.nn as nn
import torch.nn.functional as F

from model_sparc import TransformerSparcEmotionModel

class ProcessedSparcSpectrogramDataset(torch.utils.data.Dataset):
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
        assert idx >= 0 and idx < len(self), "Index error in ProcessedSparcSpectrogramDataset"
        return self.data[self.split][idx]
    
    @staticmethod
    def collate(batch):
        
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        
        ai_mel = pad_sequence(
            [item["ai_sparc"] for item in batch],
            batch_first=True, padding_value=np.nan
        )
        data_mel = pad_sequence(
            [item["data_sparc"] for item in batch],
            batch_first=True, padding_value=np.nan
        )

        labels = torch.cat(tuple([item["label"] for item in batch]))
        mask = torch.all(torch.where(torch.isnan(ai_mel), torch.full(ai_mel.shape, True), torch.full(ai_mel.shape, False)), 2)
        mask_check = torch.all(torch.where(torch.isnan(data_mel), torch.full(data_mel.shape, True), torch.full(data_mel.shape, False)), 2)
        assert torch.equal(mask, mask_check), "mask is dubious"
        ai_mel = pad_sequence(
            [item["ai_sparc"] for item in batch],
            batch_first=True, padding_value=0.0
        )
        data_mel = pad_sequence(
            [item["data_sparc"] for item in batch],
            batch_first=True, padding_value=0.0
        )

        batch_size = len(batch)
        _, ai_mel_max_length, _ = ai_mel.shape
        assert ai_mel.shape == (batch_size, ai_mel_max_length, 12) # SPARC has 12 features
        assert data_mel.shape == ai_mel.shape
        assert mask.shape == ai_mel.shape[:2]
        
        return {
            "ai_sparc": ai_mel.to(device),
            "data_sparc": data_mel.to(device), 
            "labels": labels.to(device),
            "mask": mask.to(device)
        }

transformer_encoder_model = TransformerSparcEmotionModel(d_model=512, num_encoder_layers=8, dropout=0.1)
transformer_encoder_model.to(device)

# Training
loss_curve = []
validation_curve = []

def train_processed(model, data, num_epochs, batch_size, model_file,
          learning_rate=8e-4, loss_curve=[], validation_curve=[], best_metric=None):
    training_dataset = ProcessedSparcSpectrogramDataset("train", data)
    validation_dataset = ProcessedSparcSpectrogramDataset("valid", data)
    
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
    
    for epoch in tqdm.notebook.trange(num_epochs, desc="training", unit="epoch"):
        logging.info(f"=== EPOCH {epoch + 1}")
        with tqdm.notebook.tqdm(
            data_loader,
            desc="epoch {}".format(epoch + 1),
            unit="batch",
            total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            total_num = 0
            for i, batch in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                total_loss += loss.item()
                total_num += batch["ai_sparc"].size(0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / total_num)
            validation_metric = model.get_validation_metric(validation_dataset, batch_size=batch_size)
            validation_curve.append(validation_metric.item())
            loss_curve.append(total_loss/total_num)
            batch_iterator.set_postfix(
                mean_loss=total_loss / total_num,
                validation_metric=validation_metric
            )
            print(f"epoch={epoch + 1}; training={total_loss / total_num}; validation={validation_metric}")
            logging.info(f"epoch={epoch + 1}; training={total_loss / total_num}; validation={validation_metric}")
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

train_processed(
    transformer_encoder_model, 
    dataset, 
    num_epochs=40, 
    batch_size=64,
    model_file="sparc_transformer_encoder_model_speaker_4.pt", 
    learning_rate=5e-4, 
    loss_curve=loss_curve, 
    validation_curve=validation_curve,
    best_metric=None
)

plt.figure(figsize=(15,10))
plt.plot(np.arange(len(loss_curve)), np.array(loss_curve), label="training loss")
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.grid()
plt.show()

plt.figure(figsize=(15,10))
plt.plot(np.arange(len(validation_curve)), validation_curve, label="validation loss")
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Loss Curve')
plt.grid()
plt.show()

fig, ax1 = plt.subplots(figsize=(15, 10))

ax1.plot(np.arange(len(loss_curve)), np.array(loss_curve), 'b-', label='train')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (train)', color='b') 
ax1.tick_params(axis='y', labelcolor='b')  # Color of y-ticks

ax2 = ax1.twinx()

ax2.plot(np.arange(len(validation_curve)), validation_curve, 'g-', label='valid')
ax2.set_ylabel('Loss (valid)', color='g')  # Label for the second y-axis
ax2.tick_params(axis='y', labelcolor='g')  # Color of y-ticks

plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()

print(loss_curve)
print(validation_curve)

def predict_processed(model, data, num_limit=10):

    model.eval()

    test_dataset = ProcessedSparcSpectrogramDataset("test", data)

    data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, collate_fn=test_dataset.collate
    )
    
    with tqdm.notebook.tqdm(
        data_loader,
        total=len(data_loader)) as batch_iterator:
        model.eval()

        for i, batch in enumerate(batch_iterator, start=1):
            if i > num_limit: break
            _, seq_length, n_mels = batch["ai_sparc"].shape
            assert n_mels == 12 # sparc has 
            pred = model.transform(batch)
            assert pred.shape == (1, seq_length, n_mels)
            assert pred.squeeze().shape == (seq_length, n_mels)

            wav_pred = coder.decode(
                ema=pred.squeeze().detach().cpu().numpy(), 
                pitch=test_dataset[i]["data_sparc_pitch"].numpy(), 
                loudness=test_dataset[i]["data_sparc_loudness"].numpy(), 
                spk_emb=test_dataset[i]["data_sparc_spk_emb"].numpy()
            )

            wav_human = coder.decode(
                ema=test_dataset[i]["data_sparc"].numpy(), 
                pitch=test_dataset[i]["data_sparc_pitch"].numpy(), 
                loudness=test_dataset[i]["data_sparc_loudness"].numpy(), 
                spk_emb=test_dataset[i]["data_sparc_spk_emb"].numpy()
            )

            wav_ai = coder.decode(
                ema=test_dataset[i]["ai_sparc"].numpy(), 
                pitch=test_dataset[i]["ai_sparc_pitch"].numpy(), 
                loudness=test_dataset[i]["ai_sparc_loudness"].numpy(), 
                spk_emb=test_dataset[i]["ai_sparc_spk_emb"].numpy()
            )

            torchaudio.save(f"test{i}_pred.wav", torch.tensor(wav_pred).unsqueeze(0), coder.sr)
            torchaudio.save(f"test{i}_actual.wav", torch.tensor(wav_human).unsqueeze(0), coder.sr)
            torchaudio.save(f"test{i}_ai.wav", torch.tensor(wav_ai).unsqueeze(0), coder.sr)

predict_processed(transformer_encoder_model, dataset)

waveform_pred, sample_rate_pred = torchaudio.load("/kaggle/working/test7_pred.wav")
ipd.display(ipd.Audio(waveform_pred, rate=sample_rate_pred))

view_spectrogram(get_spectrogram_from_waveform(waveform_pred, sample_rate_pred), title="Mel Spectrogram (Pred)")
