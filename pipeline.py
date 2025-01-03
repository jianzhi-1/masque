
from datasets import load_dataset
from dataset_class import MelSpectrogramDataset, ProcessedMelSpectrogramDataset
import IPython.display as ipd
import logging
import matplotlib.pyplot as plt
from models import TransformerSparcEmotionModel
import numpy as np

import torch
import torchaudio
import tqdm.notebook

from utils import mel_to_audio, view_spectrogram, get_spectrogram_from_waveform

def trim_data(speaker_id):

    dataset = load_dataset("ylacombe/expresso")

    filtered_dataset = dataset["train"].filter(lambda x: x["speaker_id"] == f"ex0{speaker_id}")

    train_dataset = MelSpectrogramDataset("train", filtered_dataset, {
            "train": (0, 2000),
            "valid": (2000, 2450),
            "test": (2450, 2902)
    })

    validation_dataset = MelSpectrogramDataset("valid", filtered_dataset, {
            "train": (0, 2000),
            "valid": (2000, 2450),
            "test": (2450, 2902)
    })

    test_dataset = MelSpectrogramDataset("test", filtered_dataset, {
            "train": (0, 2000),
            "valid": (2000, 2450),
            "test": (2450, 2902)
    })

    def cache_dataset(save_file_name="alldata.pth"):
        train_ls = []
        for i in tqdm.tqdm(range(len(train_dataset))):
            train_ls.append(train_dataset[i])
            #         if (i + 1) % 1000 == 0:
            #             torch.save({"training_data": train_ls}, f"train_{i}_{save_file_name}")
        torch.save({"training_data": train_ls}, f"train_{save_file_name}")
        print("finished train")
        
        valid_ls = []
        for i in tqdm.tqdm(range(len(validation_dataset))):
            valid_ls.append(validation_dataset[i])
            #         if (i + 1) % 1000 == 0:
            #             torch.save({"validation_data": valid_ls}, f"valid_{i}_{save_file_name}")
        torch.save({"validation_data": valid_ls}, f"valid_{save_file_name}")
        print("finished valid")
        
        test_ls = []
        for i in tqdm.tqdm(range(len(test_dataset))):
            test_ls.append(test_dataset[i])
            #         if (i + 1) % 1000 == 0:
            #             torch.save({"testing_data": test_ls}, f"test_{i}_{save_file_name}")
        torch.save({"testing_data": test_ls}, f"test_{save_file_name}")
        print("finished test")
        
        torch.save({
            "training_data": train_ls, 
            "validation_data": valid_ls,
            "testing_data": test_ls
        }, save_file_name)
        print("DONE")

    cache_dataset(save_file_name=f"speaker{speaker_id}.pth")

    # train_ls = torch.load(f"train_{speaker_id}.pth")
    # valid_ls = torch.load(f"valid_{speaker_id}.pth")
    # test_ls = torch.load(f"test_{speaker_id}.pth")
    # combined = {**train_ls, **valid_ls, **test_ls}
    # torch.save(combined, f"{speaker_id}.pth")

def train_processed(model, data, num_epochs, batch_size, model_file,
          learning_rate=8e-4, loss_curve=[], validation_curve=[], best_metric=None):
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
                total_num += batch["ai_mel"].size(0)
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

def predict_processed(model, data, num_limit=10, state=None):

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
            pred = (pred*state["sig_data_cuda"]) + state["mu_data_cuda"] # de-standardise
            assert pred.shape == (1, seq_length, n_mels)
            assert pred.squeeze().shape == (seq_length, n_mels)

            data_mel = (batch["data_mel"]*state["sig_data_cuda"]) + state["mu_data_cuda"]
            ai_mel = (batch["ai_mel"]*state["sig_cuda"]) + state["mu_cuda"]
            
            mel_to_audio(pred.squeeze(), f"test{i}_pred.wav")
            mel_to_audio(data_mel.squeeze(), f"test{i}_actual.wav")
            mel_to_audio(ai_mel.squeeze(), f"test{i}_input.wav")

if __name__ == "__main__":

    # 0. Set up logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename='transformer.log', 
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print("Using device:", device)

    # 1. Data visualisation
    ds = load_dataset("ylacombe/expresso")
    dataset = MelSpectrogramDataset("train", ds["train"])
    print(dataset[0]) # data visualisation
    print(dataset[1234]) # data visualisation
    print(dataset[5000]) # data visualisation
    print(dataset[11000]) # out of bound error

    # 2. Trim data for memory saving purposes
    speaker_id = 4
    trim_data(speaker_id) # focus on speaker 4
    dataset = torch.load(f"speaker{speaker_id}.pth")

    ### 2'. Further data processing: this is just to make the articulatory features more normal
    mu = torch.zeros(80)
    x_square = torch.zeros(80)
    sig = torch.zeros(80)
    tot = 0

    for item in dataset["training_data"]:
        tot += item["ai_mel"].size(0)
        curmu = item["ai_mel"].sum(axis=0)
        mu = mu + curmu
        x_square = x_square + (item["ai_mel"]**2).sum(axis=0)

    mu /= tot
    sig = torch.sqrt(x_square/tot - mu*mu)
    mu_cuda = mu.to(device)
    sig_cuda = sig.to(device)

    mu_data = torch.zeros(80)
    x_square_data = torch.zeros(80)
    sig_data = torch.zeros(80)
    tot_data = 0

    for item in dataset["training_data"]:
        tot_data += item["data_mel"].size(0)
        curmu_data = item["data_mel"].sum(axis=0)
        mu_data = mu_data + curmu_data
        x_square_data = x_square_data + (item["data_mel"]**2).sum(axis=0)

    mu_data /= tot_data
    sig_data = torch.sqrt(x_square_data/tot_data - mu_data*mu_data)
    mu_data_cuda = mu_data.to(device)
    sig_data_cuda = sig_data.to(device)

    state = {
        "sig_data_cuda": sig_data_cuda,
        "mu_data_cuda": mu_data_cuda,
        "sig_cuda": sig_cuda,
        "mu_cuda": mu_cuda
    }

    # 3a. Train
    transformer_encoder_model = TransformerSparcEmotionModel(d_model=512, num_encoder_layers=8, dropout=0.1)
    transformer_encoder_model.to(device)

    loss_curve = []
    validation_curve = []

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

    # 3b. Visualisation

    ### 3bi. Training loss curve
    plt.figure(figsize=(15,10))
    plt.plot(np.arange(len(loss_curve)), np.array(loss_curve), label="training loss")
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.grid()
    plt.show()

    ### 3bii. Validation loss curve
    plt.figure(figsize=(15,10))
    plt.plot(np.arange(len(validation_curve)), validation_curve, label="validation loss")
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss Curve')
    plt.grid()
    plt.show()

    ### 3biii. Dual axis plot to check overfitting
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

    # 4. Prediction
    predict_processed(transformer_encoder_model, dataset, state=state)

    waveform_pred, sample_rate_pred = torchaudio.load("test7_pred.wav")
    ipd.display(ipd.Audio(waveform_pred, rate=sample_rate_pred))

    view_spectrogram(get_spectrogram_from_waveform(waveform_pred, sample_rate_pred), title="Mel Spectrogram (Pred)")
