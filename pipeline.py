
from dataset_class import MelSpectrogramDataset, ProcessedMelSpectrogramDataset
import logging
import matplotlib.pyplot as plt
from models import TransformerEmotionModel
import numpy as np

import torch
import tqdm.notebook

from utils import mel_to_audio

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

