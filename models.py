from utils import MelSpectrogramDataset, train
import torch
import torch.nn as nn
import logging

class EmotionModel(nn.Module):
    def transform(self, batch):
        raise NotImplementedError()

    def compute_loss(self, batch):
        batch_size, seq_length, mels_dim = batch["ai_mel"].shape
        assert batch["data_mel"].shape == (batch_size, seq_length, mels_dim)

        predicted_mel = self.transform(batch)
        assert predicted_mel.shape == (batch_size, seq_length, mels_dim)
        predicted_mel = torch.where(torch.isneginf(batch["ai_mel"]), torch.tensor(0.0), predicted_mel) # purge tensor of -inf
        assert predicted_mel.shape == (batch_size, seq_length, mels_dim)
        assert not torch.any(torch.isneginf(predicted_mel))
        assert not torch.any(torch.isinf(predicted_mel))
        assert not torch.any(torch.isnan(predicted_mel))

        target_mel = batch["data_mel"]
        assert target_mel.shape == (batch_size, seq_length, mels_dim)
        target_mel = torch.where(torch.isneginf(batch["data_mel"]), torch.tensor(0.0), target_mel) # purge tensor of -inf
        assert target_mel.shape == (batch_size, seq_length, mels_dim)
        assert not torch.any(torch.isneginf(target_mel))
        assert not torch.any(torch.isinf(target_mel))
        assert not torch.any(torch.isnan(target_mel))
        assert mels_dim >= 1
        cur_loss = torch.sum((predicted_mel - target_mel)**2, dim=2)/mels_dim # average MSE across mel dim = 80
        assert cur_loss.shape == (batch_size, seq_length)
        assert torch.all(batch["sequence_length"] > 0), "sequence length must be positive"

        cur_loss_2 = torch.sum(cur_loss, dim=1)/batch["sequence_length"] # average MSE across the sequence
        assert cur_loss_2.shape == (batch_size,)

        loss = torch.sum(cur_loss_2)
        return loss
  
    def get_validation_metric(self, validation_dataset, batch_size=64):
        dataset = validation_dataset # replace because of caching efficiency
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collate
        )
        self.eval()
        total_mse = 0.0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch_size, seq_length, mels_dim = batch["ai_mel"].shape
                assert batch["data_mel"].shape == (batch_size, seq_length, mels_dim)
                
                predicted_mel = self.transform(batch)
                assert predicted_mel.shape == (batch_size, seq_length, mels_dim)
                
                predicted_mel = torch.where(torch.isneginf(batch["ai_mel"]), torch.tensor(0.0), predicted_mel) # purge tensor of -inf
                assert predicted_mel.shape == (batch_size, seq_length, mels_dim)
                assert not torch.any(torch.isneginf(predicted_mel))
                assert not torch.any(torch.isinf(predicted_mel))
                assert not torch.any(torch.isnan(predicted_mel))
                
                target_mel = batch["data_mel"]
                assert target_mel.shape == (batch_size, seq_length, mels_dim)
                target_mel = torch.where(torch.isneginf(batch["data_mel"]), torch.tensor(0.0), target_mel) # purge tensor of -inf
                assert target_mel.shape == (batch_size, seq_length, mels_dim)
                assert not torch.any(torch.isneginf(target_mel))
                assert not torch.any(torch.isinf(target_mel))
                assert not torch.any(torch.isnan(target_mel))
                
                cur_loss = torch.sum((predicted_mel - target_mel)**2, dim=2)/mels_dim # average MSE across mel dim = 80
                assert cur_loss.shape == (batch_size, seq_length)
                assert torch.all(batch["sequence_length"] > 0), "sequence length must be positive"
                
                cur_loss_2 = torch.sum(cur_loss, dim=1)/batch["sequence_length"] # average MSE across the sequence
                assert cur_loss_2.shape == (batch_size,)
                loss = torch.sum(cur_loss_2)
                
                total_mse += loss
                total += batch_size
                logging.info(f"validation; batch={i}; loss={loss.item()}; total_mse={total_mse}; ave_loss={total_mse/total}")

        return total_mse/total

class BaselineModel(EmotionModel):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(80, 80)
        self.linear_layer_2 = nn.Linear(80, 80)

    def transform(self, batch):
        batch_size, seq_length, mel_dim = batch["ai_mel"].shape
        assert batch["ai_mel"].shape == (batch_size, seq_length, mel_dim)
        
        batch_input = torch.where(torch.isneginf(batch["ai_mel"]), torch.tensor(0.0), batch["ai_mel"])
        assert not torch.any(torch.isneginf(batch_input))
        assert not torch.any(torch.isinf(batch_input))
        assert not torch.any(torch.isnan(batch_input))
        assert batch_input.shape == (batch_size, seq_length, mel_dim)
        
        after_linear_1 = self.linear_layer_1(batch_input)
        assert after_linear_1.shape == (batch_size, seq_length, mel_dim)
        after_relu = F.relu(after_linear_1)
        assert after_relu.shape == (batch_size, seq_length, mel_dim)
        after_linear_2 = self.linear_layer_2(after_relu)
        assert after_linear_2.shape == (batch_size, seq_length, mel_dim)
        return after_linear_2

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1, max_len=1024):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x):
        x = self.input_dropout(x)
        timing = self.timing_table[None, :x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing

class TransformerEmotionModel(EmotionModel):
    def __init__(self, d_model=512):
        super().__init__()
        self.n_mels = 80
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, norm_first=False, dropout=0.1, dim_feedforward=512)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, norm_first=False, dropout=0.1, dim_feedforward=512)
        self.embedding_layer = nn.Embedding(11, self.d_model) # len(self.label_encoder) = 11
        self.pre_projection_layer = nn.Linear(self.n_mels, self.d_model)
        self.post_projection_layer = nn.Linear(self.d_model, self.n_mels)
        # self.conv = nn.Conv2d(80, 80, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # self.layer_norm_layer = nn.LayerNorm(d_model)

    def transform(self, batch):
        
        batch_size, seq_length, _ = batch["ai_mel"].shape
        assert batch["ai_mel"].shape == (batch_size, seq_length, self.n_mels)
        
        batch_input = torch.where(torch.isneginf(batch["ai_mel"]), torch.tensor(0.0), batch["ai_mel"])
        assert not torch.any(torch.isneginf(batch_input))
        assert not torch.any(torch.isinf(batch_input))
        assert not torch.any(torch.isnan(batch_input))
        assert batch_input.shape == (batch_size, seq_length, self.n_mels)

        label = batch["labels"]
        mask = batch["mask"]
        
        assert mask.shape == (batch_size, seq_length)
        assert mask.get_device() == 0 # cuda
        mask = torch.cat((torch.full((batch_size, 1), False).to(device), mask), 1)
        assert mask.shape == (batch_size, 1 + seq_length)
        
        assert label.shape == (batch_size,)
        label_embedded = self.embedding_layer(label).unsqueeze(1)
        assert label_embedded.shape == (batch_size, 1, self.d_model)
        
        pre_adjoined = self.pre_projection_layer(batch_input)
        assert pre_adjoined.shape == (batch_size, seq_length, self.d_model)
        
        adjoined = torch.cat((label_embedded, pre_adjoined), 1)
        assert adjoined.shape == (batch_size, 1 + seq_length, self.d_model)
        
        adjoined_with_timing = self.add_timing(adjoined)
        assert adjoined_with_timing.shape == (batch_size, 1 + seq_length, self.d_model)
        
        after_encoder_1 = self.encoder_layer_1(
            adjoined_with_timing, 
            src_key_padding_mask=mask
        ) # what is is_causal doing?
        
        assert after_encoder_1.shape == (batch_size, 1 + seq_length, self.d_model)
        
        after_encoder_2 = self.encoder_layer_2(
            after_encoder_1, 
            src_key_padding_mask=mask
        ) # what is is_causal doing?
        
        assert after_encoder_2.shape == (batch_size, 1 + seq_length, self.d_model)
        
        post_adjoined = self.post_projection_layer(after_encoder_2)
        assert post_adjoined.shape == (batch_size, 1 + seq_length, self.n_mels)
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isneginf(res))
        assert not torch.any(torch.isinf(res))
        assert not torch.any(torch.isnan(res))
        
        return res

if __name__ == "__main__":
    from datasets import load_dataset

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print("Using device:", device)

    ds = load_dataset("ylacombe/expresso")
    filtered_ds = ds["train"].filter(lambda x: x['speaker_id'] == "ex01")

    baseline_model = BaselineModel().to(device)
    train(baseline_model, ds["train"], num_epochs=5, batch_size=64,
        model_file="baseline_model.pt", learning_rate=0.1)

    transformer_encoder_model = TransformerEmotionModel().to(device)
    transformer_validation_curve, transformer_total_loss_curve = train(
        transformer_encoder_model, 
        filtered_ds, 
        params = {
            "train": (0, 2000),
            "valid": (2000, 2450),
            "test": (2450, 2903)
        },
        num_epochs=3, 
        batch_size=64,
        model_file="transformer_encoder_model.pt", 
        learning_rate=0.1
    )