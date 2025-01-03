import logging
import numpy as np

from soft_dtw_cuda import SoftDTW

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionModel(nn.Module):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transform(self, batch):
        raise NotImplementedError()

    def compute_loss(self, batch):
        batch_size, seq_length, mels_dim = batch["ai_mel"].shape
        assert batch["data_mel"].shape == (batch_size, seq_length, mels_dim)

        predicted_mel = self.transform(batch)
        assert predicted_mel.shape == (batch_size, seq_length, mels_dim)
        assert not torch.any(torch.isnan(predicted_mel))

        target_mel = batch["data_mel"]
        assert target_mel.shape == (batch_size, seq_length, mels_dim)
        target_mel = torch.nan_to_num(batch["data_mel"], nan=0.0) # purge tensor of nan
        assert target_mel.shape == (batch_size, seq_length, mels_dim)
        assert not torch.any(torch.isnan(target_mel))
        assert mels_dim >= 1
        loss = torch.sum((predicted_mel - target_mel)**2)
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
                loss = self.compute_loss(batch)
                total_mse += loss
                total += batch["ai_mel"].size(0)
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
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1, max_len=2048):
        super().__init__()
        self.max_len = max_len
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x):
        assert x.shape[1] < self.max_len
        x = self.input_dropout(x)
        timing = self.timing_table[None, :x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing

class TransformerEmotionModel(EmotionModel):
    def __init__(self, d_model=512, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.n_mels = 80
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.num_encoder_layers = num_encoder_layers
        encoder_ls = []
        for _ in range(num_encoder_layers):
            encoder_ls.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=False, dropout=dropout, dim_feedforward=d_model))
        self.encoder_layers = nn.ModuleList(encoder_ls)
        self.embedding_layer = nn.Embedding(11, d_model) # len(self.label_encoder) = 11
        self.pre_projection_layer = nn.Linear(self.n_mels, d_model)
        self.post_projection_layer = nn.Linear(d_model, self.n_mels)

    def transform(self, batch):
        
        batch_size, seq_length, _ = batch["ai_mel"].shape
        assert batch["ai_mel"].shape == (batch_size, seq_length, self.n_mels)
        
        batch_input = torch.nan_to_num(batch["ai_mel"], nan=0.0)
        assert not torch.any(torch.isnan(batch_input))
        assert batch_input.shape == (batch_size, seq_length, self.n_mels)

        label = batch["labels"]
        mask = batch["mask"]
        
        assert mask.shape == (batch_size, seq_length)
        assert mask.get_device() == 0 # cuda
        mask = torch.cat((torch.full((batch_size, 1), False).to(self.device), mask), 1)
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
        
        after_encoder = adjoined_with_timing
        
        for i in range(self.num_encoder_layers):
            after_encoder = self.encoder_layers[i](after_encoder, src_key_padding_mask=mask)
            assert after_encoder.shape == (batch_size, 1 + seq_length, self.d_model)
        
        post_adjoined = self.post_projection_layer(after_encoder)
        assert post_adjoined.shape == (batch_size, 1 + seq_length, self.n_mels)
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(res))
        
        return res

### === Duration model ===
class DurationModel(nn.Module):

    def __init__(self, max_duration):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_duration = max_duration

    def fit(self, batch):
        raise NotImplementedError()

    def predict(self, batch, total_duration):
        raise NotImplementedError()

    def compute_loss(self, batch):
        batch_size, seq_length, _ = batch["ai_mel"].shape
        
        logits = self.fit(batch)
        assert logits.shape == (batch_size, seq_length, self.max_duration)

        target_duration = batch["duration"]
        assert target_duration.shape == (batch_size, seq_length)

        target_duration = torch.nan_to_num(target_duration, nan=-1)
        target_duration = torch.where(target_duration >= self.max_duration, torch.tensor(-1, device=device), target_duration)
        assert not torch.any(torch.isnan(target_duration))
        assert torch.all(target_duration >= -1) and torch.all(target_duration < self.max_duration)

        loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")(logits.reshape(-1, self.max_duration), target_duration.view(-1))

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
                loss = self.compute_loss(batch)
                total_mse += loss
                total += batch["ai_mel"].size(0)

        return total_mse/total

class TransformerDurationModel(DurationModel):
    def __init__(self, max_duration, d_model=512, num_encoder_layers=6, dropout=0.1):
        super().__init__(max_duration)
        self.n_mels = 80
        self.max_duration = max_duration # 0, ..., max_duration - 1
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.num_encoder_layers = num_encoder_layers
        encoder_ls = []
        for _ in range(num_encoder_layers):
            encoder_ls.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=False, dropout=dropout, dim_feedforward=d_model))
        self.encoder_layers = nn.ModuleList(encoder_ls)
        self.embedding_layer = nn.Embedding(11, d_model) # len(self.label_encoder) = 11
        self.pre_projection_layer = nn.Linear(self.n_mels, d_model)
        self.post_projection_layer = nn.Linear(d_model, self.max_duration)

    def fit(self, batch):
        """
        Output: the logits for the duration of the Melspectrogram unit
        """
        
        batch_size, seq_length, _ = batch["ai_mel"].shape
        assert batch["ai_mel"].shape == (batch_size, seq_length, self.n_mels)
        
        batch_input = batch["ai_mel"]
        assert batch_input.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(batch_input))

        label = batch["labels"]
        mask = batch["mask"]
        
        assert mask.shape == (batch_size, seq_length)
        assert not torch.any(torch.isnan(mask))
        mask = torch.cat((torch.full((batch_size, 1), False).to(self.device), mask), 1)
        assert mask.shape == (batch_size, 1 + seq_length)
        
        assert label.shape == (batch_size,)
        label_embedded = self.embedding_layer(label).unsqueeze(1)
        assert label_embedded.shape == (batch_size, 1, self.d_model)
        assert not torch.any(torch.isnan(label_embedded))
        
        pre_adjoined = self.pre_projection_layer(batch_input)
        assert pre_adjoined.shape == (batch_size, seq_length, self.d_model)
        assert not torch.any(torch.isnan(pre_adjoined))
        
        adjoined = torch.cat((label_embedded, pre_adjoined), 1)
        assert adjoined.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined))
        
        adjoined_with_timing = self.add_timing(adjoined, mask)
        assert adjoined_with_timing.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined_with_timing))
        
        after_encoder = adjoined_with_timing
        
        for i in range(self.num_encoder_layers):
            after_encoder = self.encoder_layers[i](after_encoder, src_key_padding_mask=mask)
            assert after_encoder.shape == (batch_size, 1 + seq_length, self.d_model)
            assert not torch.any(torch.isnan(after_encoder))
        
        post_adjoined = self.post_projection_layer(after_encoder)
        assert post_adjoined.shape == (batch_size, 1 + seq_length, self.max_duration)
        assert not torch.any(torch.isnan(post_adjoined))
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, seq_length, self.max_duration)
        assert not torch.any(torch.isnan(res))
        
        return res

    def predict(self, batch, total_duration):
        """
        Same as fit, but with the constraint that 
        the sum of time spans needs to be total duration.

        Input:
        batch : dict()
        total_duration: B x 1 tensor

        Output:
        overall_res: B x S tensor
        """

        self.eval()

        batch_size = batch["ai_mel"].size(0)
        res = self.fit(batch)
        _, seq_length, _ = res.shape
        assert res.shape == (batch_size, seq_length, self.max_duration)
        assert total_duration.shape == (batch_size, )
        assert seq_length == batch["ai_mel"].size(1)

        memo = dict()
        max_seq_length = None
        T = None
        def dp(pos, tot):
            if pos == max_seq_length and tot == T:
                return 0
            if pos == max_seq_length: return -np.inf
            if tot > T: return -np.inf
            if (pos, tot) in memo: return memo[pos, tot]
            best_t = None
            best_res = None
            for t in range(self.max_duration):
                cur_res = dp(pos + 1, tot + t) + batch[i, pos, t]
                if best_res is None or cur_res > best_res:
                    best_res = cur_res
                    best_t = t
            memo[(pos, tot)] = (best_res, best_t)
            return memo[(pos, tot)]

        overall_res = []
        for i in range(batch.size(0)):
            # Set up dynamic programming variables
            memo.clear()
            max_seq_length = seq_length
            T = total_duration[i]
            dp(0, 0)

            decode_arr = []
            cur_tot = 0
            for j in range():
                decode_arr.append(memo[(j, cur_tot)][1]) # append best_t
                cur_tot += memo[(j, cur_tot)][1] # consume best_t
            
            assert sum(decode_arr) == total_duration, "decoded arr does not sum to total duration"
            decode_arr = decode_arr + [torch.nan]*(seq_length - len(decode_arr)) # append with np.nan
            assert len(decode_arr) == seq_length
            overall_res.append(decode_arr)
        overall_res = torch.Tensor(overall_res)
        return overall_res

### === CNN Transformer models ===

class ConvTransformerModel(EmotionModel):
    def __init__(self, d_model=512, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.n_mels = 80
        self.cnn_d_model = 640
        self.d_model = d_model

        self.pre_projection_layer = nn.Linear(self.cnn_d_model, d_model)
        self.post_projection_layer = nn.Linear(d_model, self.cnn_d_model)

        # Transformer components
        self.add_timing = AddPositionalEncoding(d_model)
        self.num_encoder_layers = num_encoder_layers
        encoder_ls = []
        for _ in range(num_encoder_layers):
            encoder_ls.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=False, dropout=dropout, dim_feedforward=d_model))
        self.encoder_layers = nn.ModuleList(encoder_ls)
        self.embedding_layer = nn.Embedding(11, d_model) # len(self.label_encoder) = 11

        # CNN components
        self.conv_layer_1 = nn.Conv2d(1, 4, 3, stride=1, padding=1, bias=True, padding_mode="zeros") # (B, 1, S, 80) -> (B, 4, S, 80)
        self.maxpool_layer_1 = nn.MaxPool2d(2, stride=2, padding=0) # (B, 4, S, 80) -> (B, 4, S/2, 40)
        self.conv_layer_2 = nn.Conv2d(4, 16, 3, stride=1, padding=1, bias=True, padding_mode="zeros") # (B, 4, S/2, 40) -> (B, 16, S/2, 40)
        self.maxpool_layer_2 = nn.MaxPool2d(2, stride=2, padding=0) # (B, 16, S/2, 40) -> (B, 16, S/4, 20)
        self.conv_layer_3 = nn.Conv2d(16, 64, 3, stride=1, padding=1, bias=True, padding_mode="zeros") # (B, 16, S/4, 20) -> (B, 64, S/4, 20)
        self.maxpool_layer_3 = nn.MaxPool2d(2, stride=2, padding=0) # (B, 64, S/4, 20) -> (B, 64, S/8, 10)

        # Embedding layers
        self.embeddings = nn.Parameter(torch.randn(64, 8, 8)) # (64, A, A)

    def transform(self, batch):
        
        batch_size, seq_length, _ = batch["ai_mel"].shape # (B, S, 80)
        assert batch["ai_mel"].shape == (batch_size, seq_length, self.n_mels)
        assert seq_length % 8 == 0, "not valid input for CNN"
        
        batch_input = batch["ai_mel"] # (B, S, 80)
        assert batch_input.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(batch_input))

        ### CNN phase
        batch_input_d = batch_input.unsqueeze(1) # (B, 1, S, 80)
        after_conv_1 = F.relu(self.conv_layer_1(batch_input_d)) # (B, 4, S, 80)
        after_mp_1 = self.maxpool_layer_1(after_conv_1) # (B, 4, S/2, 40)
        after_conv_2 = F.relu(self.conv_layer_2(after_mp_1)) # (B, 16, S/2, 40)
        after_mp_2 = self.maxpool_layer_2(after_conv_2) # (B, 16, S/4, 20)
        after_conv_3 = F.relu(self.conv_layer_3(after_mp_2)) # (B, 64, S/4, 20)
        after_mp_3 = self.maxpool_layer_3(after_conv_3) # (B, 64, S/8, 10)
        assert not torch.any(torch.isnan(after_mp_3))
        ### End of CNN phase

        adjusted_seq_length = after_mp_3.size(2) # S/8
        transformer_input = after_mp_3.permute(0, 2, 1, 3).reshape(batch_size, adjusted_seq_length, -1) # (B, S/8, 640)
        assert transformer_input.shape == (batch_size, adjusted_seq_length, 640) # (B, S/8, 640)
        
        label = batch["labels"]

        ### Cumbersome to exploit batch_size due to CNN
        mask = torch.zeros(batch_size, 1 + adjusted_seq_length, dtype=torch.bool).to(self.device) # (B, 1 + S/8)
        assert mask.shape == (batch_size, 1 + adjusted_seq_length) # (B, 1 + S/8)
        
        assert label.shape == (batch_size,)
        label_embedded = self.embedding_layer(label).unsqueeze(1)
        assert label_embedded.shape == (batch_size, 1, self.d_model) # (B, 1, 512)
        assert not torch.any(torch.isnan(label_embedded))
        
        pre_adjoined = self.pre_projection_layer(transformer_input) # (B, S/8, 512)
        assert pre_adjoined.shape == (batch_size, adjusted_seq_length, self.d_model)
        assert not torch.any(torch.isnan(pre_adjoined))
        
        adjoined = torch.cat((label_embedded, pre_adjoined), 1) # (B, 1 + S/8, 512)
        assert adjoined.shape == (batch_size, 1 + adjusted_seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined))
        
        adjoined_with_timing = self.add_timing(adjoined, mask)
        assert adjoined_with_timing.shape == (batch_size, 1 + adjusted_seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined_with_timing))
        
        after_encoder = adjoined_with_timing
        
        for i in range(self.num_encoder_layers):
            after_encoder = self.encoder_layers[i](after_encoder, src_key_padding_mask=mask)
            assert after_encoder.shape == (batch_size, 1 + adjusted_seq_length, self.d_model)
            assert not torch.any(torch.isnan(after_encoder))
        
        post_adjoined = self.post_projection_layer(after_encoder) # (B, 1 + S/8, 640)
        assert post_adjoined.shape == (batch_size, 1 + adjusted_seq_length, 640)
        assert not torch.any(torch.isnan(post_adjoined))
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, adjusted_seq_length, self.cnn_d_model) # (B, S/8, 640)
        assert not torch.any(torch.isnan(res))

        after_transformer = res.reshape(batch_size, adjusted_seq_length, 64, 10).permute(0, 2, 1, 3) # (B, 64, S/8, 10)
        assert after_transformer.shape == (batch_size, 64, adjusted_seq_length, 10)
        assert not torch.any(torch.isnan(after_transformer))
        
        ### Start of CNN embedding phase
        softmaxed_conv = torch.softmax(after_transformer, dim=1) # (B, 64, S/8, 10)
        weighted_embeddings = torch.einsum('iljk,lmn->ijkmn', softmaxed_conv, self.embeddings) # (B, S/8, 10, A, A)
        output_tensor = torch.einsum('ijkmn->ijmkn', weighted_embeddings).contiguous().view(batch_size, seq_length, self.n_mels)  # (B, S/8, A, 10, A) -> B, (S/8*A), (10*A)
        assert output_tensor.shape == (batch_size, seq_length, self.n_mels)
        ### End of CNN embedding phase

        padded_input = F.pad(batch_input, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded_input.unsqueeze(1), torch.ones(1, 1, 5, 5).to(self.device) / 25.0, groups=batch_input.size(0)).squeeze(1) + output_tensor # residual network

### === GAN model ===

class Discriminator(nn.Module):

    """
    A CNN-based discriminator
    The unique aspect is that it must classify variable length image.
    """

    def __init__(self, n_mels=80):
        super(Discriminator, self).__init__()
        self.n_mels = n_mels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Output: B x 32 x W x 80
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Output: B x 32 x (W//2) x 40
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # Output: B x 64 x (W//2) x 40
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Output: B x 64 x (W//4) x 20
        )
        
        # Standardise output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 20))  # Output: B x 64 x 1 x 20
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Output: B x (64x20)
            nn.Linear(64 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Logit
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        assert x.shape == (batch_size, seq_length, self.n_mels) # x shape: B x W x 80
        
        x = x.unsqueeze(1)  # Add channel dimension 
        assert x.shape == (batch_size, 1, seq_length, self.n_mels) # x shape: B x 1 x W x 80

        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # Standardise dimensions
        assert x.shape == (batch_size, 64, 1, 20)

        x = self.fc_layers(x)
        return x

class GAN():

    """
    Generative adversarial network class
    Couples a generator and a discriminator and computes losses
    """

    def __init__(self, generator, discriminator, multiplier):
        self.generator = generator
        self.discriminator = discriminator
        self.multiplier = multiplier

    def compute_loss(self, batch):
        batch_size, seq_length, mels_dim = batch["ai_mel"].shape
        assert batch["data_mel"].shape == (batch_size, seq_length, mels_dim)

        feature_loss, fake_mel = self.generator.compute_loss_gan(batch) # Optimised so that the prediction only happens once
        human_mel = batch["data_mel"]

        fake_probs = self.discriminator(fake_mel)
        real_probs = self.discriminator(human_mel)

        generator_loss = feature_loss + self.multiplier * torch.sum((fake_probs - 1)**2)/batch_size # L(G; D) = E[(D(G(s)) - 1)**2]

        discriminator_loss = (torch.sum((real_probs - 1)**2) + torch.sum(fake_probs**2))/batch_size # L(D; G) = E[(D(x) - 1)**2 + D(G(s))**2]

        return generator_loss, discriminator_loss

    def get_validation_metric(self, data_loader, batch_size=64):

        self.generator.eval()
        self.discriminator.eval()

        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                loss_g, _ = self.compute_loss(batch)
                total_loss += loss_g
                total += batch["ai_mel"].size(0)
                logging.info(f"validation; batch={i}; loss={loss_g.item()}; total_mse={total_loss}; ave_loss={total_loss/total}")

        return total_loss/total

### === Soft DTW model === 

class EmotionSDTWModel(EmotionModel):

    def __init__(self):
        super().__init__()
        self.sdtw = SoftDTW(use_cuda=False, gamma=0.1)
        self.multiplier = 0.2

    def compute_loss(self, batch):
        batch_size, seq_length, mels_dim = batch["ai_mel"].shape
        assert batch["data_mel"].shape == (batch_size, seq_length, mels_dim)

        predicted_mel = self.transform(batch)
        assert predicted_mel.shape == (batch_size, seq_length, mels_dim)

        target_mel = batch["data_mel"]
        assert target_mel.shape == (batch_size, seq_length, mels_dim)

        assert mels_dim == 80

        feature_loss = torch.sum((predicted_mel - target_mel)**2)
        dtw_loss = torch.sum(self.sdtw(predicted_mel, batch["original_data_mel"])) # torch.sum(self.sdtw(predicted_mel, target_mel)) # add DTW loss

        loss = feature_loss + self.multiplier*dtw_loss # dtw_loss
        return loss

class TransformerSDTWEmotionModel(EmotionSDTWModel):
    def __init__(self, d_model=512, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.n_mels = 80
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.num_encoder_layers = num_encoder_layers
        encoder_ls = []
        for _ in range(num_encoder_layers):
            encoder_ls.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=False, dropout=dropout, dim_feedforward=d_model))
        self.encoder_layers = nn.ModuleList(encoder_ls)
        self.embedding_layer = nn.Embedding(11, d_model) # len(self.label_encoder) = 11
        self.pre_projection_layer = nn.Linear(self.n_mels, d_model)
        self.post_projection_layer = nn.Linear(d_model, self.n_mels)

    def transform(self, batch):
        
        batch_size, seq_length, _ = batch["ai_mel"].shape
        assert batch["ai_mel"].shape == (batch_size, seq_length, self.n_mels)
        
        batch_input = batch["ai_mel"]
        assert batch_input.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(batch_input))

        label = batch["labels"]
        mask = batch["mask"]
        
        assert mask.shape == (batch_size, seq_length)
        assert not torch.any(torch.isnan(mask))
        mask = torch.cat((torch.full((batch_size, 1), False).to(self.device), mask), 1)
        assert mask.shape == (batch_size, 1 + seq_length)
        
        assert label.shape == (batch_size,)
        label_embedded = self.embedding_layer(label).unsqueeze(1)
        assert label_embedded.shape == (batch_size, 1, self.d_model)
        assert not torch.any(torch.isnan(label_embedded))
        
        pre_adjoined = self.pre_projection_layer(batch_input)
        assert pre_adjoined.shape == (batch_size, seq_length, self.d_model)
        assert not torch.any(torch.isnan(pre_adjoined))
        
        adjoined = torch.cat((label_embedded, pre_adjoined), 1)
        assert adjoined.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined))
        
        adjoined_with_timing = self.add_timing(adjoined, mask)
        assert adjoined_with_timing.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined_with_timing))
        
        after_encoder = adjoined_with_timing
        
        for i in range(self.num_encoder_layers):
            after_encoder = self.encoder_layers[i](after_encoder, src_key_padding_mask=mask)
            assert after_encoder.shape == (batch_size, 1 + seq_length, self.d_model)
            assert not torch.any(torch.isnan(after_encoder))
        
        post_adjoined = self.post_projection_layer(after_encoder)
        assert post_adjoined.shape == (batch_size, 1 + seq_length, self.n_mels)
        assert not torch.any(torch.isnan(post_adjoined))
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, seq_length, self.n_mels)
        assert not torch.any(torch.isnan(res))
        
        return res

### === SPARC models and variants === 

class EmotionSparcModel(EmotionModel):

    def compute_loss(self, batch): # override
        batch_size, seq_length, sparc_dim = batch["ai_sparc"].shape
        assert batch["data_sparc"].shape == (batch_size, seq_length, sparc_dim)

        predicted_sparc = self.transform(batch)
        assert predicted_sparc.shape == (batch_size, seq_length, sparc_dim)

        target_sparc = batch["data_sparc"]
        assert target_sparc.shape == (batch_size, seq_length, sparc_dim)

        assert sparc_dim == 12 # SPARC only has 12 dimensions
        loss = torch.sum((predicted_sparc - target_sparc)**2)
        return loss
  
    def get_validation_metric(self, validation_dataset, batch_size=64): # override
        dataset = validation_dataset # replace because of caching efficiency
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collate
        )
        self.eval()
        total_mse = 0.0
        total = 0
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                loss = self.compute_loss(batch)
                total_mse += loss
                total += batch["ai_sparc"].size(0)

        return total_mse/total

class TransformerSparcEmotionModel(EmotionSparcModel):
    def __init__(self, d_model=512, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        self.n_sparc = 12 # SPARC only has 12 features
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.num_encoder_layers = num_encoder_layers
        encoder_ls = []
        for _ in range(num_encoder_layers):
            encoder_ls.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, norm_first=False, dropout=dropout, dim_feedforward=d_model))
        self.encoder_layers = nn.ModuleList(encoder_ls)
        self.embedding_layer = nn.Embedding(11, d_model) # len(self.label_encoder) = 11
        self.pre_projection_layer = nn.Linear(self.n_sparc, d_model)
        self.post_projection_layer = nn.Linear(d_model, self.n_sparc)

    def transform(self, batch):
        
        batch_size, seq_length, _ = batch["ai_sparc"].shape
        assert batch["ai_sparc"].shape == (batch_size, seq_length, self.n_sparc)
        
        batch_input = batch["ai_sparc"]
        assert batch_input.shape == (batch_size, seq_length, self.n_sparc)
        assert not torch.any(torch.isnan(batch_input))

        label = batch["labels"]
        mask = batch["mask"]
        
        assert mask.shape == (batch_size, seq_length)
        assert not torch.any(torch.isnan(mask))
        mask = torch.cat((torch.full((batch_size, 1), False).to(self.device), mask), 1)
        assert mask.shape == (batch_size, 1 + seq_length)
        
        assert label.shape == (batch_size,)
        label_embedded = self.embedding_layer(label).unsqueeze(1)
        assert label_embedded.shape == (batch_size, 1, self.d_model)
        assert not torch.any(torch.isnan(label_embedded))
        
        pre_adjoined = self.pre_projection_layer(batch_input)
        assert pre_adjoined.shape == (batch_size, seq_length, self.d_model)
        assert not torch.any(torch.isnan(pre_adjoined))
        
        adjoined = torch.cat((label_embedded, pre_adjoined), 1)
        assert adjoined.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined))
        
        adjoined_with_timing = self.add_timing(adjoined, mask)
        assert adjoined_with_timing.shape == (batch_size, 1 + seq_length, self.d_model)
        assert not torch.any(torch.isnan(adjoined_with_timing))
        
        after_encoder = adjoined_with_timing
        
        for i in range(self.num_encoder_layers):
            after_encoder = self.encoder_layers[i](after_encoder, src_key_padding_mask=mask)
            assert after_encoder.shape == (batch_size, 1 + seq_length, self.d_model)
            assert not torch.any(torch.isnan(after_encoder))
        
        post_adjoined = self.post_projection_layer(after_encoder)
        assert post_adjoined.shape == (batch_size, 1 + seq_length, self.n_sparc)
        assert not torch.any(torch.isnan(post_adjoined))
        
        res = post_adjoined[:,1:,:]
        assert res.shape == (batch_size, seq_length, self.n_sparc)
        assert not torch.any(torch.isnan(res))
        
        return res
