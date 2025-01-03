import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

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
