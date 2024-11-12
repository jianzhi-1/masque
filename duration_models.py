import torch
import torch.nn as nn
import numpy as np

class DurationModel(nn.Module):

    def __init__(self, max_duration):
        super().__init__()
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

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1, max_len=2048):
        super().__init__()
        self.max_len = max_len
        self.timing_table = nn.Parameter(torch.zeros(max_len))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x, mask):
        batch_size, seq_length, d_model = x.shape
        assert x.shape == (batch_size, seq_length, d_model)
        assert mask.shape == (batch_size, seq_length)
        assert seq_length < self.max_len
        x = self.input_dropout(x)
        timing = self.timing_table[:seq_length]
        timing = self.timing_dropout(timing)
        assert timing.shape == (seq_length,), f"{timing.shape}"
        assert timing.unsqueeze(0).unsqueeze(2).shape == (1, seq_length, 1), f"{timing.unsqueeze(0).unsqueeze(2).shape}"
        assert (x + timing.unsqueeze(0).unsqueeze(2)).shape == (batch_size, seq_length, d_model), f"{(x + timing.unsqueeze(0).unsqueeze(2)).shape}"
        assert mask.unsqueeze(-1).expand(-1, -1, d_model).shape == (batch_size, seq_length, d_model), f"{mask.unsqueeze(-1).expand(-1, -1, d_model)}"
        return torch.where(mask.unsqueeze(-1).expand(-1, -1, d_model)==False, x + timing.unsqueeze(0).unsqueeze(2), x)

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
        mask = torch.cat((torch.full((batch_size, 1), False).to(device), mask), 1)
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
            max_seq_length = seq_length # CHECK
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