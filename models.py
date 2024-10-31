from utils import MelSpectrogramDataset
import torch
import torch.nn as nn

class EmotionModel(nn.Module):
    def transform(self, batch):
        raise NotImplementedError()

    def compute_loss(self, batch):
        predicted_mel = self.transform(batch)
        loss = loss_fn(batch["data_mel"], predicted_mel) # TODO some loss function
        return loss
  
    def get_validation_metric(self, source, batch_size=8):
        dataset = MelSpectrogramDataset('valid', source)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collate
        )
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                # TODO 
        return # TODO

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1, max_len=512):
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
    def __init__(self, d_model=256):
        super().__init__()
        self.n_mels = 80
        self.d_model = d_model
        self.add_timing = AddPositionalEncoding(d_model)
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, norm_first=True, dropout=0.1, dim_feedforward=512)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, norm_first=True, dropout=0.1, dim_feedforward=512)
        self.embedding_layer = nn.Embedding(11, 80) # len(self.label_encoder) = 11
        self.conv = nn.Conv2d(80, 80, 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.layer_norm_layer = nn.LayerNorm(d_model)
        self.projection_layer = nn.Linear(d_model, 80) 

    def encode(self, batch):

        label = batch['label']
        x = batch['ai_mel']
        mask = None # TODO
        
        batch_size, = label.shape
        _, seq_length, n_mels = x.shape
        assert label.shape == (batch_size,)
        assert x.shape == (batch_size, seq_length, n_mels)
        
        label_embedded = self.embedding_layer(label)
        assert label_embedded.shape == () # TODO
        adjoined = torch.cat((label_embedded, x), dim=2)
        assert adjoined.shape == () # TODO
        
        delta = self.encoder_layer_1(adjoined, src_mask=mask) # TODO
        #delta2 = self.conv(adjoined)
        assert delta.shape == () # TODO
        
        coalesced = x + delta # residual
        #coalesced2 = x + delta # residual
        assert coalesced.shape == () # TODO
        
        # add another normalisation layer?
        # or a projection layer?
        
        return coalesced