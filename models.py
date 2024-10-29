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