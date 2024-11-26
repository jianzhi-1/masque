# Inspiration from: https://github.com/jik876/hifi-gan/blob/master/models.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

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

if __name__ == "__main__":

    # 1. Set up
    transformer_encoder_model = TransformerEmotionModel(d_model=512, num_encoder_layers=6, dropout=0.1)
    transformer_encoder_model.to(device)

    discriminator_model = Discriminator(n_mels=80)
    discriminator_model.to(device)

    gan = GAN(transformer_encoder_model, discriminator_model, alpha)

    # 2. Training parameters
    alpha = 100
    loss_curve_g = []
    loss_curve_d = []
    validation_curve = []

    # 3. Train loop
    train_gan(
        gan,
        dataset, 
        num_epochs=40, 
        batch_size=64,
        model_file_g="transformer_encoder_model_speaker_4.pt",
        model_file_d="discriminator_model_speaker_4.pt",
        learning_rate_g=5e-4, 
        learning_rate_d=5e-4, 
        loss_curve_g=loss_curve_g, 
        loss_curve_d=loss_curve_d, 
        validation_curve=validation_curve,
        best_metric=None
    )