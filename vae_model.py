import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32, in_channels=3):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 32, 4, 2, 1)  # 64->32
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)           # 32->16
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)          # 16->8
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)         # 8->4

        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)

        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4->8
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 8->16
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 16->32
        self.dec_conv4 = nn.ConvTranspose2d(32, in_channels, 4, 2, 1)  # 32->64

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))  # output 0~1
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
