import torch.nn as nn

class VideoLSTM(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, num_layers=1):
        super(VideoLSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, latent_dim)
        out, _ = self.lstm(x)             # (batch, seq_len, hidden_dim)
        out = self.fc(out)                # (batch, seq_len, latent_dim)
        return out
