import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import os

from latent_dataset import LatentSequenceDataset
from lstm_model import VideoLSTM

def train_lstm(model, dataloader, device, epochs=30, lr=1e-3, checkpoint_dir='./lstm_checkpoints', save_interval=5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for input_seq, target_seq in progress_bar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * input_seq.size(0)
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.6f}")

        if epoch % save_interval == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'lstm_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    latent_dir = './latent_sequences'
    batch_size = 64
    seq_len = 20
    latent_dim = 32
    epochs = 3000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_interval = 10

    dataset = LatentSequenceDataset(latent_dir, seq_len=seq_len)
    print(f"Total sequences: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VideoLSTM(latent_dim=latent_dim, hidden_dim=64, num_layers=1)
    train_lstm(model, dataloader, device, epochs=epochs, lr=1e-3, checkpoint_dir='./lstm_checkpoints', save_interval=save_interval)
