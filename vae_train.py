import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from PIL import Image
import glob

from vae_model import ConvVAE

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

def train(model, dataloader, optimizer, device, epochs=50, checkpoint_dir='./vae_checkpoints',
          save_interval=5, patience=7, min_delta=1e-3, latent_dim=32, result_dir='./vae_results',
          start_epoch=1, resume_checkpoint=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    # 체크포인트 로드
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Loading checkpoint {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', start_epoch)
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item() / len(data))

        avg_loss = train_loss / len(dataloader.dataset)
        print(f"Epoch {epoch} Average loss: {avg_loss:.4f}")

        if epoch % save_interval == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'vae_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch + 1,  # 다음 에포크부터 시작하도록 저장
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            sample = data[:8]
            recon, _, _ = model(sample)
            comparison = torch.cat([sample, recon])
            save_image(comparison.cpu(), os.path.join(result_dir, f'reconstruction_{epoch}.png'), nrow=8)

        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

if __name__ == "__main__":
    data_dir = './walking_frames'
    batch_size = 256
    epochs = 300
    latent_dim = 32
    learning_rate = 5*1e-4
    save_interval = 10
    patience = 7
    min_delta = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FrameDataset(root_dir=data_dir, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConvVAE(latent_dim=latent_dim, in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 이어하기 체크포인트 경로 지정 (없으면 None)
    resume_checkpoint = './vae_checkpoints/vae_epoch_50.pth'  # 예시, 없으면 None

    start_epoch = 1
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        start_epoch = checkpoint.get('epoch', 1)

    train(model, dataloader, optimizer, device, epochs=epochs, checkpoint_dir='./vae_checkpoints',
          save_interval=save_interval, patience=patience, min_delta=min_delta, latent_dim=latent_dim,
          result_dir='./vae_results', start_epoch=start_epoch, resume_checkpoint=resume_checkpoint)
