import os
import cv2
import torch
import numpy as np
from torchvision.utils import save_image
from vae_model import ConvVAE
from lstm_model import VideoLSTM
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_latent_sequence(model_lstm, seed_seq, predict_len, noise_std=0.0):
    """
    noise_std: latent 벡터에 추가할 가우시안 노이즈 표준편차 (0.0이면 노이즈 없음)
    """
    model_lstm.eval()
    generated = []
    current_seq = seed_seq.to(device)

    with torch.no_grad():
        for _ in range(predict_len):
            output = model_lstm(current_seq)
            next_latent = output[:, -1, :].unsqueeze(1)

            if noise_std > 0:
                noise = torch.randn_like(next_latent) * noise_std
                next_latent = next_latent + noise

            generated.append(next_latent.squeeze(1).cpu())
            current_seq = torch.cat([current_seq[:, 1:, :], next_latent], dim=1)

    generated = torch.stack(generated).squeeze(1)
    return generated

def latent_to_video(decoder, latent_seq, video_path, fps=25):
    decoder.eval()
    frames = []
    with torch.no_grad():
        for z in tqdm(latent_seq, desc="Decoding frames"):
            z = z.unsqueeze(0).to(device)
            recon = decoder.decode(z)  # (1, C, H, W)
            img_tensor = recon.squeeze(0).cpu()
            img_np = img_tensor.permute(1, 2, 0).numpy()  # HWC
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            frames.append(img_np)

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

if __name__ == "__main__":
    latent_dim = 32
    seq_len = 10
    predict_len = 1000
    noise_std = 0.01  # 여기에 원하는 노이즈 세기 조절

    vae_checkpoint = './vae_checkpoints/vae_epoch_90.pth'
    lstm_checkpoint = './lstm_checkpoints/lstm_epoch_180.pth'
    seed_latent_path = './latent_sequences/person19_walking_d1_uncomp.npy'

    vae = ConvVAE(latent_dim=latent_dim, in_channels=3).to(device)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    lstm = VideoLSTM(latent_dim=latent_dim, hidden_dim=64, num_layers=1).to(device)
    lstm_ckpt = torch.load(lstm_checkpoint, map_location=device)
    lstm.load_state_dict(lstm_ckpt['model_state_dict'])
    lstm.eval()

    seed_seq_np = np.load(seed_latent_path)[:seq_len]
    seed_seq = torch.tensor(seed_seq_np, dtype=torch.float32).unsqueeze(0)

    generated_latents = generate_latent_sequence(lstm, seed_seq, predict_len, noise_std=noise_std)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = f'./generated_video_{now_str}.mp4'
    latent_to_video(vae, generated_latents, output_video_path, fps=25)

    print(f"Video saved to {output_video_path}")
