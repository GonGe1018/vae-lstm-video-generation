import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from vae_model import ConvVAE
from lstm_model import VideoLSTM
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_latent(vae, image_tensor):
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(image_tensor.unsqueeze(0).to(device))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
    return z.cpu()

def generate_from_single_latent(model_lstm, init_latent, seq_len, noise_std=0.0):
    model_lstm.eval()
    generated = []
    # 초기 시퀀스는 동일 latent 벡터를 seq_len번 반복
    current_seq = init_latent.unsqueeze(1).repeat(1, seq_len, 1).to(device)

    with torch.no_grad():
        for _ in range(seq_len):
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
            recon = decoder.decode(z)
            img_tensor = recon.squeeze(0).cpu()
            img_np = img_tensor.permute(1, 2, 0).numpy()
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
    seq_len = 100  # 생성 프레임 수
    noise_std = 0.01  # 노이즈 강도 조절

    vae_checkpoint = './vae_checkpoints/vae_epoch_50.pth'
    lstm_checkpoint = './lstm_checkpoints/lstm_epoch_30.pth'
    input_image_path = './walking_frames/person01_walking_d1_uncomp_frame060.png'  # 예시

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 모델 로드
    vae = ConvVAE(latent_dim=latent_dim, in_channels=3).to(device)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    lstm = VideoLSTM(latent_dim=latent_dim, hidden_dim=64, num_layers=1).to(device)
    lstm_ckpt = torch.load(lstm_checkpoint, map_location=device)
    lstm.load_state_dict(lstm_ckpt['model_state_dict'])
    lstm.eval()

    # 이미지 -> latent 변환
    image = Image.open(input_image_path).convert('RGB')
    image_tensor = transform(image)
    init_latent = image_to_latent(vae, image_tensor)

    # latent 시퀀스 생성
    generated_latents = generate_from_single_latent(lstm, init_latent, seq_len, noise_std=noise_std)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = f'./generated_video_from_image_{now_str}.mp4'
    latent_to_video(vae, generated_latents, output_video_path, fps=25)

    print(f"Video saved to {output_video_path}")
