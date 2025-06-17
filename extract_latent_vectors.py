import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from vae_model import ConvVAE
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = './walking_frames'           # 프레임 이미지 폴더
latent_save_root = './latent_sequences'  # latent 저장 폴더
os.makedirs(latent_save_root, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])

# 학습된 VAE 모델 불러오기
vae_model = ConvVAE(latent_dim=32, in_channels=3).to(device)
checkpoint = torch.load('./vae_checkpoints/vae_epoch_90.pth', map_location=device)
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()

# 프레임 이미지들을 시퀀스별로 그룹핑
all_frames = glob.glob(os.path.join(data_root, '*.png'))

# 파일 이름 패턴에서 시퀀스 이름 추출 함수 예)
def get_sequence_name(filename):
    # 예: person09_walking_d3_uncomp_frame280.png -> person09_walking_d3_uncomp
    base = os.path.basename(filename)
    return '_'.join(base.split('_')[:-1])  # 마지막 _frameXXX 제거

# 시퀀스별 프레임 리스트 만들기
from collections import defaultdict
seq_dict = defaultdict(list)
for f in all_frames:
    seq_name = get_sequence_name(f)
    seq_dict[seq_name].append(f)

# 프레임 정렬 후 latent 추출 및 저장
for seq_name, frame_files in seq_dict.items():
    frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[-1].replace('frame','').replace('.png','')))
    latents = []

    for f in tqdm(frame_files, desc=f"Processing {seq_name}"):
        img = Image.open(f).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)
        with torch.no_grad():
            mu, logvar = vae_model.encode(img_tensor)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # (1, latent_dim)
            latents.append(z.squeeze(0).cpu().numpy())

    latents = np.stack(latents)  # (seq_len, latent_dim)
    save_path = os.path.join(latent_save_root, f"{seq_name}.npy")
    np.save(save_path, latents)
    print(f"Saved latent sequence: {save_path}")
