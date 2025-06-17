import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LatentSequenceDataset(Dataset):
    def __init__(self, latent_dir, seq_len=20):
        """
        latent_dir: latent 벡터 저장 폴더
        seq_len: 시퀀스 길이 (입력 시퀀스 길이)
        """
        self.seq_len = seq_len
        self.latent_files = sorted(glob.glob(os.path.join(latent_dir, '*.npy')))
        self.sequences = []

        # 각 latent 파일을 불러서 seq_len 단위로 자르기
        for file_path in self.latent_files:
            latent_seq = np.load(file_path)  # (total_frames, latent_dim)
            total_len = latent_seq.shape[0]
            # seq_len+1 만큼 자르는 이유는 입력과 타겟 시퀀스를 만드려구요
            for start_idx in range(0, total_len - seq_len):
                input_seq = latent_seq[start_idx:start_idx + seq_len]
                target_seq = latent_seq[start_idx + 1:start_idx + seq_len + 1]
                self.sequences.append((input_seq, target_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        # numpy -> torch tensor 변환
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)
