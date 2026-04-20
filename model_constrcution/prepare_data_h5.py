import os
import h5py
import torch
import numpy as np
import librosa
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


SR = 16000
DURATION = 10
MAX_SAMPLES = SR * DURATION  # 160,000 个采样点
FRAME_SIZE = 512
FRAME_SHIFT = 256

REJOIN_LEN = 626 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
NUM_WORKERS = 16 

class MTASSPreprocessDataset(Dataset):
    def __init__(self, dataset_root, split):
        self.root = dataset_root
        self.split = split
        self.mixture_base = os.path.join(dataset_root, split, 'mixture')
        # 仅获取目录
        self.folders = sorted([d for d in os.listdir(self.mixture_base) 
                               if os.path.isdir(os.path.join(self.mixture_base, d))])

    def __len__(self):
        return len(self.folders)

    def load_audio(self, p):
        try:
            wav, _ = librosa.load(p, sr=SR)
            if len(wav) >= MAX_SAMPLES: 
                return wav[:MAX_SAMPLES]
            return np.pad(wav, (0, MAX_SAMPLES - len(wav)), 'constant')
        except Exception as e: 
            print(f"读取错误 {p}: {e}")
            return np.zeros(MAX_SAMPLES)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        idx_str = folder.replace('mixture', '')
        
        def get_p(sub, prefix):
            d = os.path.join(self.root, self.split, sub, f'{prefix}{idx_str}')
            files = [f for f in os.listdir(d) if f.endswith('.wav')]
            if not files: return None
            return os.path.join(d, files[0])

        mix = self.load_audio(get_p('mixture', 'mixture'))
        s1 = self.load_audio(get_p('speech', 'speech'))
        s2 = self.load_audio(get_p('music', 'music'))
        s3 = self.load_audio(get_p('noise', 'noise'))
        
        return (mix.astype(np.float32), 
                s1.astype(np.float32), 
                s2.astype(np.float32), 
                s3.astype(np.float32))

def process_fast(dataset_root, split='train'):
    save_path = os.path.join(dataset_root, f'{split}_new_ready.h5')
    dataset = MTASSPreprocessDataset(dataset_root, split)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                        pin_memory=True, shuffle=False)
    
    win_gpu = torch.hamming_window(FRAME_SIZE, periodic=True).to(DEVICE)

    with h5py.File(save_path, 'w') as h5f:
        h5f.create_dataset('X1', shape=(0, 514, REJOIN_LEN), maxshape=(None, 514, REJOIN_LEN), dtype='float32', compression='gzip')
        for i in range(1, 4):
            h5f.create_dataset(f'Y{i}', shape=(0, 514, REJOIN_LEN), maxshape=(None, 514, REJOIN_LEN), dtype='float32', compression='gzip')
            h5f.create_dataset(f'R{i}', shape=(0, MAX_SAMPLES), maxshape=(None, MAX_SAMPLES), dtype='float32', compression='gzip')

        print(f">>> 启动 GPU 加速预处理: {split} 集 (center=True, 帧数={REJOIN_LEN})")
        
        for batch in tqdm(loader):
            mix_batch, s1_batch, s2_batch, s3_batch = [b.to(DEVICE) for b in batch]
            
            with torch.no_grad():
                def get_batch_ri(wav_gpu):

                    spec = torch.stft(
                        wav_gpu,
                        n_fft=FRAME_SIZE,
                        hop_length=FRAME_SHIFT,
                        win_length=FRAME_SIZE,
                        window=win_gpu,
                        center=True,
                        normalized=False,
                        onesided=True,
                        return_complex=True
                    )

                    return torch.cat([spec.real, spec.imag], dim=1).cpu().numpy()

                mix_ri = get_batch_ri(mix_batch)
                y1_ri = get_batch_ri(s1_batch)
                y2_ri = get_batch_ri(s2_batch)
                y3_ri = get_batch_ri(s3_batch)


            curr = h5f['X1'].shape[0]
            count = mix_batch.shape[0]
            
            h5f['X1'].resize(curr + count, axis=0)
            h5f['X1'][curr : curr + count] = mix_ri
            

            for i, (y, r) in enumerate(zip([y1_ri, y2_ri, y3_ri], [s1_batch, s2_batch, s3_batch])):
                h5f[f'Y{i+1}'].resize(curr + count, axis=0)
                h5f[f'Y{i+1}'][curr : curr + count] = y
                
                h5f[f'R{i+1}'].resize(curr + count, axis=0)
                h5f[f'R{i+1}'][curr : curr + count] = r.cpu().numpy()

if __name__ == '__main__':

    root_path = '/work104/lishuailong/dataset/MTASS-dataset-16K'
    
    for s in ['train', 'dev', 'test']:
        if os.path.exists(os.path.join(root_path, s)):
            process_fast(root_path, s)
        else:
            print(f"跳过 {s}: 路径不存在")