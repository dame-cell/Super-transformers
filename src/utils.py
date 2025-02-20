import torch 
import numpy as np 
from torch.utils.data import Dataset 

class GPTDatasetV1(Dataset):
    def __init__(self, npz_file_path):
        # Load precomputed inputs and targets from .npz file
        data = np.load(npz_file_path)
        self.inputs = data['inputs']
        self.targets = data['targets']
        
        self.num_samples = len(self.inputs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_chunk = self.inputs[idx]
        target_chunk = self.targets[idx]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)