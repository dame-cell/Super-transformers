from torch.utils.data import Dataset
import torch 

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,)).item()
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()

        input_seq = full_seq[:-1]  # First seq_len tokens
        target_seq = full_seq[1:]  # Shifted by one token

        return input_seq.cuda(), target_seq.cuda()  # Return two tensors

    def __len__(self):
        return self.data.size(0) // self.seq_len

