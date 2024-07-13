import re
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from PIL import Image

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


def check_ddp_consistency(module: nn.Module, ignore_regex: str = None):
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert torch.all(tensor == other), fullname

class VideoDataset(Dataset):
    def __init__(self, folder_path, transform=None, num_frames=40, sample_frames=25):
        self.folder_path = folder_path
        self.transform = transform
        self.num_frames = num_frames
        self.sample_frames = sample_frames
        self.video_folders = sorted(os.listdir(folder_path))

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.folder_path, video_folder)
        images = self.load_sequence(video_path)
        images= images.to(torch.float32) / 127.5 - 1.0
        actions = self.load_actions(video_path)
        return images, actions

    def load_sequence(self, folder_path):
        file_names = sorted([fn for fn in os.listdir(folder_path) if fn.endswith('.png')])
        total_frames = len(file_names)
        
        if total_frames < self.num_frames:
            raise ValueError(f"Not enough frames in {folder_path}. Expected at least {self.num_frames}, found {total_frames}.")
        
        start_idx = random.randint(0, total_frames - self.sample_frames)
        sampled_file_names = file_names[start_idx:start_idx + self.sample_frames]
        
        images = []
        for file_name in sampled_file_names:
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        return torch.stack(images, dim=0)

    def load_actions(self, folder_path):
        npz_file = os.path.join(folder_path, 'action.npy')
        actions = np.load(npz_file)
        return torch.tensor(actions, dtype=torch.float32).squeeze()