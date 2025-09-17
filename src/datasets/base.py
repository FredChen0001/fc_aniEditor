from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os
import json
class BaseDataset(Dataset):
    def __init__(self, dataset_dir=None, sample_file=None, size=(384, 512), ratio=(3,4), mean=0.5, std=0.5, preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=220):
        self.dataset_dir = dataset_dir
        self.sample_file = sample_file
        self.samples = self.load_sample_file(os.path.join(dataset_dir, sample_file))
        if keep_num > 0:
            self.samples = self.samples[:keep_num]
        self.ratio = ratio
        self.background_color = background_color
        self.size = size
        self.mean = mean
        self.std = std
        self.preview_dir = preview_dir
        self.repeat_len = repeat_len

    def __len__(self):
        if self.repeat_len > 0:
            return self.repeat_len
        return len(self.samples)


    def __getitem__(self, idx):
        raise NotImplemented

    def load_sample_file(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf8') as fp:
            for line in fp.readlines():
                samples.append(json.loads(line))
        return samples

    def normalize(self, img):
        img = TF.to_tensor(img)
        img = (img - self.mean) / self.std
        return img

class DummyDataset(BaseDataset):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir, sample_file='dummy')

    def __getitem__(self, idx):
        return self.samples[idx]

    def load_sample_file(self, file_path):
        return ['dummy data'] * 1000