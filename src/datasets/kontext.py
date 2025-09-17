from src.datasets.base import BaseDataset
import os
import json
import numpy as np
from PIL import Image
from src.utils.image_utils import pad_to_ratio, crop_to_ratio

class T2I(BaseDataset):
    def __init__(self, dataset_dir, sample_file, img_dir, target_resolution_multiplier=1.5, long_edge=2048, size=(576, 768), ratio=(3, 4), preview_dir=None,
                 repeat_len=0, keep_num=0, background_color=240):
        super(T2I, self).__init__(dataset_dir=dataset_dir, sample_file=sample_file,ratio=ratio,
                                       size=size, preview_dir=preview_dir, repeat_len=repeat_len, keep_num=keep_num,
                                       background_color=background_color)
        self.img_dir = img_dir
        self.preferred_resolution = [
            (33, 134), #0.246
            (34, 130),
            (35, 126),
            (36, 122),
            (37, 118),
            (38, 114),
            (39, 110),
            (40, 106),
            (41, 102),
            (42, 98),
            (42, 94),
            (43, 94),
            (43, 91),
            (45, 91),
            (45, 87),
            (47, 87),
            (47, 83),
            (50, 83),
            (50, 78),
            (52, 78),
            (52, 74),
            (55, 74),
            (55, 69),
            (59, 69),
            (59, 64),
            (64, 64), # 1
            (64, 59),
            (69, 59),
            (69, 55),
            (74, 55),
            (74, 52),
            (78, 52),
            (78, 50),
            (83, 50),
            (83, 47),
            (87, 47),
            (87, 45),
            (91, 45),
            (91, 43),
            (94, 43),
            (94, 42),
            (98, 42),
            (102, 41),
            (106, 40),
            (110, 39),
            (114, 38),
            (118, 37),
            (122, 36),
            (126, 35),
            (130, 34),
            (134, 33) # 4.06
        ]
        self.target_resolution_factor = 16 * target_resolution_multiplier
        self.vae_factor = 16
        self.long_edge = long_edge
        self.prompt_prefix = 'Create an animation,'

    def find_nearset_resolution(self, ratio):
        ratio_delta = [abs(ratio - preferred[0]/preferred[1]) for preferred in self.preferred_resolution]
        min_value = min(ratio_delta)
        min_index = ratio_delta.index(min_value)
        return self.preferred_resolution[min_index]



    def __getitem__(self, idx):
        if self.repeat_len > 0:
            idx = idx % len(self.samples)
        sample = self.samples[idx]
        file = sample['file_new']
        file_path = os.path.join(self.img_dir, file)
        try:
            img = self.load_img(file_path, self.target_resolution_factor, 'crop')
        except Exception as e:
            print(e)
            img = Image.fromarray(np.zeros([self.size[1], self.size[0], 3], dtype=np.uint8) + self.background_color)

        prompt = self.gen_prompt(sample)
        target = self.normalize(img)

        example = {}
        example['target'] = target
        example['prompt'] = prompt
        return example




    def gen_prompt(self, sample):
        prompt = self.prompt_prefix + '{},tag:{},favorite count:{}'.format(sample['caption_en'], sample['tag_string'], sample['fav_count'])
        return prompt




    def load_img(self, img_path, resolution_factor, ratio_mode='crop'):
        assert ratio_mode in ['crop', 'pad']
        img = Image.open(img_path).convert('RGB')
        target_ratio = self.find_nearset_resolution(img.size[0]/img.size[1])
        if ratio_mode == 'pad':
            img = Image.fromarray(pad_to_ratio(np.asarray(img), target_ratio, self.background_color))
        else:
            img = crop_to_ratio(img, target_ratio)
        img_w = int((target_ratio[0] * resolution_factor) // self.vae_factor * self.vae_factor)
        img_h = int((target_ratio[1] * resolution_factor) // self.vae_factor * self.vae_factor)
        img = img.resize((img_w, img_h), Image.LANCZOS)
        return img
