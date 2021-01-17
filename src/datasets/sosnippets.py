from __future__ import division, print_function

import os
import random
from warnings import filterwarnings

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.backends import cudnn
# from torch.utils.data import DataLoader

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class SOSnippetsDataset(torch.utils.data.Dataset):

    def __init__(self, img_root_dir, txt_root_dir):
        super(SOSnippetsDataset, self).__init__()
        self.classes = []
        self.targets = []
        self.imgs = []
        self.txts = []

        label_to_target = {}
        i = 0
        for label in sorted(os.listdir(img_root_dir)):
            if os.path.isdir(f'{img_root_dir}/{label}'):
                txt_class_path = f'{txt_root_dir}/{label}'
                img_class_path = f'{img_root_dir}/{label}'
                self.classes += [label]
                label_to_target[label] = i
                i += 1
                for img_path in os.scandir(img_class_path):
                    filename, file_extension = os.path.splitext(img_path.name)
                    if file_extension.lower() in ['.jpg']:
                        txt_path = f'{txt_class_path}/{filename}.ptr' # to change with .txt
                        if not os.path.isfile(txt_path) or not os.path.isfile(img_path.path):
                            raise FileNotFoundError(f'Did not find {txt_path} or {img_path.path}!')
                        self.targets += [label_to_target[label]]
                        self.imgs += [img_path.path]
                        self.txts += [txt_path]

    def __getitem__(self, item):
        # img = TF.to_tensor(Image.open(self.imgs[item]).convert('RGB').resize((384, 384)))
        img = TF.to_tensor(Image.open(self.imgs[item]))
        txt = torch.load(self.txts[item]).type(torch.LongTensor)
        return (img, txt), self.targets[item]

    def __len__(self):
        return len(self.targets)