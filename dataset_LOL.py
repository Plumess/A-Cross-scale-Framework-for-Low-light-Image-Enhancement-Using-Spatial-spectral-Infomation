import cv2
from torch.utils.data import Dataset
from glob import glob
import random
import torch
import os
import rawpy
import numpy as np
import torchvision.transforms as TF


def get_len(route, phase):
    if phase == 'train':
        # train_low_data_names = glob(route + 'Low/*.png')
        train_low_data_names = glob('D:/LHY/LLE/Raw_Datasets/LOL/our485/low/*.png')
        train_low_data_names.sort()

        # train_high_data_names = glob(route + 'Normal/*.png')
        train_high_data_names = glob('D:/LHY/LLE/Raw_Datasets/LOL/our485/low/*.png')
        train_high_data_names.sort()

        return len(train_high_data_names), train_low_data_names, train_high_data_names
    elif phase == 'test':
        # test_low_data_names = glob(route + 'Test/Low/*.png')
        test_low_data_names = glob('D:/LHY/LLE/Raw_Datasets/LOL/eval15/low/*.png')
        test_low_data_names.sort()
        # test_high_data_names = glob(route + 'Test/Normal/*.png')
        test_high_data_names = glob('D:/LHY/LLE/Raw_Datasets/LOL/eval15/high/*.png')
        test_high_data_names.sort()
        return len(test_low_data_names), test_low_data_names, test_high_data_names
    # elif phase == 'eval':
    #     eval_low_data_names = glob(route + 'Test/Low/*.png')
    #     eval_low_data_names.sort()
    #     eval_high_data_names = glob(route + 'Test/Normal/*.png')
    #     eval_high_data_names.sort()
    #     return len(eval_high_data_names[0:1]), eval_low_data_names[0:1], eval_high_data_names[0:1]
    else:
        return 0, []


class myDataset(Dataset):
    def __init__(self, route=r'D:/LHY/LLE/Raw_Datasets/LOL/', phase='train', patch_size=256):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size
        self.input_images = [None] * 10000
        self.gt_images = [None] * 10000
        self.num = [0] * 10000
        self.pre = [0] * 10000
        self.len, self.low_names, self.high_names = get_len(route, phase)
        print("train_input_image", len(self.low_names))

        if self.phase == 'train':
            for ids in range(self.len):
                gt = cv2.imread(self.high_names[ids])
                self.gt_images[ids] = np.float32(gt / 255.0)
                low = cv2.imread(self.low_names[ids])
                self.pre[ids] = np.float32(low / 255.0)

    def __getitem__(self, index):
        if self.phase == 'train':
            train_low_data = self.pre[index]
            train_high_data = self.gt_images[index]
            h, w, _ = train_low_data.shape
            x = random.randint(0, h - self.patch_size)
            y = random.randint(0, w - self.patch_size)
            low_im = train_low_data[x:x + self.patch_size, y:y + self.patch_size, :]
            high_im = train_high_data[x:x + self.patch_size, y:y + self.patch_size, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                low_im = np.flip(low_im, axis=0)
                high_im = np.flip(high_im, axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                low_im = np.flip(low_im, axis=1)
                high_im = np.flip(high_im, axis=1)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                low_im = np.transpose(low_im, (1, 0, 2))
                high_im = np.transpose(high_im, (1, 0, 2))
            return TF.ToTensor()(low_im.copy()), TF.ToTensor()(high_im.copy())

        elif self.phase == 'test':
            # low_im = load_images(self.low_names[index])
            low_im = cv2.imread(self.low_names[index])
            high_im = cv2.imread(self.high_names[index])
            low_im = cv2.cvtColor(low_im, cv2.COLOR_BGR2RGB)
            high_im = cv2.cvtColor(high_im, cv2.COLOR_BGR2RGB)
            # low_im = low_im[:, :, ::-1]
            # high_im = high_im[:, :, ::-1]
            # # print(low_im.shape)
            #
            h, w, _ = low_im.shape
            # choose center patch
            x = int(h / 2 - self.patch_size / 2)
            y = int(w / 2 - self.patch_size / 2)
            low_im = low_im[x:x + self.patch_size, y:y + self.patch_size, :]
            high_im = high_im[x:x + self.patch_size, y:y + self.patch_size, :]

            return TF.ToTensor()(low_im), TF.ToTensor()(high_im)
            # return TF.ToTensor()(low_im*ratio),TF.ToTensor()(high_im),int(a),ratio
        # elif self.phase == 'eval':
        #     low_im = cv2.imread(self.low_names[index])
        #     h, w, _ = low_im.shape
        #     gt = cv2.imread(self.high_names[index])
        #     x = int(h / 2 - self.patch_size / 2)
        #     y = int(w / 2 - self.patch_size / 2)
        #     low_im = low_im[x:x + self.patch_size, y:y + self.patch_size, :]
        #     gt = gt[ x:x + self.patch_size,y: y + self.patch_size, :]
        #     return TF.ToTensor()(low_im ), TF.ToTensor()(gt)

    def __len__(self):
        return self.len
