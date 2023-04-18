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
        train_low_data_names = glob(route + 'short/0*_00*.RAF')
        train_low_data_names.sort()

        train_high_data_names = glob(route + 'long/0*_00*.RAF')
        train_high_data_names.sort()

        return len(train_high_data_names), train_low_data_names, train_high_data_names
    elif phase == 'test':
        test_low_data_names = glob(route + 'short/1*_00*.RAF')
        # test_low_data_names = glob(route + 'short/*.ARW')
        test_low_data_names.sort()
        test_high_data_names = glob(route + 'long/1*_00*.RAF')
        # test_high_data_names = glob(route + 'long/*.ARW')
        test_high_data_names.sort()
        return len(test_low_data_names), test_low_data_names, test_high_data_names
    elif phase == 'eval':
        eval_low_data_names = glob(route + 'short/2*.RAF')
        eval_low_data_names.sort()
        eval_high_data_names = glob(route + 'long/2*.RAF')
        eval_high_data_names.sort()
        return len(eval_high_data_names[0:1]), eval_low_data_names[0:1], eval_high_data_names[0:1]
    else:
        return 0, []


def pack_raw(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
    return out.astype(np.float32)


class myDataset(Dataset):
    def __init__(self, route=r'D:/LHY/LLE/Raw_Datasets/Fuji/', phase='train', patch_size=48):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size
        self.input_images = [None] * 500
        self.gt_images = [None] * 250
        self.num = [0] * 250
        self.pre = [0] * 250
        self.len, self.low_names, self.high_names = get_len(route, phase)
        print("train_input_image", len(self.low_names))

        if self.phase == 'train':

            j = 0
            for ids in range(self.len):
                gt_raw = rawpy.imread(self.high_names[ids])

                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                self.gt_images[ids] = np.float32(im / 65535.0)
                idstr = os.path.basename(self.high_names[ids])[0:5]
                gt_exposure = float(os.path.basename(self.high_names[ids])[9:-5])
                self.pre[ids] = j
                while j < len(self.low_names):
                    if idstr == os.path.basename(self.low_names[j])[0:5]:
                        in_exposure = float(os.path.basename(self.low_names[j])[9:-5])
                        ratio = min(gt_exposure / in_exposure, 300)
                        raw = rawpy.imread(self.low_names[j])
                        self.input_images[j] = pack_raw(raw) * ratio
                        self.num[ids] += 1
                        j += 1
                    else:
                        break

    def __getitem__(self, index):
        if self.phase == 'train':
            randomx = random.randint(0, self.num[index] - 1)
            train_low_data = self.input_images[self.pre[index] + randomx]
            train_high_data = self.gt_images[index]
            h, w, _ = train_low_data.shape
            x = random.randint(0, h - self.patch_size)
            y = random.randint(0, w - self.patch_size)
            low_im = train_low_data[x:x + self.patch_size, y:y + self.patch_size, :]
            high_im = train_high_data[3 * x:3 * x + self.patch_size * 3, 3 * y:3 * y + self.patch_size * 3, :]

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
            low_im = pack_raw(rawpy.imread(self.low_names[index]))
            idstr = os.path.basename(self.low_names[index])
            for ids in range(len(self.high_names)):
                tepstr = os.path.basename(self.high_names[ids])
                if idstr[0:5] == tepstr[0:5]:
                    in_exposure = float(idstr[9:-5])
                    gt_exposure = float(tepstr[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    gt_raw = rawpy.imread(self.high_names[ids])
                    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                    high_im = np.float32(im / 65535.0)

                    h, w, _ = low_im.shape

                    x = int(h / 2 - self.patch_size / 2)
                    y = int(w / 2 - self.patch_size / 2)

                    low_im = low_im[x:x + self.patch_size, y:y + self.patch_size, :]
                    high_im = high_im[3 * x:3 * x + self.patch_size * 3, 3 * y:3 * y + self.patch_size * 3, :]

                    return TF.ToTensor()(low_im * ratio), TF.ToTensor()(high_im), int(idstr[0:5]), ratio
                    # return TF.ToTensor()(low_im*ratio),TF.ToTensor()(high_im),int(a),ratio
        elif self.phase == 'eval':
            low_im = pack_raw(rawpy.imread(self.low_names[index]))
            in_exposure = float(os.path.basename(self.low_names[index])[9:-5])
            gt_raw = rawpy.imread(self.high_names[index])
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            high_im = np.float32(im / 65535.0)
            gt_exposure = float(os.path.basename(self.high_names[index])[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            h, w, _ = low_im.shape
            x = random.randint(0, h - self.patch_size)
            y = random.randint(0, w - self.patch_size)
            low_im = low_im[x:x + self.patch_size, y:y + self.patch_size, :]
            high_im = high_im[3 * x:3 * x + self.patch_size * 3, 3 * y:3 * y + self.patch_size * 3, :]

            return TF.ToTensor()(low_im * ratio), TF.ToTensor()(high_im)

    def __len__(self):
        return self.len
