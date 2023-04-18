from torch.utils.data import Dataset
from glob import glob
import random
import os
import rawpy
import numpy as np
import torchvision.transforms as TF


def get_len(route, phase):
    if phase == 'train':
        train_low_data_names = glob(route + 'short/0*_00*.ARW')
        # train_low_data_names1 = glob(route + 'short/200*_00*.ARW')
        # train_low_data_names = train_low_data_names+train_low_data_names1
        train_low_data_names.sort()
        train_high_data_names = glob(route + 'long/0*_00*.ARW')
        # train_high_data_names1 = glob(route + 'long/200*_00*.ARW')
        # train_high_data_names = train_high_data_names+train_high_data_names1
        train_high_data_names.sort()

        return len(train_high_data_names), train_low_data_names, train_high_data_names
    elif phase == 'test':
        test_low_data_names = glob(route + 'short/1*_00*.ARW')
        test_low_data_names.sort()
        test_high_data_names = glob(route + 'long/1*_00*.ARW')
        test_high_data_names.sort()
        return len(test_low_data_names), test_low_data_names, test_high_data_names
    elif phase == 'eval':
        eval_low_data_names = glob(route + 'short/2*.ARW')
        eval_low_data_names.sort()
        eval_high_data_names = glob(route + 'long/2*.ARW')
        eval_high_data_names.sort()
        return len(eval_high_data_names[0:1]), eval_low_data_names[0:1], eval_high_data_names[0:1]
    else:
        return 0, []


def pack_raw(raw):  # Sony
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


class myDataset(Dataset):
    def __init__(self, route='D:/LHY/LLE/Raw_Datasets/Sony/', phase='train', patch_size=512):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size
        self.input_images = [None] * 500
        self.gt_images = [None] * 250
        self.num = [0] * 250
        self.pre = [0] * 250
        self.len, self.low_names, self.high_names = get_len(route, phase)
        print(len(self.low_names))
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
            high_im = train_high_data[2 * x:2 * x + self.patch_size * 2, 2 * y:2 * y + self.patch_size * 2, :]
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
            low_im = pack_raw(rawpy.imread(self.low_names[index]))
            h, w, _ = low_im.shape

            # choose center patch
            x = int(h / 2 - self.patch_size / 2)
            y = int(w / 2 - self.patch_size / 2)
            low_im = low_im[x:x + self.patch_size, y:y + self.patch_size, :]
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
                    high_im = high_im[2 * x:2 * x + self.patch_size * 2, 2 * y:2 * y + self.patch_size * 2, :]
                    return TF.ToTensor()(low_im * ratio), TF.ToTensor()(high_im), int(idstr[0:5]), ratio

        elif self.phase == 'eval':
            low_im = pack_raw(rawpy.imread(self.low_names[index]))
            h, w, _ = low_im.shape
            in_exposure = float(os.path.basename(self.low_names[index])[9:-5])
            gt_raw = rawpy.imread(self.high_names[index])
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            high_im = np.float32(im / 65535.0)
            gt_exposure = float(os.path.basename(self.high_names[index])[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            x = int(h / 2 - self.patch_size / 2)
            y = int(w / 2 - self.patch_size / 2)
            low_im = low_im[x:x + self.patch_size, y:y + self.patch_size, :]
            high_im = high_im[2 * x:2 * x + self.patch_size * 2, 2 * y:2 * y + self.patch_size * 2, :]
            return TF.ToTensor()(low_im * ratio), TF.ToTensor()(high_im)

    def __len__(self):
        return self.len
