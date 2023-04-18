import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import argparse
# from model_ext import EnhanceNet
from model import EnhanceNet
# from dataset_sony import myDataset
from dataset_fuji import myDataset
from torch.utils.data import DataLoader
from utils import save_images, final_loss
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--epoch', dest='epoch', type=int, default=4001, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=512, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20,
                    help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--dir', dest='dir', default='ours_ext_fuji', help='directory')
parser.add_argument('--test_dir', dest='test_dir', default=r'D:/LHY/LLE/Present/datasets/Fuji/short',
                    # D:/LHY/LLE/Raw_Datasets/Fuji/short
                    # D:/LHY/LLE/Raw_Datasets/Sony/short
                    help='directory for testing inputs')
parser.add_argument('--train_dir', dest='train_dir', default=r'D:/LHY/LLE/Present/datasets/Fuji/',
                    # D:/LHY/LLE/Raw_Datasets/Fuji/
                    # D:/LHY/LLE/Raw_Datasets/Sony/
                    help='directory for training inputs')  # 路径后面要有斜杆
parser.add_argument('--continue_train', dest='continue_train', default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES  "] = args.gpu_idx
if not os.path.exists("checkpoint_" + args.dir):
    os.makedirs("checkpoint_" + args.dir)
if not os.path.exists("test_results_" + args.dir):
    os.makedirs("test_results_" + args.dir)

enhancenet = EnhanceNet()
print('#parameters: ', sum(param.numel() for param in enhancenet.parameters()))
loss_list = []
eps = 1e-12
if args.use_gpu:
    enhancenet = enhancenet.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True
enhance_optim = torch.optim.Adam(enhancenet.parameters(), lr=args.start_lr)
w_log = SummaryWriter(log_dir='logs')


def eval():
    eval_set = myDataset(args.train_dir, 'eval', args.patch_size)
    with torch.no_grad():
        dataLoader = DataLoader(eval_set, batch_size=1)
        for (step, data) in enumerate(dataLoader):
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
            print("low,high", low_im.shape, high_im.shape)
            S_low = enhancenet(low_im)
            out = S_low
            print("out", out.shape)
            mse = torch.abs(high_im - out).mean()
            print(float(mse))
            torch.cuda.empty_cache()


def train():
    f = open(args.dir + '.txt', 'w')
    f.close()
    start_epoch = 0
    print("continue_train:", args.continue_train)
    if args.continue_train:
        checkpoint = torch.load("D:/LHY/LLE/checkpoint_ours_sony/3400_state_final.pth")
        start_epoch = checkpoint['epoch'] + 1
        print("continue_train/start_epoch:", start_epoch)
        if start_epoch < args.epoch:
            print("continue_train/start_epoch:", start_epoch)
            enhancenet.load_state_dict(checkpoint['enhance'])
            enhance_optim.load_state_dict(checkpoint['enhance_optim'])
        else:
            pass
    print("start_epoch:", start_epoch)
    train_set = myDataset(args.train_dir, 'train', args.patch_size)
    print("train set:", len(train_set))
    sum_loss_f = 0.0
    print("train start time: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for epoch in range(start_epoch, args.epoch):
        ep_start = time.time()
        step_num = 0
        sum_loss = 0.0
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        for (_, data) in enumerate(dataloader):
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
            enhance_optim.zero_grad()
            S_low = enhancenet(low_im)
            loss = final_loss(S_low, high_im)
            loss.backward()
            enhance_optim.step()
            sum_loss += float(loss)
            step_num += 1
        print('epoch: %d, loss: %f' % (epoch, sum_loss / step_num))
        sum_loss_f += sum_loss / step_num
        w_log.add_scalar('loss', sum_loss / step_num, epoch)
        if epoch % 100 == 1:
            eval()
        elif epoch == 2000:
            for pa in enhance_optim.param_groups:
                pa['lr'] = args.start_lr / 10.0
        elif epoch == 3000:
            for pa in enhance_optim.param_groups:
                pa['lr'] = args.start_lr / 50.0
        elif epoch == 4000:
            for pa in enhance_optim.param_groups:
                pa['lr'] = args.start_lr / 100.0
        if epoch % 50 == 0:
            f = open(args.dir + '.txt', 'a')
            if epoch > 1:
                f.write(str(sum_loss_f / 50) + '\n')
            sum_loss_f = 0.0
            f.close()
            state = {'enhance': enhancenet.state_dict(), 'enhance_optim': enhance_optim.state_dict(), 'epoch': epoch}
            if not os.path.isdir("checkpoint_" + args.dir):
                os.makedirs("checkpoint_" + args.dir)
            torch.save(state, "checkpoint_" + args.dir + '/%4d_state_final.pth' % epoch)
        ep_cost = time.time() - ep_start
        print("one epoch cost: " + str(ep_cost) + " seconds")
    state = {'enhance': enhancenet.state_dict(), 'enhance_optim': enhance_optim.state_dict(), 'epoch': epoch}
    if epoch % 10 == 0:
        torch.save(state, "checkpoint_" + args.dir + '/%4d_state_final.pth' % epoch)
    w_log.close()


def test():
    # checkpoint = torch.load("D:/LHY/LLE/checkpoint_ours_sony/3400_state_final.pth")
    checkpoint = torch.load("D:/LHY/LLE/checkpoint_ours_fuji/2394_state_final.pth")
    enhancenet.load_state_dict(checkpoint['enhance'], strict=True)
    print('load weights successfully')
    args.batch_size = 1
    test_set = myDataset(args.train_dir, 'test', args.patch_size)
    print('number of test samples: %d' % test_set.len)
    dataLoader = DataLoader(test_set, batch_size=args.batch_size)
    with torch.no_grad():
        for (step, data) in enumerate(dataLoader):
            low_im, high_im, idstr, ratio = data
            low_im = low_im.cuda()
            print("low_in_shape:", np.shape(low_im))
            S_low = enhancenet(low_im)
            out = S_low
            out_cpu = out.cpu()
            out_cpu = np.minimum(out_cpu, 1.0)
            out_cpu = np.maximum(out_cpu, 0.0)
            save_images(os.path.join('test_results_' + args.dir + '/out', '%d_00_%d_out.png' % (idstr, ratio)),
                        out_cpu.detach().numpy())
            save_images(os.path.join('test_results_' + args.dir + '/gt', '%d_00_%d_gt.png' % (idstr, ratio)),
                        high_im.detach().numpy())
            torch.cuda.empty_cache()


def main():
    if args.use_gpu:
        if args.phase == 'train':
            train()
            test()
        elif args.phase == 'test':
            test()
        else:
            print('unknown phase')
    else:
        print('please use gpu')


if __name__ == '__main__':
    main()
