from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average = True, max_val = 255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
    def _ssim(self, img1, img2, size_average = True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels,).cuda())
        mcs = Variable(torch.Tensor(levels,).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2):

        return self.ms_ssim(img1, img2)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32")/255.0

def save_images(filepath, result_1, result_2 = None):
    #result_1 = np.array(result_1)
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)
    cat_image = cat_image.transpose(1, 2, 0)
    # cat_image = cat_image[:, :, ::-1]  # for LOL
    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))

    #im = unloader(result_1)
    im.save(filepath, 'png')

def tv_loss(S,I):
    S_gray = 0.299*S[:,0,:,:]+0.587*S[:,1,:,:]+0.114*S[:,2,:,:]
    S_gray = S_gray.unsqueeze(1)
    L = torch.log(S_gray+0.0001)
    dx = L[:,:,:-1,:-1]-L[:,:,:-1,1:]
    dy = L[:,:,:-1,:-1]-L[:,:,1:,:-1]
    p_alpha = 1.2
    p_lambda = 1.5
    dx = p_lambda/(torch.pow(torch.abs(dx),p_alpha)+0.00001)
    dy = p_lambda/(torch.pow(torch.abs(dy),p_alpha)+0.00001)
    x_loss = dx*((I[:, :, :-1, :-1]-I[:, :, :-1, 1:])**2)
    y_loss = dy*((I[:, :, :-1, :-1]-I[:, :, 1:, :-1])**2)
    tvloss = 0.5*torch.abs(x_loss+y_loss).mean()
    return tvloss
ms_ssim_loss = MS_SSIM(max_val=1)

def ms_ssim(x,y):
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    c1 = 1e-4
    c2 = 0.03
    l = (2*mu_x*mu_y+c1)/(mu_x**2+mu_y**2+c1)
    scale = 1
    for i in range(4):
        resize_x = F.interpolate(x,scale_factor=scale)
        resize_y = F.interpolate(y,scale_factor=scale)
        scale = scale*0.5
        sigma_x = torch.var(resize_x)
        sigma_y = torch.var(resize_y)
        cov = torch.mean((resize_x-torch.mean(resize_x))*(resize_y-torch.mean(resize_y)))
        s = (2*cov+c2)/(sigma_x+sigma_y+c2)
        l = l*s
    return 1.0-l

def gradient(x):
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = torch.abs(r - l), torch.abs(b - t)
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    return dx, dy

def gdl_loss(S_low,high_im):
    dy_gen, dx_gen = gradient(S_low)
    dy_gt, dx_gt = gradient(high_im)
    grad_loss = torch.mean(torch.abs(dy_gen - dy_gt) + torch.abs(dx_gen - dx_gt))
    return grad_loss

def final_loss(S_low,high_im):
    alpha = 0.8
    beta = 0.2
    # theta = 0.2
    pix_loss = torch.abs(S_low-high_im).mean()
    ssim_loss = 1-ms_ssim_loss(S_low,high_im)
    # grad_loss = gdl_loss(S_low, high_im)
    # loss1 = (alpha*pix_loss + beta*ssim_loss + theta*grad_loss)/10
    loss1 = (alpha * pix_loss + beta * ssim_loss) / 10
    # f1 = torch.fft.fft2(S_low, dim=(-2, -1))
    # f1 = torch.stack((f1.real, f1.imag), -1)
    # f2 = torch.fft.fft2(high_im, dim=(-2, -1))
    # f2 = torch.stack((f2.real, f2.imag), -1)
    # loss2 = torch.abs(f1 - f2).mean() / 1000
    # loss = loss1 + loss2
    return loss1