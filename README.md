# README.md

# A Cross-scale Framework for Low-light Image Enhancement Using Spatial-spectral Infomation

This is a pytorch implementation of the [CE Paper](https://www.sciencedirect.com/science/article/pii/S0045790623000332?dgcid=coauthor) in Computers & Electrical Engineering 2023 and the [VCIP Paper](https://ieeexplore.ieee.org/abstract/document/10008898) in 2022 IEEE VCIP

This is the link of pre-trained model in [GoogleDrive](https://drive.google.com/file/d/1_FIP_bz29hSXGJ4QBesm_5za1KWKGiYe/view?usp=share_link) or [BaiduNetDisk](https://pan.baidu.com/s/1a7w0ynDDQCIpc6S03JGPhg)(23tc).

This is the link of a selection of test images used to verify that the code has been deployed correctly, including images from the Sony, Fuji and LOL datasets in  [GoogleDrive](https://drive.google.com/file/d/12be57WxrCl6gzTOREav47paDwPg9OyMI/view?usp=share_link) or [BaiduNetDisk](https://pan.baidu.com/s/1fdZEqYm9zn-20VsVXz5z6g)(9pru).

---

### Requirements

- numpy~=1.19.5
- torch~=1.8.2+cu111
- tensorboardx~=2.4
- opencv-python~=4.7.0.72
- scipy~=1.7.1
- timm~=0.4.12
- einops~=0.3.2
- pillow~=8.3.2
- rawpy~=0.18.0
- torchvision~=0.9.2+cu111

---

### Description

- *main.py* is for the RAW dataset (Sony and Fuji in SID);
- *model.py* is ours model in the paper called “ours” for dataset (Sony and Fuji in SID);
- *model_ext.py* is our larger model in the paper called “ours++” for dataset (Sony and Fuji in SID);
- *dataset_sony.py & dataset_fuji.py* are the codes for preprocessing and loading sony/fuji dataset.

Additionally:

- *main_LOL.py* is for the LOL dataset;
- *model_LOL.py* is ours model in the paper called “ours” for LOL dataset;
- *dataset_LOL.py* is the code for preprocessing and loading LOL dataset.
- *psnr.py & utils.py* are some auxiliary codes

Since window attention is in a square window, the original image is cropped to a fixed size square patch for both testing and training in the code. If you wish to train and test at full size, please refer to the modifications to the SpectralTransform, LeFF, LeWinTransformer, Input, Output and ResLeWinTransformerLayer's forward methods in *model_fullSize.py*, And the hiding of the crop operation in *dataset_sony_fullSize.py*.

---

### Test

1. Download the pre-trained model
2. Prepare test dataset
3. Run the corresponding main.py

---

### Train

1. Prepare the dataset
2. Train the network. That's all.

---

### Citation

If you use our code, please cite our paper.

---

### Notes

- You need to change the dataset path when testing or training your images, and you need to pay attention to the pack function in dataset.py when you test raw images captured by other devices.
- If you have any questions and do not receive a timely response, you can also contact the first author of the paper: [https://github.com/Volodymyr233](https://github.com/Volodymyr233)
