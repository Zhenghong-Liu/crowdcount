import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
# from density_jianhua import GhostNet
from jianxiaowucha_backend import GhostNet
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from torch.nn import functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from matplotlib import cm as c

import gradio as gr

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])

model = GhostNet()
# model = CSRNet()
model = model.cuda()
checkpoint = torch.load('./checkpoint/2.19 MAE=25.044 model-jianxiaowucha-backend/300ghost_checkpoint.pth.tar')
# checkpoint = torch.load('./checkpoint/csrmodel/0ghost_model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])


def processImg(img_path):
    # img = 255.0 * F.to_tensor(Image.open(img_path).convert('RGB'))
    #
    # img[0, :, :] = img[0, :, :] - 92.8207477031
    # img[1, :, :] = img[1, :, :] - 95.2757037428
    # img[2, :, :] = img[2, :, :] - 104.877445883
    # img = img.cuda()
    img = transform(Image.open(img_path).convert('RGB')).cuda()
    output = model(img.unsqueeze(0))
    count_num = int(output.detach().cpu().sum().numpy())
    print("人数：", count_num)
    output = F.interpolate(output, scale_factor=4, mode='bilinear')
    # 将输出的密度图转换为numpy数组，并将其转换为uint8类型，再将其转换为彩色图像
    # output = output.cpu().data.numpy()
    # output = output[0][0]
    # output = np.array(output, dtype=np.uint8)
    # output = Image.fromarray(output)
    temp = np.asarray(output.detach().cpu())
    # 去除前两个维度
    # temp = temp.squeeze(0).squeeze(0)
    # 通过c.jet将密度图转换为彩色图像,并返回
    output = c.jet(temp)[..., :3]

    # print(temp.shape)
    # print(camp.shape)

    output = output.squeeze(0).squeeze(0)

    # plt.imshow(output)
    # plt.show()
    output = Image.fromarray(np.uint8(output * 255))
    return [output, count_num]


if __name__ == '__main__':
    demo = gr.Interface(processImg,
                        inputs=gr.inputs.Image(type="filepath"),
                        outputs=["image", "text"],
                        title="Crowd Counting",
                        examples=[["./Shanghai/part_A_final/train_data/images/IMG_102.jpg"],
                                  ["./Shanghai/part_A_final/train_data/images/IMG_197.jpg"],
                                  ["./Shanghai/part_A_final/train_data/images/IMG_66.jpg"],
                                  ["./Shanghai/part_A_final/train_data/images/IMG_50.jpg"]])
    demo.launch()
    # processImg("./Shanghai/part_A_final/test_data/images/IMG_1.jpg")
