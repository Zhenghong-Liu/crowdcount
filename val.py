from jianxiaowucha_backend import GhostNet

from torch.nn import functional as F

from image import *

import torch
from matplotlib import cm as c

import gradio as gr

from torchvision import  transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

model = GhostNet()

checkpoint = torch.load('./checkpoint/2.19 MAE=25.044 model-jianxiaowucha-backend/300ghost_checkpoint.pth.tar')
# checkpoint = torch.load('./checkpoint/csrmodel/0ghost_model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

def processImg(img):

    # img = transform(Image.open(img_path).convert('RGB'))
    try:
        print("img shape:",img.shape)
        img = transform(img)
        output = model(img.unsqueeze(0))
        count_num = int(output.detach().cpu().sum().numpy())
        print("人数：", count_num)
        output = F.interpolate(output, scale_factor=4, mode='bilinear')

        temp = np.asarray(output.detach())
        #去除前两个维度
        # temp = temp.squeeze(0).squeeze(0)
        #通过c.jet将密度图转换为彩色图像,并返回
        output = c.jet(temp)[..., :3]

        output = output.squeeze(0).squeeze(0)

        output = Image.fromarray(np.uint8(output*255))
        return [output, count_num]
    except:
        print("error")
        return [img, 0]

if __name__ == '__main__':
    demo =gr.Interface(processImg,
                       inputs="image",
                       outputs=["image","text"],
                       title="Crowd Counting",
                       examples=[["./Shanghai/part_A_final/train_data/images/IMG_102.jpg"],
                                 ["./Shanghai/part_A_final/train_data/images/IMG_197.jpg"],
                                 ["./Shanghai/part_A_final/train_data/images/IMG_66.jpg"],
                                 ["./Shanghai/part_A_final/train_data/images/IMG_50.jpg"]])
    demo.launch()
    # processImg("./Shanghai/part_A_final/test_data/images/IMG_1.jpg")







