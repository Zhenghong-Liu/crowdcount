from ghostnet import ghostnet
import torch.nn as nn
import torch
import numpy as np
class GhostNet(nn.Module):
    def __init__(self,pretrained=True):
        super(GhostNet, self).__init__()
        model = ghostnet()
        if pretrained:
            pass
        self.model = model

        # self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend_feat  = [512, 512, 512,256,256,256,128,128,64,64]
        self.backend = make_layers(self.backend_feat, in_channels=40, dilation=True)

        # self.backend = nn.Sequential(
        #     nn.Conv2d(160, 64, kernel_size=3, padding=2, dilation=2)
        # )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self,x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)

        for idx,block in enumerate(self.model.blocks):
            x = block(x)
            if idx in [4]:
                #print("befor backend:",x.shape)
                x = self.backend(x)
                x = self.output_layer(x)
                return x
        # return feature_maps

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

if __name__ == '__main__':
    model = GhostNet()
    model.eval()
    print(model)
    feature_maps = []
    input = torch.randn(3,3,768,1024)
    s = input[:,:,slice(0,100)]
    # print(s.shape)
    y = model(input)
    # y_n = np.numpy(y)
    # print(y_n)
    print("after backend",y.shape)
    # print(len(y),len(y[0][0]),len(y[0][0][0]),len(y[0][0][0][0]))
# def cov2d(filter_in,filter_out,kernel_size):