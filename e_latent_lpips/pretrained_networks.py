from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models as tv


class LatentVGG16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, channel_intialize=None):
        super(LatentVGG16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5

        # Implementation Detail
        # 1. change input channels 3 -> 4
        #     Since the exact implementation method is not presented,
        #     the weight of the new channel is specified by copying and using the Red channel
        # 2. Skip Maxpool 3 layer
        #     The exact implementation method is not shown, Remove 3 maxpool layers from the front

        vgg_input_layer = vgg_pretrained_features[0]
        latent_lpips_input_layer = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1)

        vgg_input_layer_weight = vgg_input_layer.weight.data
        vgg_input_layer_bias = vgg_input_layer.bias.data

        latent_lpips_input_weight = torch.zeros_like(latent_lpips_input_layer.weight.data)
        latent_lpips_input_weight[:, :3, :, :] = vgg_input_layer_weight
        latent_lpips_input_weight[:, 3:, :, :] = vgg_input_layer_weight[:, :1, :, :]

        latent_lpips_input_layer.weight.data = latent_lpips_input_weight
        latent_lpips_input_layer.bias.data = vgg_input_layer_bias

        self.slice1.add_module(str(0), latent_lpips_input_layer)

        for x in range(1, 4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(5, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class VGG16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


if __name__ == '__main__':
    model = LatentVGG16(pretrained=True)
    model.eval()

    inputs = torch.randn(1, 4, 64, 64)
    outputs = model(inputs)
    for i in range(5):
        print(outputs[i].size())
