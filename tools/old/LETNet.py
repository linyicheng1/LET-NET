import cv2
import torch
from torch import nn
from torchvision.models import resnet
from typing import Optional, Callable


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 gate: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = resnet.conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = resnet.conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


# class LETNet(nn.Module):
#     def __init__(self, c1: int = 8, c2: int = 16, grayscale: bool = False):
#         super().__init__()
#         self.gate = nn.ReLU(inplace=True)
#         # ================================== feature encoder
#         if grayscale:
#             self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
#         else:
#             self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
#         self.conv1 = resnet.conv1x1(c1, c2)
#         # ================================== detector and descriptor head
#         self.conv_head = resnet.conv1x1(c2, 4)
#
#     def forward(self, x: torch.Tensor):
#         # ================================== feature encoder
#         x = self.block1(x)
#         x = self.gate(self.conv1(x))
#         # ================================== detector and descriptor head
#         x = self.conv_head(x)
#         scores_map = torch.sigmoid(x[:, 3, :, :]).unsqueeze(1)
#         local_descriptor = torch.sigmoid(x[:, :-1, :, :])
#         return scores_map, local_descriptor
class LETNet(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, grayscale: bool = False):
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        # ================================== feature encoder
        if grayscale:
            self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
        else:
            self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.conv1 = resnet.conv1x1(c1, c2)
        # ================================== detector and descriptor head
        self.conv_head = resnet.conv1x1(c2, 2)

    def forward(self, x: torch.Tensor):
        # ================================== feature encoder
        x = self.block1(x)
        x = self.gate(self.conv1(x))
        # ================================== detector and descriptor head
        x = self.conv_head(x)
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)
        local_descriptor = torch.sigmoid(x[:, :-1, :, :])
        return scores_map, local_descriptor

if __name__ == '__main__':
    import numpy as np
    net = LETNet(c1=8, c2=16, grayscale=True)
    weight = torch.load('./letnet-only_gary.pth',map_location='cpu')

    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in weight.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    net.eval()

    # image = torch.tensor(np.random.random((1, 1, 480, 640)), dtype=torch.float32)
    image = cv2.imread('img.png')
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=2)
    print(image.shape)
    image = torch.from_numpy(image).to(torch.float32).permute(2, 0, 1)[None] / 255.
    print(image.shape)
    with torch.no_grad():
        scores_map, local_descriptor = net(image)
    # 保存模型
    # torch.save(net, 'letnet.pt')
    # 转换成onnx模型(为什么这里是(480, 640)大小,不是((640, 480)).这里主要是onnx架构的问题)
    torch.onnx.export(net, torch.randn(1, 1, 480, 640), 'letnet_out.onnx', verbose=True, opset_version=11,output_names=['score','descriptor'])

    cv2.imshow('local_descriptor',
                (local_descriptor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    cv2.imshow('scores_map', (scores_map[0, 0].cpu().numpy() * 255).astype(np.uint8))
    cv2.waitKey(-1)

