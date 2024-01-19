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


# copied from torchvision\models\resnet.py#27->BasicBlock
class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


class ALNet(nn.Module):
    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 agg_mode: str = 'cat',  # sum, cat, fpn
                 single_head: bool = True,
                 pe: bool = False,
                 grayscale: bool = False
                 ):
        super().__init__()

        self.gate = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.pe = pe

        # ================================== feature encoder
        if self.pe:
            self.position_encoding = None
            self.block1 = ConvBlock(3 + 8, c1, self.gate, nn.BatchNorm2d)
        else:
            if grayscale:
                self.block1 = ConvBlock(1, c1, self.gate, nn.BatchNorm2d)
            else:
                self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)

        self.block2 = ResBlock(inplanes=c1, planes=c2, stride=1,
                               downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(inplanes=c2, planes=c3, stride=1,
                               downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(inplanes=c3, planes=c4, stride=1,
                               downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate,
                               norm_layer=nn.BatchNorm2d)

        # ================================== feature aggregation
        self.agg_mode = agg_mode
        if self.agg_mode == 'sum' or self.agg_mode == 'fpn':
            self.conv1 = resnet.conv1x1(c1, dim)
            self.conv2 = resnet.conv1x1(c2, dim)
            self.conv3 = resnet.conv1x1(c3, dim)
            self.conv4 = resnet.conv1x1(dim, dim)
        elif self.agg_mode == 'cat':
            self.conv1 = resnet.conv1x1(c1, dim // 4)
            # self.conv1_ = resnet.conv1x1(c1, 6)
            self.conv2 = resnet.conv1x1(c2, dim // 4)
            self.conv3 = resnet.conv1x1(c3, dim // 4)
            self.conv4 = resnet.conv1x1(dim, dim // 4)
        else:
            raise ValueError(f"Unkown aggregation mode: '{self.agg_mode}', should be 'sum' or 'cat'!")

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.conv_head = resnet.conv1x1(dim // 4, 2)

        # ================================== detector and descriptor head
        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = resnet.conv1x1(dim, dim)
        self.convhead2 = resnet.conv1x1(dim, dim)

    def forward(self, image):
        # ================================== feature encoder
        if self.pe:
            x1 = self.position_encoding(image)
            x1 = self.block1(x1)  # B x c1 x H x W
        else:
            x1 = self.block1(image)  # B x c1 x H x W
            # x1 = self.IRblock1(image)
            # x1 = self.IRblock2(x1)
            # x1 = self.dsblock10(image)
            # x1 = self.dsblock11(x1)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32

        # ================================== feature aggregation
        if self.agg_mode == 'sum':
            x1234 = self.gate(self.conv1(x1))  # B x dim x H x W
            x2 = self.gate(self.conv2(x2))  # B x dim x H//2 x W//2
            x1234 = x1234 + self.upsample2(x2)
            x3 = self.gate(self.conv3(x3))  # B x dim x H//8 x W//8
            x1234 = x1234 + self.upsample8(x3)
            x4 = self.gate(self.conv4(x4))  # B x dim x H//32 x W//32
            x1234 = x1234 + self.upsample32(x4)
        elif self.agg_mode == 'fpn':
            x1234 = self.gate(self.conv4(x4))  # B x dim x H//32 x W//32
            x1234 = self.upsample4(x1234)  # B x dim x H//8 x W//8
            x1234 = self.gate(self.conv3(x3) + x1234)  # B x dim x H//8 x W//8
            x1234 = self.upsample4(x1234)  # B x dim x H//2 x W//2
            x1234 = self.gate(self.conv2(x2) + x1234)  # B x dim x H//2 x W//2
            x1234 = self.upsample2(x1234)  # B x dim x H x W
            x1234 = self.gate(self.conv1(x1) + x1234)  # B x dim x H x W
        elif self.agg_mode == 'cat':
            x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
            x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
            x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
            x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
            x2_up = self.upsample2(x2)  # B x dim//4 x H x W
            x3_up = self.upsample8(x3)  # B x dim//4 x H x W
            x4_up = self.upsample32(x4)  # B x dim//4 x H x W
            x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        else:
            raise ValueError(f"Unkown aggregation mode: '{self.agg_mode}', should be 'sum' or 'cat'!")

        # ================================== detector and descriptor head
        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        x = self.convhead2(x1234)  # B x dim+1 x H x W

        # descriptor_map = x[:, :-1, :, :]
        # scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)
        descriptor_map = x
        head = self.conv_head(x1)
        scores_map = torch.sigmoid(head[:, -1, :, :]).unsqueeze(1)
        return scores_map, descriptor_map


if __name__ == '__main__':
    import cv2
    import numpy as np
    net = ALNet(c1=8, c2=16, c3=32, c4=64, dim=64,
                agg_mode='cat',
                single_head=True,
                pe=False,
                grayscale=True)

    weight = torch.load('./model.pt',map_location='cpu')
    net.load_state_dict(weight)

    net.eval()
    #
    image = cv2.imread('./img.png')
    cv2.imshow("原图",image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (640, 480))
    cv2.imshow("resize",image)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.permute(0, 1, 3, 2)
    with torch.no_grad():
        scores_map, local_descriptor = net(image_tensor)

        # 转换成onnx模型(为什么这里是(480, 640)大小,不是((640, 480)).这里主要是onnx架构的问题)
    torch.onnx.export(net, torch.randn(1, 1, 480, 640), 'ALIKE.onnx', verbose=True, opset_version=11,output_names=['score','descriptor'])

    cv2.imshow('scores_map', (scores_map[0, 0].cpu().numpy() * 255).astype(np.uint8))
    cv2.waitKey(-1)

