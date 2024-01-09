import torch
import numpy as np
def to_homogeneous(kpts):
    print('to_homogeneous')

def warp_homography(kpts0, params):
    print('warp_homography')

class IlluConsLoss(object):
    """ Illumination consistency loss between warp and img matching """

    def __init__(self):
        super().__init__()

    def __call__(self, pred0, pred1, params):
        b, c, h, w = pred0['local_descriptor'].shape
        device = pred0['local_descriptor'].device
        local_desc0 = pred0['local_descriptor']
        local_desc1 = pred1['local_descriptor']
        loss_mean = 0
        CNT = 0

        xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w)).to(device)
        xx = torch.flatten(xx).view(-1, 1)
        yy = torch.flatten(yy).view(-1, 1)
        kpts0, kpts1 = torch.cat((xx, yy), 1)
        kpts0, kpts01, _, _ = warp_homography(kpts0, params)
        kpts1, kpts10, _, _ = warp_homography(kpts1, params)

        for idx in range(b):

            local_desc_kpts0 = torch.nn.functional.grid_sample(local_desc0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            local_desc_kpts1 = torch.nn.functional.grid_sample(local_desc1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            local_desc_kpts01 = torch.nn.functional.grid_sample(local_desc1.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            local_desc_kpts10 = torch.nn.functional.grid_sample(local_desc0.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                                mode='bilinear', align_corners=True)[0, 0, 0, :]

            illu_differ0 = torch.norm(local_desc_kpts0 - local_desc_kpts01, dim=1).squeeze()
            illu_differ1 = torch.norm(local_desc_kpts1 - local_desc_kpts10, dim=1).squeeze()

            ill_loss0 = torch.mean(illu_differ0, dim=0)
            ill_loss1 = torch.mean(illu_differ1, dim=0)

            loss_mean = loss_mean + ((ill_loss0 + ill_loss1) / 2)
            CNT = CNT + 1

        loss_mean = loss_mean / CNT if CNT != 0 else local_desc0.new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean

class InfoEntropyLoss(object):
    """ Information entropy loss between warp and img matching """

    def image_entropy(self, image_tensor):
        scaled_image = image_tensor / 255.0
        histograms = [torch.histc(scaled_image[c], bins=256, min=0, max=1) for c in range(image_tensor.shape[0])]
        pixel_probabilities = [histogram / histogram.sum() for histogram in histograms]
        channel_entropies = [-torch.sum(prob * torch.log2(prob + 1e-10)) for prob in pixel_probabilities]
        return sum(channel_entropies)

    def __init__(self):
        super().__init__()

    def __call__(self, pred0, pred1, batch):
        b, c, h, w = batch['image0'].shape
        img0 = batch['image0'][0]
        img1 = batch['image1'][0]
        local_desc0 = pred0['local_descriptor']
        local_desc1 = pred1['local_descriptor']
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            entropy_img0 = self.image_entropy(img0)
            entropy_img1 = self.image_entropy(img1)
            entropy_local_desc0 = self.image_entropy(local_desc0)
            entropy_local_desc1 = self.image_entropy(local_desc1)

            loss_mean = loss_mean + torch.mean(abs(entropy_img0 - entropy_local_desc0) + abs(entropy_img1 - entropy_local_desc1))
            CNT = CNT + 1

        loss_mean = loss_mean / CNT if CNT != 0 else pred0['image'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean

if __name__ == '__main__':

    print("每个通道的信息熵:")




