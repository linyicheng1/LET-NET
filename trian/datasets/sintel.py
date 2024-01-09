import numpy as np
import torch
import torch.utils.data as data
import random
from glob import glob
import os.path as osp
import os
from utils import frame_utils


class Sintel(data.Dataset):
    def __init__(self, root: str = '../data/sintel', alteration: str = 'training', dstype: str = 'clean'):
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        flow_root = osp.join(root, alteration, 'flow')
        image_root = osp.join(root, alteration, dstype)

        if alteration == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if alteration != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valid = None
        flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return {'img1': img1,
                'img2': img2,
                'flow': flow,
                'valid': valid.float()}

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    sintel_dataset = Sintel(root='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/sintel',
                          alteration='training')
    sintel_dataloader = DataLoader(sintel_dataset, batch_size=1, pin_memory=False, num_workers=8)
    for data in sintel_dataloader:
        print(data['img1'].shape)
        print(data['img2'].shape)
        print(data['flow'].shape)
        print(data['valid'].shape)

