import os
import copy
import h5py
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
import json
import random
import logging

import cv2
from utils import scale_intrinsics, warp_depth, warp_points2d


class MegaDepthDataset(Dataset):
    def __init__(
            self,
            root='../data/megadepth',
            train=True,
            using_cache=False,
            pairs_per_scene=100,
            image_size=256,
            colorjit=True,
            gray=False,
            crop_or_scale='scale',  # crop, scale, crop_scale
    ):
        self.data_path = Path(root)
        self.train = train
        self.using_cache = using_cache
        # print('self.using_cache', self.using_cache)
        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size
        self.gray = gray
        self.crop_or_scale = crop_or_scale

        self.dataset = []

        if train and colorjit:
            self.transforms = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                                  transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.ToTensor()])

        self.build_dataset(init=False)

    def build_dataset(self, init=False):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(0)

        logging.info('Building dataset...')

        json_path = self.data_path / 'dataset.json'
        cache_file = Path(__file__).parent / (self.data_path.stem + '_cache.pkl')

        if cache_file.exists() and self.using_cache:
            logging.info(f'Loading cached meta data: {cache_file}')
            self.dataset = torch.load(cache_file)
        else:

            with open(json_path, 'r') as load_f:
                datasets = json.load(load_f)

            if init:
                self.dataset = [0] * (self.pairs_per_scene * len(datasets.items()))
                return

            pro_bar = tqdm(datasets.items())
            for scene_name, scene_data in pro_bar:
                pro_bar.set_description('scene' + scene_name)

                image_path = Path(scene_data['image_path'])
                depth_path = Path(scene_data['depth_path'])
                calib_path = Path(scene_data['calib_path'])

                if self.train:
                    scene_indices = random.sample(scene_data['tuples'], self.pairs_per_scene)
                else:
                    scene_indices = scene_data['tuples']

                def getpath(image_name):
                    if image_name.endswith('jpg'):
                        base_name = image_name.split('.')[0]
                    else:
                        base_name = copy.deepcopy(image_name)
                        image_name = base_name + '.jpg'

                    imagepath = image_path / image_name
                    depthpath = depth_path / (base_name + '.h5')
                    if self.train:
                        calibpath = self.data_path / calib_path / ('calibration_' + image_name + '.h5')
                    else:
                        calibpath = self.data_path / calib_path / ('calibration_' + base_name + '.h5')

                    return imagepath, depthpath, calibpath

                def readcalib(calibpath):
                    pose = np.identity(4)
                    with h5py.File(calibpath, 'r') as f:
                        intrinsic = f['K'][()]
                        pose[:3, :3] = f['R'][()]
                        pose[:3, 3] = f['T'][()]

                    return intrinsic.astype(np.float32), pose.astype(np.float32)

                for indices in scene_indices:
                    if self.train:
                        i, j = random.sample(indices, 2)
                    else:
                        i, j = indices[0], indices[1]

                    imagepathi, depthpathi, calibpathi = getpath(scene_data['images'][i])
                    imagepathj, depthpathj, calibpathj = getpath(scene_data['images'][j])

                    intrinsicsi, posei = readcalib(calibpathi)
                    intrinsicsj, posej = readcalib(calibpathj)

                    self.dataset.append({'image_path1': imagepathi,
                                         'depth_path1': depthpathi,
                                         'intrinsics1': intrinsicsi,
                                         'pose1': posei,
                                         'image_path2': imagepathj,
                                         'depth_path2': depthpathj,
                                         'intrinsics2': intrinsicsj,
                                         'pose2': posej,
                                         })
            if self.using_cache:
                torch.save(self.dataset, cache_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        #print(pair_metadata['depth_path1'])
        depth_path1 = self.data_path / pair_metadata['depth_path1']

        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert (np.min(depth1) >= 0)
        image_path1 = self.data_path / pair_metadata['image_path1']
        image1 = Image.open(image_path1)
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        image1 = np.array(image1)
        assert (image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = self.data_path / pair_metadata['depth_path2']
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert (np.min(depth2) >= 0)
        image_path2 = self.data_path / pair_metadata['image_path2']
        image2 = Image.open(image_path2)
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        image2 = np.array(image2)
        assert (image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        pose12 = pose2 @ np.linalg.inv(pose1)
        pose21 = np.linalg.inv(pose12)

        if self.train:
            if "crop" in self.crop_or_scale:
                # ================================================= compute central_match
                DOWNSAMPLE = 10
                # resize to speed up
                depth1s = cv2.resize(depth1, (depth1.shape[1] // DOWNSAMPLE, depth1.shape[0] // DOWNSAMPLE))
                depth2s = cv2.resize(depth2, (depth2.shape[1] // DOWNSAMPLE, depth2.shape[0] // DOWNSAMPLE))
                intrinsic1s = scale_intrinsics(intrinsics1, (DOWNSAMPLE, DOWNSAMPLE))
                intrinsic2s = scale_intrinsics(intrinsics2, (DOWNSAMPLE, DOWNSAMPLE))

                # warp
                depth12s = warp_depth(depth1s, intrinsic1s, intrinsic2s, pose12, depth2s.shape)
                depth21s = warp_depth(depth2s, intrinsic2s, intrinsic1s, pose21, depth1s.shape)

                depth12s[depth12s < 0] = 0
                depth21s[depth21s < 0] = 0

                valid12s = np.logical_and(depth12s > 0, depth2s > 0)
                valid21s = np.logical_and(depth21s > 0, depth1s > 0)

                pos1 = np.array(valid21s.nonzero())
                try:
                    idx1_random = np.random.choice(np.arange(pos1.shape[1]), 1)
                    uv1s = pos1[:, idx1_random][[1, 0]].reshape(1, 2)
                    d1s = np.array(depth1s[uv1s[0, 1], uv1s[0, 0]]).reshape(1, 1)

                    uv12s, z12s = warp_points2d(uv1s, d1s, intrinsic1s, intrinsic2s, pose12)

                    uv1 = uv1s[0] * DOWNSAMPLE
                    uv2 = uv12s[0] * DOWNSAMPLE
                except ValueError:
                    uv1 = [depth1.shape[1] / 2, depth1.shape[0] / 2]
                    uv2 = [depth2.shape[1] / 2, depth2.shape[0] / 2]

                central_match = [uv1[1], uv1[0], uv2[1], uv2[0]]
                # ================================================= compute central_match

            if self.crop_or_scale == 'crop':
                # =============== padding
                h1, w1, _ = image1.shape
                h2, w2, _ = image2.shape
                if h1 < self.image_size:
                    padding = np.zeros((self.image_size - h1, w1, 3))
                    image1 = np.concatenate([image1, padding], axis=0).astype(np.uint8)
                    depth1 = np.concatenate([depth1, padding[:, :, 0]], axis=0).astype(np.float32)
                    h1, w1, _ = image1.shape
                if w1 < self.image_size:
                    padding = np.zeros((h1, self.image_size - w1, 3))
                    image1 = np.concatenate([image1, padding], axis=1).astype(np.uint8)
                    depth1 = np.concatenate([depth1, padding[:, :, 0]], axis=1).astype(np.float32)
                if h2 < self.image_size:
                    padding = np.zeros((self.image_size - h2, w2, 3))
                    image2 = np.concatenate([image2, padding], axis=0).astype(np.uint8)
                    depth2 = np.concatenate([depth2, padding[:, :, 0]], axis=0).astype(np.float32)
                    h2, w2, _ = image2.shape
                if w2 < self.image_size:
                    padding = np.zeros((h2, self.image_size - w2, 3))
                    image2 = np.concatenate([image2, padding], axis=1).astype(np.uint8)
                    depth2 = np.concatenate([depth2, padding[:, :, 0]], axis=1).astype(np.float32)
                # =============== padding
                image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

                depth1 = depth1[bbox1[0]: bbox1[0] + self.image_size, bbox1[1]: bbox1[1] + self.image_size]
                depth2 = depth2[bbox2[0]: bbox2[0] + self.image_size, bbox2[1]: bbox2[1] + self.image_size]
            elif self.crop_or_scale == 'scale':
                image1, depth1, intrinsics1 = self.scale(image1, depth1, intrinsics1)
                image2, depth2, intrinsics2 = self.scale(image2, depth2, intrinsics2)
                bbox1 = bbox2 = np.array([0., 0.])
            elif self.crop_or_scale == 'crop_scale':
                bbox1 = bbox2 = np.array([0., 0.])
                image1, depth1, intrinsics1 = self.crop_scale(image1, depth1, intrinsics1, central_match[:2])
                image2, depth2, intrinsics2 = self.crop_scale(image2, depth2, intrinsics2, central_match[2:])
            else:
                raise RuntimeError(f"Unkown type {self.crop_or_scale}")
        else:
            bbox1 = bbox2 = np.array([0., 0.])

        return (image1, depth1, intrinsics1, pose12, bbox1,
                image2, depth2, intrinsics2, pose21, bbox2)

    def scale(self, image, depth, intrinsic):
        img_size_org = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(intrinsic, (img_size_org[1] / self.image_size, img_size_org[0] / self.image_size))
        return image, depth, intrinsic

    def crop_scale(self, image, depth, intrinsic, centeral):
        h_org, w_org, three = image.shape
        image_size = min(h_org, w_org)
        if h_org > w_org:
            if centeral[1] - image_size // 2 < 0:
                h_start = 0
            elif centeral[1] + image_size // 2 > h_org:
                h_start = h_org - image_size
            else:
                h_start = int(centeral[1]) - image_size // 2
            w_start = 0
        else:
            if centeral[0] - image_size // 2 < 0:
                w_start = 0
            elif centeral[0] + image_size // 2 > w_org:
                w_start = w_org - image_size
            else:
                w_start = int(centeral[0]) - image_size // 2
            h_start = 0

        croped_image = image[h_start: h_start + image_size, w_start: w_start + image_size]
        croped_depth = depth[h_start: h_start + image_size, w_start: w_start + image_size]
        intrinsic[0, 2] = intrinsic[0, 2] - w_start
        intrinsic[1, 2] = intrinsic[1, 2] - h_start

        image = cv2.resize(croped_image, (self.image_size, self.image_size))
        depth = cv2.resize(croped_depth, (self.image_size, self.image_size))
        intrinsic = scale_intrinsics(intrinsic, (image_size / self.image_size, image_size / self.image_size))

        return image, depth, intrinsic

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (image1[bbox1_i: bbox1_i + self.image_size, bbox1_j: bbox1_j + self.image_size],
                np.array([bbox1_i, bbox1_j]),
                image2[bbox2_i: bbox2_i + self.image_size, bbox2_j: bbox2_j + self.image_size],
                np.array([bbox2_i, bbox2_j])
                )

    def __getitem__(self, idx):
        # print('***********************************')
        # print(idx)
        # print(self.dataset[idx])
        # print('***********************************')
        (image1, depth1, intrinsics1, pose12, bbox1,
         image2, depth2, intrinsics2, pose21, bbox2) \
            = self.recover_pair(self.dataset[idx])

        if self.gray:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            gray1 = transforms.ToTensor()(gray1)
            gray2 = transforms.ToTensor()(gray2)
            image1 = gray1
            image2 = gray2

        if self.transforms is not None:
            image1, image2 = self.transforms(image1), self.transforms(image2)  # [C,H,W]

        ret = {'image0': image1,
               'image1': image2,
               'angle': self.dataset[idx]['angle'] if 'angle' in self.dataset[idx] else 0,
               'overlap': self.dataset[idx]['overlap_ratio'] if 'overlap_ratio' in self.dataset[idx] else 0,
               'warp01_params': {'mode': 'se3',
                                 'width': self.image_size if self.train else image1.shape[2],
                                 'height': self.image_size if self.train else image1.shape[1],
                                 'pose01': torch.from_numpy(pose12.astype(np.float32)),
                                 'bbox0': torch.from_numpy(bbox1.astype(np.float32)),
                                 'bbox1': torch.from_numpy(bbox2.astype(np.float32)),
                                 'depth0': torch.from_numpy(depth1.astype(np.float32)),
                                 'depth1': torch.from_numpy(depth2.astype(np.float32)),
                                 'intrinsics0': torch.from_numpy(intrinsics1.astype(np.float32)),
                                 'intrinsics1': torch.from_numpy(intrinsics2.astype(np.float32))},
               'warp10_params': {'mode': 'se3',
                                 'width': self.image_size if self.train else image2.shape[2],
                                 'height': self.image_size if self.train else image2.shape[2],
                                 'pose01': torch.from_numpy(pose21.astype(np.float32)),
                                 'bbox0': torch.from_numpy(bbox2.astype(np.float32)),
                                 'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
                                 'depth0': torch.from_numpy(depth2.astype(np.float32)),
                                 'depth1': torch.from_numpy(depth1.astype(np.float32)),
                                 'intrinsics0': torch.from_numpy(intrinsics2.astype(np.float32)),
                                 'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32))},
               }
        if self.gray:
            ret['gray0'] = gray1
            ret['gray1'] = gray2
        return ret


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt


    def visualize(image0, image1, depth0, depth1):
        # visualize image and depth
        plt.figure(figsize=(9, 9))
        plt.subplot(2, 2, 1)
        plt.imshow(image0, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(depth0)
        plt.subplot(2, 2, 3)
        plt.imshow(image1, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(depth1)
        plt.show()


    dataset = MegaDepthDataset(  # root='../data/megadepth',
        root='/home/stonehpc/Dataset/disk-data/imw2020-val',
        train=False,
        using_cache=True,
        pairs_per_scene=100,
        image_size=256,
        colorjit=True,
        gray=False,
        crop_or_scale='scale',
    )
    dataset.build_dataset()

    batch_size = 2

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, batch in enumerate(tqdm(loader)):
        image0, image1 = batch['image0'], batch['image1']  # [B,3,H,W]
        depth0, depth1 = batch['warp01_params']['depth0'], batch['warp01_params']['depth1']  # [B,H,W]
        intrinsics0, intrinsics1 = batch['warp01_params']['intrinsics0'], batch['warp01_params'][
            'intrinsics1']  # [B,3,3]

        batch_size, channels, h, w = image0.shape

        # for b_idx in range(batch_size):
        #     visualize(image0[b_idx].permute(1, 2, 0), image1[b_idx].permute(1, 2, 0), depth0[b_idx], depth1[b_idx])
