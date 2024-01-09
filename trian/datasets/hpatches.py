import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch.utils.data as data


class HPatchesDataset(data.Dataset):
    def __init__(self, root: str = '../data/hpatches', alteration: str = 'all', gray: bool = False):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root
        self.gray = gray
        # get all image file name
        self.image0_list = []
        self.image1_list = []
        self.homographies = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue
            # count images
            file_ext = '.ppm'
            pattern = folder / ('*' + file_ext)
            img_names = glob.glob(pathname=str(pattern))
            num_images = len(img_names)
            # get image pair file names and homographies
            for i in range(2, 1 + num_images):
                self.image0_list.append(str(Path(folder, '1' + file_ext)))
                self.image1_list.append(str(Path(folder, str(i) + file_ext)))
                self.homographies.append(str(Path(folder, 'H_1_' + str(i))))

        self.len = len(self.image0_list)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        # read image
        img0 = cv2.imread(self.image0_list[item], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.image1_list[item], cv2.IMREAD_COLOR)
        h_r, w_r, _ = img0.shape

        # img0 = cv2.resize(img0, (320, 240))
        # img1 = cv2.resize(img1, (320, 240))
        assert img0 is not None, 'can not load: ' + self.image0_list[item]
        assert img1 is not None, 'can not load: ' + self.image1_list[item]

        if self.gray:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
            img0 = np.expand_dims(img0, axis=2)
            img1 = np.expand_dims(img1, axis=2)
        else:
            # bgr -> rgb
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype('float32') / 255.  # HxWxC
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype('float32') / 255.  # HxWxC

        h0, w0, _ = img0.shape
        h1, w1, _ = img1.shape

        # read homography
        homography = np.loadtxt(self.homographies[item]).astype('float32')

        # pack return dict
        return {'image0': img0.transpose(2, 0, 1),  # [C,H,W]
                'image1': img1.transpose(2, 0, 1),  # [C,H,W]
                'warp01_params': {'mode': 'homo', 'width': w1, 'height': h1,
                                  'homography_matrix': homography, 'k_w': 1, 'k_h': 1},
                'warp10_params': {'mode': 'homo', 'width': w0, 'height': h0,
                                  'homography_matrix': np.linalg.inv(homography), 'k_w': 1, 'k_h': 1}
                }

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__


class HPatchesSquenceDataset(data.Dataset):
    def __init__(self, root: str = '../data/hpatches', alteration: str = 'all',
                 gray: bool = False):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root
        self.gray = gray

        # get all image file name
        self.image_list = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue
            # count images
            file_ext = '.ppm'
            pattern = folder / ('*' + file_ext)
            img_names = glob.glob(pathname=str(pattern))
            self.image_list += img_names

        self.len = len(self.image_list)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        # read image
        image0 = cv2.imread(self.image_list[item], cv2.IMREAD_COLOR)
        assert image0 is not None, 'can not load: ' + self.image_list[item]

        # bgr -> rgb
        image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB).astype('float32') / 255.  # HxWxC

        # pack return dict
        ret = {'image': image.transpose(2, 0, 1),  # [C,H,W]
               'path': self.image_list[item]}

        if self.gray:
            gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY).astype('float32') / 255.  # HxW
            ret['gray'] = gray[np.newaxis, ...]  # [1,H,W]

        return ret

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__


if __name__ == '__main__':
    from tqdm import tqdm

    hpatches_dataset = HPatchesSquenceDataset(root='../data/hpatches', alteration='i')
    max_shapes = []
    for data in tqdm(hpatches_dataset):
        plt.imshow(data['image'].transpose(1, 2, 0))
        plt.show()
