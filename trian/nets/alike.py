import os

import cv2
import math
import torch
import logging
from copy import deepcopy
from torchvision.transforms import ToTensor

from nets.alnet import ALNet
from nets.soft_detect import SoftDetect
import time


class ALIKE(ALNet):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 agg_mode: str = 'cat',  # sum, cat, fpn
                 single_head: bool = False,
                 pe: bool = False,
                 grayscale: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5,
                 n_limit: int = 0,
                 **kwargs
                 ):
        super().__init__(c1, c2, c3, c4, dim, agg_mode, single_head, pe, grayscale)

        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th

        self.update_softdetect_parameters()

    def update_softdetect_parameters(self):
        self.softdetect = SoftDetect(radius=self.radius, top_k=self.top_k,
                                     scores_th=self.scores_th, n_limit=self.n_limit)

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
        w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map, local_descriptor = super().forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
            local_descriptor = local_descriptor[:, :, :h, :w]
        # ====================================================

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        if ret_dict:
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, 'local_descriptor': local_descriptor}
        else:
            return descriptor_map, scores_map, local_descriptor

    def extract(self, image):
        descriptor_map, scores_map, local_descriptor = self.extract_dense_map(image)
        keypoints, descriptors, kptscores, scoredispersitys = self.softdetect(scores_map, descriptor_map)

        return {'keypoints': keypoints,  # B M 2
                'descriptors': descriptors,  # B M D
                'scores': kptscores,  # B M D
                'score_dispersity': scoredispersitys,
                'descriptor_map': descriptor_map,  # BxCxHxW
                'scores_map': scores_map,  # Bx1xHxW
                'local_descriptor': local_descriptor,  # BxCxHxW
                }

    def forward(self, img, scale_f=2 ** 0.5,
                min_scale=1., max_scale=1.,
                min_size=0., max_size=99999.,
                image_size_max=99999,
                verbose=False, n_k=0, sort=False,
                scoremap=True,
                descmap=True):
        """
        :param img: np array, HxWx3
        :param scale_f:
        :param min_scale:
        :param max_scale:
        :param min_size:
        :param max_size:
        :param verbose:
        :param n_k:
        :param sort:
        :return: keypoints, descriptors, scores
        """
        old_bm = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # speedup

        H_, W_, three = img.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== image size constraint
        image = deepcopy(img)
        max_hw = max(H_, W_)
        if max_hw > image_size_max:
            ratio = float(image_size_max / max_hw)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # ==================== convert image to tensor
        H, W, three = image.shape
        image = ToTensor()(image).unsqueeze(0)
        image = image.to(self.device)

        # ==================== extract keypoints at multiple scales
        start = time.time()

        s = 1.0  # current scale factor
        if verbose:
            logging.info('')
        keypoints, descriptors, scores, scores_maps, descriptor_maps = [], [], [], [], []
        while s + 0.001 >= max(min_scale, min_size / max(H, W)):
            if s - 0.001 <= min(max_scale, max_size / max(H, W)):
                nh, nw = image.shape[2:]

                # extract descriptors
                with torch.no_grad():
                    descriptor_map, scores_map, local_descriptor = self.extract_dense_map(image)
                    keypoints_, descriptors_, scores_, _ = self.softdetect(scores_map, descriptor_map)

                if scoremap:
                    scores_maps.append(scores_map[0, 0].cpu())
                if descmap:
                    descriptor_maps.append(descriptor_map[0].cpu())
                keypoints.append(keypoints_[0])
                descriptors.append(descriptors_[0])
                scores.append(scores_[0])

                if verbose:
                    logging.info(
                        f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}. Number of keypoints {len(keypoints)}.")

            s /= scale_f

            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            image = torch.nn.functional.interpolate(image, (nh, nw), mode='bilinear', align_corners=False)

        # restore value
        torch.backends.cudnn.benchmark = old_bm

        keypoints = torch.cat(keypoints)
        descriptors = torch.cat(descriptors)
        scores = torch.cat(scores)
        keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W_ - 1, H_ - 1]])

        if sort or 0 < n_k < len(keypoints):
            indices = torch.argsort(scores, descending=True)
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]
            scores = scores[indices]

        if 0 < n_k < len(keypoints):
            keypoints = keypoints[0:n_k]
            descriptors = descriptors[0:n_k]
            scores = scores[0:n_k]

        return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores,
                'descriptor_maps': descriptor_maps,
                'scores_maps': scores_maps, 'time': time.time() - start, }

def flow_hsv(flow):
    flow = np.clip(flow, -512, 512)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def calc_epe(flow, gt_flow, valid):
    epe = torch.sum((flow - gt_flow) ** 2, dim=0).sqrt()
    mag = torch.sum(gt_flow ** 2, dim=0).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()


    return epe, out, val

def dense_flow(alteration = 'training', datasets = 'kitti'):
    if datasets == 'kitti':
        dataset = KITTI(root='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti', alteration=alteration)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory= False, num_workers=8)
    elif datasets == 'sintel':
        dataset = Sintel(root='/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/sintel', alteration=alteration)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=False, num_workers=8)

    epe_list = []
    out_list = []
    epe_list_o = []
    out_list_o = []
    idx = 0
    for data in (dataloader):
        img1 = data['img1']
        img2 = data['img2']
        flow = data['flow']
        valid = data['valid']
        # to cuda
        img1 = img1.cuda()
        img2 = img2.cuda()

        with torch.no_grad():
            desc1, score1, local_desc1 = net.extract_dense_map(img1)
            desc2, score2, local_desc2 = net.extract_dense_map(img2)

        local_desc1 = (local_desc1.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        local_desc2 = (local_desc2.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        flow_img = flow_hsv(flow)


        prvs = cv2.cvtColor(local_desc1, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(local_desc2, cv2.COLOR_BGR2GRAY)
        prvs_o = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        next_o = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('local_desc1', local_desc1)
        # cv2.imshow('local_desc2', local_desc2)
        # cv2.imshow('prvs_o', img1)
        # cv2.imshow('next_o', img2)

        inst = cv2.optflow.DenseRLOFOpticalFlow_create()
        # inst = cv2.optflow.createOptFlow_PCAFlow()
        fflow = inst.calc(local_desc1, local_desc2, None)
        ffflow = inst.calc(img1, img2, None)
        # fflow = inst.calc(prvs, next, None)
        # ffflow = inst.calc(prvs_o, next_o, None)
        # fflow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # ffflow = cv2.calcOpticalFlowFarneback(prvs_o, next_o, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fflow_img = flow_hsv(fflow)
        ffflow_img = flow_hsv(ffflow)
        delta_img = flow_hsv(fflow - flow)
        delta_o_img = flow_hsv(ffflow - flow)

        flow = torch.from_numpy(flow).permute(2, 0, 1)
        fflow = torch.from_numpy(fflow).permute(2, 0, 1)
        ffflow = torch.from_numpy(ffflow).permute(2, 0, 1)

        # cv2.imshow('flow_img', flow_img)
        # cv2.imshow('fflow_img', fflow_img)
        # cv2.imshow('ffflow_img', ffflow_img)
        # cv2.imshow('delta_img', delta_img)
        # cv2.imshow('delta_o_img', delta_o_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        epe, out, val = calc_epe(fflow, flow, valid)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        print(idx)
        idx = idx + 1

        epe_o, out_o, val_o = calc_epe(ffflow, flow, valid)
        epe_list_o.append(epe_o[val_o].mean().item())
        out_list_o.append(out_o[val_o].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    epe_list_o = np.array(epe_list_o)
    out_list_o = np.concatenate(out_list_o)
    epe_o = np.mean(epe_list_o)
    f1_o = 100 * np.mean(out_list_o)

    print("Validation %s: %f, %f" % (datasets, epe, f1))
    # print('%s-epe: ' % (datasets, epe))
    # print('%s-f1: ' % (datasets, f1))

    print("Validation %s: %f, %f" % (datasets, epe_o, f1_o))
    # print('%s-epe: ' % (datasets, epe))
    # print('%s-f1: ' % (datasets, f1))

def calc_repeatability(alteration = 'i', top_k: int = 300, scores_th: float = 0,
                 n_limit: int = 0, scores_th_eval: float = 0.2, n_limit_eval: int = 20000,
                 eval_gt_th: int = 3):
    hpatch_dir = '/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/WangShuo/datasets/HPatch'
    hpatch_i_dataset = HPatchesDataset(root=hpatch_dir, alteration='i')
    hpatch_v_dataset = HPatchesDataset(root=hpatch_dir, alteration='v')
    hpatch_i_dataloader = DataLoader(hpatch_i_dataset, batch_size=1, pin_memory=False, num_workers=1)
    hpatch_v_dataloader = DataLoader(hpatch_v_dataset, batch_size=1, pin_memory=False, num_workers=1)

    num = []
    repeatability = []
    repeatability_ = []
    accuracy = []
    matching_score = []
    dataloader = []
    dataset = []
    if alteration == 'i':
        dataloader.append(hpatch_i_dataloader)
        dataset.append(hpatch_i_dataset)
    elif alteration == 'v':
        dataloader.append(hpatch_v_dataloader)
        dataset.append(hpatch_v_dataset)
    elif alteration == 'all':
        dataloader.append(hpatch_i_dataloader)
        dataloader.append(hpatch_v_dataloader)
        dataset.append(hpatch_i_dataset)
        dataset.append(hpatch_v_dataset)
    else:
        raise ValueError('alteration should be i or v or all')

    for dataloader_ in dataloader:
        for data in dataloader_:
            # print(data)
            img0 = data['image0']
            img1 = data['image1']
            # to cuda
            # img0 = img0.cuda()
            # img1 = img1.cuda()
            warp01_params, warp10_params = {}, {}
            for k, v in data['warp01_params'].items():
                warp01_params[k] = v[0]
            for k, v in data['warp10_params'].items():
                warp10_params[k] = v[0]
            b, _, h0, w0 = img0.shape
            _, _, h1, w1 = img1.shape
            assert b == 1

            # ==================================== extract keypoints and descriptors
            net.top_k, net.scores_th, net.n_limit = top_k, scores_th_eval, n_limit_eval

            net.update_softdetect_parameters()
            # s = time.time()
            pred0 = net.extract(img0)
            # e = time.time()
            # execution_time = e - s
            # print(f"Execution Time: {execution_time:.6f} seconds")
            pred1 = net.extract(img1)

            # net.top_k, net.scores_th, net.n_limit = top_k_old, scores_th_old, n_limit_old
            # net.update_softdetect_parameters()
            k_w = warp01_params['k_w']
            k_h = warp01_params['k_h']

            kpts0 = keypoints_normal2pixel(pred0['keypoints'], w0 / k_w, h0 / k_h)[0]
            kpts1 = keypoints_normal2pixel(pred1['keypoints'], w1 / k_w, h1 / k_h)[0]



            # num_feat = min(kpts0.shape[0], kpts1.shape[0])  # number of detected keypoints

            # ==================================== covisible keypoints
            kpts0_cov, kpts01_cov, _, _ = warp(kpts0.cpu(), warp01_params)
            kpts1_cov, kpts10_cov, _, _ = warp(kpts1.cpu(), warp10_params)
            # if(kpts01_cov.shape[0] > 300):
            #     kpts0_cov = kpts0_cov[:300]
            #     kpts01_cov = kpts01_cov[:300]
            # if(kpts10_cov.shape[0] > 300):
            #     kpts1_cov = kpts1_cov[:300]
            #     kpts10_cov = kpts10_cov[:300]
            if(kpts01_cov.shape[0] == 0 or kpts10_cov.shape[0] == 0):
                continue

            num_cov_feat = (len(kpts0_cov) + len(kpts1_cov)) / 2  # number of covisible keypoints

            # ==================================== new repeatability
            dist0 = compute_keypoints_distance(kpts0.cpu(), kpts10_cov)
            dist1 = compute_keypoints_distance(kpts1.cpu(), kpts01_cov)
            total = len(kpts01_cov) + len(kpts10_cov)
            min1, _ = torch.min(dist0, dim=1)
            min2, _ = torch.min(dist1, dim=1)
            # print(min1.shape, min2.shape)
            # num1 = 0
            # num2 = 0
            # mask1 = min1 <= eval_gt_th
            # mask2 = min2 <= eval_gt_th
            num1 = (min1 <= eval_gt_th / min(k_w, k_h)).sum().cpu()
            num2 = (min2 <= eval_gt_th / min(k_w, k_h)).sum().cpu()

            # # ==================================== putative matches
            # matches_est = mutual_argmax(desc0 @ desc1.t())
            # mkpts0, mkpts1 = kpts0[matches_est[0]], kpts1[matches_est[1]]
            #
            # num_putative = len(mkpts0)  # number of putative matches
            #
            # # ==================================== warp putative matches
            # mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            # mkpts1 = mkpts1[ids0]
            #
            # dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1)).cpu()
            # if dist.shape[0] == 0:
            #     dist = dist.new_tensor([float('inf')])
            #
            # num_inlier = sum(dist <= eval_gt_th)

            num.append(total)
            # print(total)
            repeatability.append((num1 + num2) / total)
            # accuracy.append(num_inlier / max(num_putative, 1))
            # matching_score.append(num_inlier / max(num_cov_feat, 1))
            print(num1, num2, total)
            print((num1 + num2) / total)
            # if (num1 + num2) / total < 0.2:
            #     img0_show = img0.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            #     plt.imshow(img0_show)
            #     plt.savefig('rep/img0.png')
            #     img1_show = img1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            #     plt.imshow(img1_show)
            #     plt.savefig('rep/img1.png')
            #     # draw keypoints
            #     plt.imshow(img0_show)
            #     for i in range(kpts0.shape[0]):
            #         plt.scatter(kpts0[i, 0].cpu().detach().numpy() * k_w.numpy(), kpts0[i, 1].cpu().detach().numpy() * k_h.numpy(), s=3, c='r')
            #     for i in range(kpts10_cov.shape[0]):
            #         plt.scatter(kpts10_cov[i, 0].cpu().detach().numpy() * k_w.numpy(), kpts10_cov[i, 1].cpu().detach().numpy() * k_h.numpy(), s=1, c='b')
            #
            #     plt.savefig('rep/kpts0.png')
            #
            #     plt.imshow(img1_show)
            #     for i in range(kpts1.shape[0]):
            #         plt.scatter(kpts1[i, 0].cpu().detach().numpy() * k_w.numpy(), kpts1[i, 1].cpu().detach().numpy() * k_h.numpy(), s=3, c='r')
            #     for i in range(kpts01_cov.shape[0]):
            #         plt.scatter(kpts01_cov[i, 0].cpu().detach().numpy() * k_w.numpy(), kpts01_cov[i, 1].cpu().detach().numpy() * k_h.numpy(), s=1, c='b')
            #     plt.savefig('rep/kpts1.png')
            #     input('pause')
        repeatability_.append(sum(repeatability) / len(repeatability))


    print('repeatability_: ', repeatability_)
    # print('repeatability: {:.4f}'.format(sum(repeatability) / len(repeatability)))
    # print('accuracy: {:.4f}'.format(sum(accuracy) / len(accuracy)))
    # print('matching score: {:.4f}'.format(sum(matching_score) / len(matching_score)))


if __name__ == '__main__':
    import numpy as np
    from datasets.kitti import KITTI
    from datasets.sintel import Sintel
    from datasets.hpatches import HPatchesDataset
    from torch.utils.data import DataLoader
    import cv2
    import io
    from utils import keypoints_normal2pixel, warp, compute_keypoints_distance

    net = ALIKE(c1=8, c2=16, c3=32, c4=64, dim=64, agg_mode='cat', res_block=True, single_head=True)

    # net.load_from_checkpoint('/home/stonehpc/linyicheng/py_project/ALIKE_code/training/log_train/train/Version-0708-221706/checkpoints/epoch=79-mean_metric=0.4917.ckpt')
    with open('/home/server/wangshuo/ALIKE_code/training/model_old.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    net.load_state_dict(torch.load(buffer))
    # net.eval()
    # net.cuda()
    # dense_flow(datasets='sintel')
    # calc_repeatability(alteration='all')


    img1 = cv2.imread('/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/lk_flow/1/1.jpg')
    img2 = cv2.imread('/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/lk_flow/1/2.jpg')
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    h, w, c = img1.shape
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    net.top_k, net.scores_th, net.n_limit = 300, 0.5, 2000
    net.update_softdetect_parameters()
    with torch.no_grad():
        pred0 = net.extract(img1_tensor)
        pred1 = net.extract(img2_tensor)

    local_desc1 = pred0['local_descriptor']
    local_desc2 = pred1['local_descriptor']
    local_desc1_numpy = (local_desc1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    local_desc2_numpy = (local_desc2.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    score1 = pred0['scores_map']
    score2 = pred1['scores_map']
    score1_numpy = (score1.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    score2_numpy = (score2.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    kpts0 = keypoints_normal2pixel(pred0['keypoints'], w, h)[0].numpy()
    # kpts0 = cv2.goodFeaturesToTrack(cv2.cvtColor(local_desc1_numpy, cv2.COLOR_BGR2GRAY), 300, 0.01, 3)


    lk_params = dict(winSize=(15, 15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kpts1, status, err = cv2.calcOpticalFlowPyrLK(local_desc1_numpy, local_desc2_numpy, kpts0, None, **lk_params)
    print(kpts0.shape, kpts1.shape)



    for i, (old, new) in enumerate(zip(kpts0, kpts1)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(img2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        cv2.circle(img2, (int(a), int(b)), 1, (0, 0, 255), -1)



    cv2.imshow('local_desc1', local_desc1_numpy)
    cv2.imshow('local_desc2', local_desc2_numpy)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    cv2.imshow('score1', score1_numpy)
    cv2.imshow('score2', score2_numpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
















    # img1 = cv2.imread('/home/server/wangshuo/RAFT/demo-frames/frame_0016.png')
    # img2 = cv2.imread('/home/server/wangshuo/RAFT/demo-frames/frame_0018.png')
    # img1_tensor = torch.from_numpy(img1)
    # img2_tensor = torch.from_numpy(img2)
    # with torch.no_grad():
    #     desc1, score1, local_desc1 = net.extract_dense_map(img1_tensor.permute(2, 0, 1).unsqueeze(0))
    #     desc2, score2, local_desc2 = net.extract_dense_map(img2_tensor.permute(2, 0, 1).unsqueeze(0))
    #
    # local_desc1 = (local_desc1.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # local_desc2 = (local_desc2.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #
    # prvs = cv2.cvtColor(local_desc1, cv2.COLOR_BGR2GRAY)
    # next = cv2.cvtColor(local_desc2, cv2.COLOR_BGR2GRAY)
    # prvs_o = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # next_o = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    # inst = cv2.optflow.createOptFlow_SimpleFlow()
    # flow = inst.calc(prvs, next, None)
    # # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 5, 5, 1.1, 0)
    # flow_img = flow_hsv(flow)
    # fflow = inst.calc(prvs_o, next_o, None)
    # # fflow = cv2.calcOpticalFlowFarneback(prvs_o, next_o, None, 0.5, 3, 15, 5, 5, 1.1, 0)
    # fflow_img = flow_hsv(fflow)
    # # cv2.imshow('img1', img1)
    # # cv2.imshow('img2', img2)
    # cv2.imshow('flow', flow_img)
    # cv2.imshow('fflow', fflow_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()













