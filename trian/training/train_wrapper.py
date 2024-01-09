import os
import logging
import functools
from typing import Optional

import cv2
import numpy as np
from pathlib import Path

import torch

from nets.alike import ALIKE
from nets.losses import *

from training.scheduler import WarmupConstantSchedule
from utils import keypoints_normal2pixel, mutual_argmin, \
    mutual_argmax, plot_keypoints, plot_matches, EmptyTensorError, \
    warp, compute_keypoints_distance
from training.val_hpatches_utils import load_precompute_errors, draw_MMA


class TrainWrapper(ALIKE):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128,
                 c4: int = 128, dim: int = 128,
                 agg_mode: str = 'cat',  # sum, cat, fpn
                 single_head: bool = False,
                 pe: bool = False,
                 grayscale: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5, n_limit: int = 0,  # in training stage
                 scores_th_eval: float = 0.2, n_limit_eval: int = 5000,  # in evaluation stage
                 # ================================== gt reprojection th
                 train_gt_th: int = 5, eval_gt_th: int = 3,
                 # ================================== loss weight
                 w_pk: float = 1.0,  # weight of peaky loss
                 w_rp: float = 1.0,  # weight of reprojection loss
                 w_sp: float = 1.0,  # weight of score map repetability loss
                 w_ds: float = 1.0,  # weight of descritptor loss
                 w_triplet: float = 0.,
                 w_l_ds: float = 0.5,  # weight of local descriptor loss
                 w_l_pk: float = 0.,  # weight of local peaky loss
                 w_i_c: float = 0.0,  # weight of image consistency loss
                 w_i_e: float = 0.0,  # weight of image entropy loss
                 sc_th: float = 0.1,  # score threshold in peaky and  reprojection loss
                 norm: int = 1,  # distance norm
                 temp_sp: float = 0.1,  # temperature in ScoreMapRepLoss
                 temp_ds: float = 0.02,  # temperature in DescReprojectionLoss
                 # ================================== learning rate
                 lr: float = 1e-3,
                 log_freq_img: int = 2000,  # log image every log_freq_img steps
                 # ================================== pretrained_model
                 pretrained_model: Optional[str] = None,
                 lr_scheduler=functools.partial(WarmupConstantSchedule, warmup_steps=10000),
                 debug: bool = False,
                 ):
        super().__init__(c1, c2, c3, c4, dim, agg_mode, single_head, pe, grayscale, radius, top_k, scores_th, n_limit, )
        self.save_hyperparameters()

        self.lr = lr

        # =================== hyper parameters
        # soft detetor parameters
        self.radius = radius
        # loss configs
        self.w_pk = w_pk
        self.w_rp = w_rp
        self.w_sp = w_sp
        self.w_ds = w_ds
        self.w_triplet = w_triplet
        self.w_l_ds = w_l_ds
        self.w_l_pk = w_l_pk
        self.w_i_c = w_i_c
        self.w_i_e = w_i_e
        # reprojection loss parameters
        self.train_gt_th = train_gt_th
        # evaluation th for MMA on training dataset
        self.eval_gt_th = eval_gt_th

        self.scores_th_eval = scores_th_eval
        self.n_limit_eval = n_limit_eval

        self.log_freq_img = log_freq_img

        self.pretrained_model = pretrained_model
        self.lr_scheduler = lr_scheduler
        self.debug = debug

        # =================== model weight
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                logging.info(f"Loading {pretrained_model}")
                if pretrained_model.endswith('ckpt'):
                    state_dict = torch.load(pretrained_model, torch.device('cpu'))['state_dict']
                elif pretrained_model.endswith('pt'):
                    state_dict = torch.load(pretrained_model, torch.device('cpu'))
                else:
                    logging.error(f"Error model file: {pretrained_model}")
                self.load_state_dict(state_dict, strict=False)
            else:
                logging.error(f"File dose not exists: {pretrained_model}")

        # =================== losses
        if self.w_pk > 0:
            self.PeakyLoss = PeakyLoss(scores_th=sc_th)
        if self.w_rp > 0:
            self.ReprojectionLocLoss = ReprojectionLocLoss(norm=norm, scores_th=sc_th)
        if self.w_sp > 0:
            self.ScoreMapRepLoss = ScoreMapRepLoss(temperature=temp_sp)
        if self.w_ds > 0:
            self.DescReprojectionLoss = DescReprojectionLoss(temperature=temp_ds)
        if self.w_triplet > 0:
            self.TripletLoss = TripletLoss()
        if self.w_l_ds > 0:
            self.LocalDescLoss = LocalDescLoss()
        if self.w_l_pk > 0:
            self.LocalPeakyLoss = LocalPeakyLoss()
        if self.w_i_c > 0:
            self.IlluConsLoss = IlluConsLoss()
        if self.w_i_e > 0:
            self.InfoEntropyLoss = InfoEntropyLoss()

        # ================== to compute MMA on hpatches
        lim = [1, 15]
        self.rng = np.arange(lim[0], lim[1] + 1)
        self.i_err = {thr: 0 for thr in self.rng}
        self.v_err = {thr: 0 for thr in self.rng}
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []
        self.errors = load_precompute_errors(str(Path(__file__).parent / 'errors.pkl'))

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"train_acc": 0,
                                                   'val_kpt_num': 0,
                                                   'val_repeatability': 0,
                                                   'val_acc': 0,
                                                   'val_matching_score': 0,
                                                   'val_kpt_num_mean': 0,
                                                   'val_repeatability_mean': 0,
                                                   'val_acc/mean': 0,
                                                   'val_matching_score/mean': 0,
                                                   'val_metrics/mean': 0,
                                                   "val_mma_mean": 0, })

    '''

    '''


    def compute_correspondence(self, pred0, pred1, batch, rand=True):
        # image size
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['scores_map'][0].new_tensor([[w - 1, h - 1]])

        if self.debug:
            from utils import display_image_in_actual_size
            image0 = batch['image0'][0].permute(1, 2, 0).cpu().numpy()
            image1 = batch['image1'][0].permute(1, 2, 0).cpu().numpy()
            display_image_in_actual_size(image0)
            display_image_in_actual_size(image1)

        pred0_with_rand = pred0
        pred1_with_rand = pred1
        pred0_with_rand['scores'] = []
        pred1_with_rand['scores'] = []
        pred0_with_rand['descriptors'] = []
        pred1_with_rand['descriptors'] = []
        pred0_with_rand['local_descriptors'] = []
        pred1_with_rand['local_descriptors'] = []
        pred0_with_rand['num_det'] = []
        pred1_with_rand['num_det'] = []
        # 1. detect keypoints of image0 and image1
        kps, score_dispersity, scores = self.softdetect.detect_keypoints(pred0['scores_map'])
        pred0_with_rand['keypoints'] = kps
        pred0_with_rand['score_dispersity'] = score_dispersity
        # 2. detect keypoints
        kps, score_dispersity, scores = self.softdetect.detect_keypoints(pred1['scores_map'])
        pred1_with_rand['keypoints'] = kps
        pred1_with_rand['score_dispersity'] = score_dispersity

        correspondences = []
        for idx in range(b):
            # =========================== prepare keypoints
            kpts0, kpts1 = pred0['keypoints'][idx], pred1['keypoints'][idx]  # (x,y), shape: Nx2

            # additional random keypoints
            if rand:
                rand0 = torch.rand(len(kpts0), 2, device=kpts0.device) * 2 - 1  # -1~1
                rand1 = torch.rand(len(kpts1), 2, device=kpts1.device) * 2 - 1  # -1~1
                kpts0 = torch.cat([kpts0, rand0])
                kpts1 = torch.cat([kpts1, rand1])

                pred0_with_rand['keypoints'][idx] = kpts0
                pred1_with_rand['keypoints'][idx] = kpts1

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]

            scores_kpts0 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()
            scores_kpts1 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()

            kpts0_wh_ = (kpts0 / 2 + 0.5) * wh  # N0x2, (w,h)
            kpts1_wh_ = (kpts1 / 2 + 0.5) * wh  # N1x2, (w,h)

            # ========================= nms
            dist = compute_keypoints_distance(kpts0_wh_.detach(), kpts0_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts0[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts0_wh = kpts0_wh_[valid_mask]
            kpts0 = kpts0[valid_mask]
            scores_kpts0 = scores_kpts0[valid_mask]
            pred0_with_rand['keypoints'][idx] = kpts0

            valid_mask = valid_mask[:len(pred0_with_rand['score_dispersity'][idx])]
            pred0_with_rand['score_dispersity'][idx] = pred0_with_rand['score_dispersity'][idx][valid_mask]
            pred0_with_rand['num_det'].append(valid_mask.sum())

            dist = compute_keypoints_distance(kpts1_wh_.detach(), kpts1_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts1[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts1_wh = kpts1_wh_[valid_mask]
            kpts1 = kpts1[valid_mask]
            scores_kpts1 = scores_kpts1[valid_mask]
            pred1_with_rand['keypoints'][idx] = kpts1

            valid_mask = valid_mask[:len(pred1_with_rand['score_dispersity'][idx])]
            pred1_with_rand['score_dispersity'][idx] = pred1_with_rand['score_dispersity'][idx][valid_mask]
            pred1_with_rand['num_det'].append(valid_mask.sum())

            # del dist, local_mask, valid_cnt, indices_need_nms, scs_max_idx, tmp_mask, suppressed_indices, valid_mask
            # torch.cuda.empty_cache()
            # ========================= nms

            pred0_with_rand['scores'].append(scores_kpts0)
            pred1_with_rand['scores'].append(scores_kpts1)

            descriptor_map0, descriptor_map1 = pred0['descriptor_map'][idx], pred1['descriptor_map'][idx]
            local_descriptor_map0, local_descriptor_map1 = pred0['local_descriptor'][idx], pred1['local_descriptor'][
                idx]

            desc0 = torch.nn.functional.grid_sample(descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, :, 0, :].t()
            desc1 = torch.nn.functional.grid_sample(descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, :, 0, :].t()
            local_desc0 = torch.nn.functional.grid_sample(local_descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                          mode='bilinear', align_corners=True)[0, :, 0, :].t()
            local_desc1 = torch.nn.functional.grid_sample(local_descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                          mode='bilinear', align_corners=True)[0, :, 0, :].t()

            desc0 = torch.nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)
            local_desc0 = torch.nn.functional.normalize(local_desc0, p=2, dim=1)
            local_desc1 = torch.nn.functional.normalize(local_desc1, p=2, dim=1)

            pred0_with_rand['descriptors'].append(desc0)
            pred1_with_rand['descriptors'].append(desc1)
            pred0_with_rand['local_descriptors'].append(local_desc0)
            pred1_with_rand['local_descriptors'].append(local_desc1)

            # =========================== prepare warp parameters
            warp01_params = {}
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[idx]
            warp10_params = {}
            for k, v in batch['warp10_params'].items():
                warp10_params[k] = v[idx]

            # =========================== warp keypoints across images
            try:
                # valid keypoint, valid warped keypoint, valid indices
                kpts0_wh, kpts01_wh, ids0, ids0_out = warp(kpts0_wh, warp01_params)
                kpts1_wh, kpts10_wh, ids1, ids1_out = warp(kpts1_wh, warp10_params)
                if len(kpts0_wh) == 0 or len(kpts1_wh) == 0 or len(kpts0) == 0 or len(kpts1) == 0:
                    raise EmptyTensorError
            except EmptyTensorError:
                correspondences.append({'correspondence0': None, 'correspondence1': None,
                                        'dist': kpts0_wh.new_tensor(0),
                                        })
                continue

            if self.debug:
                from utils import display_image_in_actual_size
                image0 = batch['image0'][0].permute(1, 2, 0).cpu().numpy()
                image1 = batch['image1'][0].permute(1, 2, 0).cpu().numpy()

                p0 = kpts0_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts0 = plot_keypoints(image0, p0, radius=1, color=(255, 0, 0))
                # display_image_in_actual_size(img_kpts0)

                p1 = kpts1_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts1 = plot_keypoints(image1, p1, radius=1, color=(255, 0, 0))
                # display_image_in_actual_size(img_kpts1)

                p01 = kpts01_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts01 = plot_keypoints(img_kpts1, p01, radius=1, color=(0, 255, 0))
                display_image_in_actual_size(img_kpts01)

                p10 = kpts10_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts10 = plot_keypoints(img_kpts0, p10, radius=1, color=(0, 255, 0))
                display_image_in_actual_size(img_kpts10)

            # ============================= compute reprojection error
            dist01 = compute_keypoints_distance(kpts0_wh, kpts10_wh)
            dist10 = compute_keypoints_distance(kpts1_wh, kpts01_wh)

            dist_l2 = (dist01 + dist10.t()) / 2.
            # find mutual correspondences by calculating the distance
            # between keypoints (I1) and warpped keypoints (I2->I1)
            mutual_min_indices = mutual_argmin(dist_l2)

            dist_mutual_min = dist_l2[mutual_min_indices]
            valid_dist_mutual_min = dist_mutual_min.detach() < self.train_gt_th

            ids0_d = mutual_min_indices[0][valid_dist_mutual_min]
            ids1_d = mutual_min_indices[1][valid_dist_mutual_min]

            correspondence0 = ids0[ids0_d]
            correspondence1 = ids1[ids1_d]

            # L1 distance
            dist01_l1 = compute_keypoints_distance(kpts0_wh, kpts10_wh, p=1)
            dist10_l1 = compute_keypoints_distance(kpts1_wh, kpts01_wh, p=1)

            dist_l1 = (dist01_l1 + dist10_l1.t()) / 2.

            # =========================== compute cross image descriptor similarity_map
            similarity_map_01 = torch.einsum('nd,dhw->nhw', desc0, descriptor_map1)
            similarity_map_10 = torch.einsum('nd,dhw->nhw', desc1, descriptor_map0)
            local_similarity_map_01 = torch.einsum('nd,dhw->nhw', local_desc0, local_descriptor_map1)
            local_similarity_map_10 = torch.einsum('nd,dhw->nhw', local_desc1, local_descriptor_map0)

            similarity_map_01_valid = similarity_map_01[ids0]  # valid descriptors
            similarity_map_10_valid = similarity_map_10[ids1]

            local_similarity_map_01_valid = local_similarity_map_01[ids0]  # valid descriptors
            local_similarity_map_10_valid = local_similarity_map_10[ids1]

            kpts01 = 2 * kpts01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kpts10 = 2 * kpts10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]

            correspondences.append({'correspondence0': correspondence0,  # indices of matched kpts0 in all kpts
                                    'correspondence1': correspondence1,  # indices of matched kpts1 in all kpts
                                    'scores0': scores_kpts0[ids0],
                                    'scores1': scores_kpts1[ids1],
                                    'kpts01': kpts01, 'kpts10': kpts10,  # warped valid kpts
                                    'ids0': ids0, 'ids1': ids1,  # valid indices of kpts0 and kpts1
                                    'ids0_out': ids0_out, 'ids1_out': ids1_out,
                                    'ids0_d': ids0_d, 'ids1_d': ids1_d,  # match indices of valid kpts0 and kpts1
                                    'dist_l1': dist_l1,  # cross distance matrix of valid kpts using L1 norm
                                    'dist': dist_l2,  # cross distance matrix of valid kpts using L2 norm
                                    'similarity_map_01': similarity_map_01,  # all
                                    'similarity_map_10': similarity_map_10,  # all
                                    'similarity_map_01_valid': similarity_map_01_valid,  # valid
                                    'similarity_map_10_valid': similarity_map_10_valid,  # valid
                                    'local_similarity_map_01': local_similarity_map_01,  # all
                                    'local_similarity_map_10': local_similarity_map_10,  # all
                                    'local_similarity_map_01_valid': local_similarity_map_01_valid,  # valid
                                    'local_similarity_map_10_valid': local_similarity_map_10_valid,  # valid
                                    })

        return correspondences, pred0_with_rand, pred1_with_rand

    def evaluate(self, pred, batch):
        b = len(pred['kpts0'])

        accuracy = []
        for idx in range(b):
            kpts0, kpts1 = pred['kpts0'][idx][:self.top_k].detach(), pred['kpts1'][idx][:self.top_k].detach()
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()

            matches_est = mutual_argmax(desc0 @ desc1.t())

            mkpts0, mkpts1 = kpts0[matches_est[0]], kpts1[matches_est[1]]

            # warp
            warp01_params = {}
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[idx]

            try:
                mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            except EmptyTensorError:
                continue

            mkpts1 = mkpts1[ids0]

            dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1))
            if dist.shape[0] == 0:
                dist = dist.new_tensor([float('inf')])

            correct = dist < self.eval_gt_th
            accuracy.append(correct.float().mean())

        accuracy = torch.stack(accuracy).mean() if len(accuracy) != 0 else pred['kpts0'][0].new_tensor(0)
        return accuracy

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        b, c, h, w = batch['image0'].shape

        pred0 = super().extract_dense_map(batch['image0'], True)
        pred1 = super().extract_dense_map(batch['image1'], True)
        img0 = batch['image0'].clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        img1 = batch['image1'].clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        s0 = pred0['scores_map'].clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        d0 = pred0['local_descriptor'].clone().detach().cpu().squeeze().permute(1, 2, 0).numpy()
        s1 = pred1['scores_map'].clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        d1 = pred1['local_descriptor'].clone().detach().cpu().squeeze().permute(1, 2, 0).numpy()
        cv2.imwrite('img0.png', img0 * 255)
        cv2.imwrite('img1.png', img1 * 255)
        cv2.imwrite('s0.png', s0 * 255)
        cv2.imwrite('d0.png', d0 * 255)
        cv2.imwrite('s1.png', s1 * 255)
        cv2.imwrite('d1.png', d1 * 255)
        correspondences, pred0_with_rand, pred1_with_rand = self.compute_correspondence(pred0, pred1, batch)

        loss = 0
        loss_package = {}

        if self.w_l_ds > 0:
            loss_ds = self.LocalDescLoss(pred0_with_rand, pred1_with_rand, correspondences)
            loss += self.w_l_ds * loss_ds
            loss_package['loss_local_ds'] = loss_ds

        if self.w_l_pk > 0:
            loss_pk0 = self.LocalPeakyLoss(pred0_with_rand)
            loss_pk1 = self.LocalPeakyLoss(pred1_with_rand)
            loss_pk = self.w_l_pk * (loss_pk0 + loss_pk1) / 2.
            loss += loss_pk
            loss_package['loss_dk'] = loss_pk

        if self.w_pk > 0:
            loss_peaky0 = self.PeakyLoss(pred0_with_rand)
            loss_peaky1 = self.PeakyLoss(pred1_with_rand)
            loss_peaky = (loss_peaky0 + loss_peaky1) / 2.

            loss += self.w_pk * loss_peaky
            loss_package['loss_peaky'] = loss_peaky

        if self.w_rp > 0:
            loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, correspondences)

            loss += self.w_rp * loss_reprojection
            loss_package['loss_reprojection'] = loss_reprojection

        if self.w_sp > 0:
            loss_score_map_rp = self.ScoreMapRepLoss(pred0_with_rand, pred1_with_rand, correspondences)

            loss += self.w_sp * loss_score_map_rp
            loss_package['loss_score_map_rp'] = loss_score_map_rp

        if self.w_ds > 0:
            loss_des = self.DescReprojectionLoss(pred0_with_rand, pred1_with_rand, correspondences)

            loss += self.w_ds * loss_des
            loss_package['loss_des'] = loss_des

        if self.w_triplet > 0:
            loss_triplet = self.TripletLoss(pred0_with_rand, pred1_with_rand, correspondences)

            loss += self.w_triplet * loss_triplet
            loss_package['loss_triplet'] = loss_triplet

        if self.w_i_c > 0:
            loss_i_c = self.IlluConsLoss(pred0, pred1, batch)
            loss += self.w_i_c * loss_i_c
            loss_package['loss_i_c'] = loss_i_c

        if self.w_i_e > 0:
            loss_i_e = self.InfoEntropyLoss(pred0, pred1, batch)
            loss += self.w_i_e * loss_i_e
            loss_package['loss_i_e'] = loss_i_e

        self.log('train/loss', loss)
        for k, v in loss_package.items():
            if 'loss' in k:
                self.log('train/' + k, v)

        pred = {'scores_map0': pred0['scores_map'],
                'scores_map1': pred1['scores_map'],
                'kpts0': [], 'kpts1': [],
                'desc0': [], 'desc1': [],
                'local_desc0': pred0['local_descriptor'],
                'local_desc1': pred1['local_descriptor']}
        for idx in range(b):
            num_det0 = pred0_with_rand['num_det'][idx]
            num_det1 = pred1_with_rand['num_det'][idx]
            pred['kpts0'].append(
                (pred0_with_rand['keypoints'][idx][:num_det0] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
            pred['kpts1'].append(
                (pred1_with_rand['keypoints'][idx][:num_det1] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
            pred['desc0'].append(pred0_with_rand['descriptors'][idx][:num_det0])
            pred['desc1'].append(pred1_with_rand['descriptors'][idx][:num_det1])

        accuracy = self.evaluate(pred, batch)
        self.log('train_acc', accuracy, prog_bar=True)

        if batch_idx % self.log_freq_img == 0:
            self.log_image_and_score(batch, pred, 'train_')

        assert not torch.isnan(loss)
        return loss

    def val_match(self, batch):
        b, _, h0, w0 = batch['image0'].shape
        _, _, h1, w1 = batch['image1'].shape
        assert b == 1

        # ==================================== extract keypoints and descriptors
        top_k_old = self.top_k
        scores_th_old = self.scores_th
        n_limit_old = self.n_limit

        self.top_k, self.scores_th, self.n_limit = 0, self.scores_th_eval, self.n_limit_eval
        self.update_softdetect_parameters()

        pred0 = super().extract(batch['image0'])
        pred1 = super().extract(batch['image1'])


        self.top_k = top_k_old
        self.scores_th = scores_th_old
        self.n_limit = n_limit_old
        self.update_softdetect_parameters()

        kpts0 = keypoints_normal2pixel(pred0['keypoints'], w0, h0)[0]
        kpts1 = keypoints_normal2pixel(pred1['keypoints'], w1, h1)[0]
        desc0 = pred0['descriptors'][0]
        desc1 = pred1['descriptors'][0]

        num_feat = min(kpts0.shape[0], kpts1.shape[0])  # number of detected keypoints

        # ==================================== pack warp params
        warp01_params, warp10_params = {}, {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]

        try:
            # ==================================== covisible keypoints
            kpts0_cov, kpts01_cov, _, _ = warp(kpts0, warp01_params)
            kpts1_cov, kpts10_cov, _, _ = warp(kpts1, warp10_params)
            if len(kpts0_cov) == 0 or len(kpts01_cov) == 0 or len(kpts1_cov) == 0 or len(kpts10_cov) == 0:
                raise EmptyTensorError
            num_cov_feat = (len(kpts0_cov) + len(kpts1_cov)) / 2  # number of covisible keypoints

            # ==================================== get gt matching keypoints
            dist01 = compute_keypoints_distance(kpts0_cov, kpts10_cov)
            dist10 = compute_keypoints_distance(kpts1_cov, kpts01_cov)

            dist_mutual = (dist01 + dist10.t()) / 2.
            imutual = torch.arange(min(dist_mutual.shape), device=dist_mutual.device)
            dist_mutual[imutual, imutual] = 99999  # mask out diagonal

            mutual_min_indices = mutual_argmin(dist_mutual)
            dist = dist_mutual[mutual_min_indices]
            gt_num = (dist <= self.eval_gt_th).sum().cpu()  # number of gt matching keypoints

            # ==================================== putative matches
            matches_est = mutual_argmax(desc0 @ desc1.t())
            mkpts0, mkpts1 = kpts0[matches_est[0]], kpts1[matches_est[1]]

            num_putative = len(mkpts0)  # number of putative matches

            # ==================================== warp putative matches
            mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            mkpts1 = mkpts1[ids0]

            dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1)).cpu()
            if dist.shape[0] == 0:
                dist = dist.new_tensor([float('inf')])

            num_inlier = sum(dist <= self.eval_gt_th)

            return (dist,
                    num_feat,  # feature number
                    gt_num / max(num_cov_feat, 1),  # repeatability
                    num_inlier / max(num_putative, 1),  # accuracy
                    num_inlier / max(num_cov_feat, 1),  # matching score
                    num_inlier / max(gt_num, 1),  # recall
                    )
        except EmptyTensorError:
            return torch.tensor([[0]]), num_feat, 0, 0, 0, 0

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dist, num_feat, repeatability, accuracy, matching_score, recall = self.val_match(batch)

        self.log('val_kpt_num', num_feat)
        self.log('val_repeatability', repeatability)
        self.log('val_acc', accuracy)
        self.log('val_matching_score', matching_score)
        self.log('val_recall', recall)

        self.num_feat.append(num_feat)
        self.repeatability.append(repeatability)
        self.accuracy.append(accuracy)
        self.matching_score.append(matching_score)

        # compute the MMA
        dist = dist.cpu().detach().numpy()
        if dataloader_idx == 0:
            for thr in self.rng:
                self.i_err[thr] += np.mean(dist <= thr)
        elif dataloader_idx == 1:
            for thr in self.rng:
                self.v_err[thr] += np.mean(dist <= thr)
        else:
            pass

        return {'num_feat': num_feat, 'repeatability': repeatability, 'accuracy': accuracy,
                'matching_score': matching_score}


    def on_validation_epoch_start(self):
        # # img1 = '/home/stonehpc/linyicheng/py_project/ALIKE_code/data/image/0053.png'
        # # img2 = '/home/stonehpc/linyicheng/py_project/ALIKE_code/data/image/0073.png'
        # img1 = '/home/stonehpc/Dataset/lk_flow/1/DSC03261.JPG'
        # img2 = '/home/stonehpc/Dataset/lk_flow/1/DSC03262.JPG'
        #
        # img1 = cv2.imread(img1)
        # img2 = cv2.imread(img2)
        # w, h = img1.shape[1], img1.shape[0]
        # img1 = cv2.resize(img1, (640, 360))
        # img2 = cv2.resize(img2, (640, 360))
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # img1 = torch.from_numpy(img1).to(torch.float32).permute(2, 0, 1)[None] / 255.
        # img2 = torch.from_numpy(img2).to(torch.float32).permute(2, 0, 1)[None] / 255.
        #
        # with torch.no_grad():
        #     descriptor_map_1, scores_map_1, local_descriptor_1 = super().extract_dense_map(img1.to(self.device))
        #     descriptor_map_2, scores_map_2, local_descriptor_2 = super().extract_dense_map(img2.to(self.device))
        #     keypoints_1, descriptors_1, scores_1, _ = self.softdetect(scores_map_1, descriptor_map_1)
        #     indices_1 = torch.argsort(scores_1[0], descending=True)
        #     keypoints_1 = keypoints_1[0][indices_1]
        #     descriptors_1 = descriptors_1[0][indices_1]
        #     scores_1 = scores_1[0][indices_1]
        #     keypoints_1 = (keypoints_1 + 1) / 2 * keypoints_1.new_tensor([[w - 1, h - 1]])
        #
        # cv2.imwrite('local_descriptor_1.png',
        #             (local_descriptor_1[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # cv2.imwrite('local_descriptor_2.png',
        #             (local_descriptor_2[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # cv2.imwrite('scores_map_1.png', (scores_map_1[0, 0].cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite('scores_map_2.png', (scores_map_2[0, 0].cpu().numpy() * 255).astype(np.uint8))
        # keypoints_1_top = keypoints_1[:250, :].cpu().numpy()
        #
        # with open('keypoints.txt', 'w') as f:
        #     for i in range(len(keypoints_1_top)):
        #         f.write(str(keypoints_1_top[i][0]) + ',' + str(keypoints_1_top[i][1]) + '\n')
        # f.close()

        # reset
        for thr in self.rng:
            self.i_err[thr] = 0
            self.v_err[thr] = 0
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []

    def on_validation_epoch_end(self):
        # save model
        torch.save(super().state_dict(), 'model.pt')
        # ============= compute average
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))
        accuracy_mean = np.mean(np.array(self.accuracy))
        matching_score_mean = np.mean(np.array(self.matching_score))

        self.log('val_kpt_num_mean', num_feat_mean)
        self.log('val_repeatability_mean', repeatability_mean)
        self.log('val_acc_mean', accuracy_mean)
        self.log('val_matching_score_mean', matching_score_mean)
        self.log('val_metrics/mean',
                 (repeatability_mean + accuracy_mean + matching_score_mean) / 3)

        # ============= compute and draw MMA
        self.errors['ours'] = (self.i_err, self.v_err, 0)
        n_i = 52
        n_v = 56
        MMA = 0
        for i in range(10):
            MMA += (self.i_err[i + 1] + self.v_err[i + 1]) / ((n_i + n_v) * 5)
        MMA = MMA / 10
        # MMA3 = (self.i_err[self.eval_gt_th] + self.v_err[self.eval_gt_th]) / ((n_i + n_v) * 5)
        self.log('val_mma_mean', MMA)

        MMA_image = draw_MMA(self.errors)

        self.logger.experiment.add_image(f'hpatches_MMA', torch.tensor(MMA_image),
                                         global_step=self.global_step, dataformats='HWC')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = self.lr_scheduler(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'scheduled_lr'}]

    def backward(self, loss, *args, **kwargs):
        if loss.requires_grad:
            loss.backward()
        else:
            logging.debug('loss is not backward')

    def log_image_and_score(self, batch, pred, suffix):
        b, c, h, w = pred['scores_map0'].shape

        for idx in range(b):
            if idx > 1:
                break

            image0 = (batch['image0'][idx] * 255).to(torch.uint8).cpu().permute(1, 2, 0)
            image1 = (batch['image1'][idx] * 255).to(torch.uint8).cpu().permute(1, 2, 0)
            scores0 = (pred['scores_map0'][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()
            scores1 = (pred['scores_map1'][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()
            local_desc0 = (pred['local_desc0'][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()
            local_desc1 = (pred['local_desc1'][idx].detach() * 255).to(torch.uint8).cpu().squeeze().numpy()

            kpts0, kpts1 = pred['kpts0'][idx][:self.top_k].detach(), pred['kpts1'][idx][:self.top_k].detach()

            # =================== score map
            s = cv2.applyColorMap(scores0, cv2.COLORMAP_JET)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            self.logger.experiment.add_image(f'{suffix}score/{idx}_src', torch.tensor(s),
                                             global_step=self.global_step, dataformats='HWC')

            s = cv2.applyColorMap(scores1, cv2.COLORMAP_JET)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            self.logger.experiment.add_image(f'{suffix}score/{idx}_tgt', torch.tensor(s),
                                             global_step=self.global_step, dataformats='HWC')

            self.logger.experiment.add_image(f'{suffix}image/{idx}_desc_src', local_desc0,
                                             global_step=self.global_step, dataformats='CHW')
            self.logger.experiment.add_image(f'{suffix}image/{idx}_desc_tgt', local_desc1,
                                             global_step=self.global_step, dataformats='CHW')

            # =================== image with keypoints
            image0_kpts = plot_keypoints(image0, kpts0[:, [1, 0]], radius=1)
            image1_kpts = plot_keypoints(image1, kpts1[:, [1, 0]], radius=1)

            self.logger.experiment.add_image(f'{suffix}image/{idx}_src', torch.tensor(image0_kpts),
                                             global_step=self.global_step, dataformats='HWC')
            self.logger.experiment.add_image(f'{suffix}image/{idx}_tgt', torch.tensor(image1_kpts),
                                             global_step=self.global_step, dataformats='HWC')

            # =================== matches
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()
            matches_est = mutual_argmax(desc0 @ desc1.t())
            mkpts0, mkpts1 = kpts0[matches_est[0]][:, [1, 0]], kpts1[matches_est[1]][:, [1, 0]]

            match_image = plot_matches(image0, image1, mkpts0, mkpts1)
            self.logger.experiment.add_image(f'{suffix}matches/{idx}', torch.tensor(match_image),
                                             global_step=self.global_step, dataformats='HWC')
