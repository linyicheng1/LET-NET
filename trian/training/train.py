import os
import sys
import time
import logging
import functools
from pathlib import Path

sys.path.append('../')

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.hpatches import HPatchesDataset
from datasets.kitti import KITTI
from datasets.megadepth import MegaDepthDataset
from datasets.cat_datasets import ConcatDatasets
from training.train_wrapper import TrainWrapper
from training.scheduler import WarmupConstantSchedule

from pytorch_lightning.callbacks import Callback


class RebuildDatasetCallback(Callback):
    def __init__(self):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        pass
        # train_loader_dataset = trainer.train_dataloader
        # train_loader_dataset[0].dataset.rebuild_dataset()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    logging.basicConfig(level=logging.INFO)
    torch.autograd.set_detect_anomaly(True)
    pretrained_model = '/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/linyicheng/py_project/ALIKE_code/training/log_train/train/Version-0708-221706/checkpoints/epoch=79-mean_metric=0.4917.ckpt'
    # pretrained_model = None


    debug = False
    # debug = True

    model_size = 'tiny'
    # model_size = 'tiny'
    # model_size = 'small'
    # model_size = 'big'
    # pe = True
    pe = False
    gray_img = False

    agg_mode = 'cat'
    # agg_mode = 'sum'
    # agg_mode = 'fpn'
    # ========================================= configs
    if model_size == 'small':
        c1 = 8
        c2 = 16
        c3 = 48
        c4 = 96
        dim = 96
        single_head = True
    elif model_size == 'big':
        c1 = 32
        c2 = 64
        c3 = 128
        c4 = 128
        dim = 128
        single_head = False
    elif model_size == 'tiny':
        c1 = 8
        c2 = 16
        c3 = 32
        c4 = 64
        dim = 64
        single_head = True
    else:
        c1 = 16
        c2 = 32
        c3 = 64
        c4 = 128
        dim = 128
        single_head = True

    # ================================== detect parameters
    radius = 2
    top_k = 400
    scores_th_eval = 0.2
    n_limit_eval = 5000

    # ================================== gt reprojection th
    train_gt_th = 5
    eval_gt_th = 3

    # ================================== loss weight
    w_pk = 1
    w_rp = 1
    w_sp = 1
    w_ds = 5
    w_triplet = 0
    w_l_ds = 1
    w_l_pk = 0.
    w_i_c = 0.0
    w_i_e = 0.00
    sc_th = 0.1
    norm = 1
    temp_sp = 0.1
    temp_ds = 0.1

    # ================================== training parameters
    # gpus = [1]
    warmup_steps = 500
    t_total = 10000
    image_size = 480
    log_freq_img = 2000

    # ================================== dataset dir and log dir
    hpatch_dir = '/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/WangShuo/datasets/HPatch'
    kitti_dir = '/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti'
    mega_dir = '/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/disk-data/megadepth/'
    imw2020val_dir = '/media/server/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/disk-data/imw2020-val/'
    log_dir = 'log_' + Path(__file__).stem

    batch_size = 1
    if debug:
        accumulate_grad_batches = 1
        num_workers = 0
        num_sanity_val_steps = 0
        # pretrained_model = 'log_train/train/Version-0701-231352/checkpoints/last.ckpt'
        pretrained_model = 'log_train/train/Version-0708-174505/checkpoints/last.ckpt'
    else:
        accumulate_grad_batches = 16
        num_workers = 8
        num_sanity_val_steps = 1

    # ========================================= model
    lr_scheduler = functools.partial(WarmupConstantSchedule, warmup_steps=warmup_steps)

    model = TrainWrapper(
        # ================================== feature encoder
        c1=c1, c2=c2, c3=c3, c4=c4, dim=dim,
        agg_mode=agg_mode,  # sum, cat, fpn
        single_head=single_head,
        pe=pe,
        grayscale=gray_img,
        # ================================== detect parameters
        radius=radius,
        top_k=top_k, scores_th=0, n_limit=0,
        scores_th_eval=scores_th_eval, n_limit_eval=n_limit_eval,
        # ================================== gt reprojection th
        train_gt_th=train_gt_th, eval_gt_th=eval_gt_th,
        # ================================== loss weight
        w_pk=w_pk,  # weight of peaky loss
        w_rp=w_rp,  # weight of reprojection loss
        w_sp=w_sp,  # weight of score map rep loss
        w_ds=w_ds,  # weight of descriptor loss
        w_triplet=w_triplet,
        w_l_ds=w_l_ds,   # weight of local descriptor loss
        w_l_pk=w_l_pk,  # weight of local peaky loss
        w_i_c=w_i_c,  # weight of image consistency loss
        w_i_e=w_i_e,  # weight of image entropy loss
        sc_th=sc_th,  # score threshold in peaky and  reprojection loss
        norm=norm,  # distance norm
        temp_sp=temp_sp,  # temperature in ScoreMapRepLoss
        temp_ds=temp_ds,  # temperature in DescReprojectionLoss
        # ================================== learning rate
        lr=3e-4,
        log_freq_img=log_freq_img,
        # ================================== pretrained_model
        pretrained_model=pretrained_model,
        lr_scheduler=lr_scheduler,
        debug=debug
    )

    # ========================================= dataloaders
    if debug:
        reload_dataloaders_every_epoch = 0
        limit_train_batches = 1
        limit_val_batches = 1.
        max_epochs = 100
    else:
        reload_dataloaders_every_epoch = 1
        limit_train_batches = 5000 // batch_size
        limit_val_batches = 1.
        max_epochs = 200

    # ========================================= datasets & dataloaders
    # ========== training dataset
    mega_dataset1 = MegaDepthDataset(root=mega_dir, train=True, using_cache=debug, pairs_per_scene=100,
                                     image_size=image_size, gray=gray_img, colorjit=True, crop_or_scale='crop')
    mega_dataset2 = MegaDepthDataset(root=mega_dir, train=True, using_cache=debug, pairs_per_scene=100,
                                     image_size=image_size, gray=gray_img, colorjit=True, crop_or_scale='scale')

    # mega_dataset1 = MegaDepthDataset(root=mega_dir, train=True, using_cache=True, pairs_per_scene=100,
    #                                  image_size=image_size, gray=False, colorjit=True, crop_or_scale='crop')
    # mega_dataset2 = MegaDepthDataset(root=mega_dir, train=True, using_cache=True, pairs_per_scene=100,
    #                                  image_size=image_size, gray=False, colorjit=True, crop_or_scale='scale')
    train_datasets = ConcatDatasets(mega_dataset1, mega_dataset2)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, pin_memory=not debug,
                              num_workers=num_workers)
    # ========== evaluation dataset
    hpatch_i_dataset = HPatchesDataset(root=hpatch_dir, alteration='i', gray=gray_img)
    hpatch_v_dataset = HPatchesDataset(root=hpatch_dir, alteration='v', gray=gray_img)
    hpatch_i_dataloader = DataLoader(hpatch_i_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers)
    hpatch_v_dataloader = DataLoader(hpatch_v_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers)
    kitti_dataset = KITTI(root=kitti_dir, alteration='training')
    kitti_dataloader = DataLoader(kitti_dataset, batch_size=1, pin_memory=not debug, num_workers=num_workers)

    imw2020val = MegaDepthDataset(root=imw2020val_dir, train=False, using_cache=True, colorjit=False, gray=gray_img)
    imw2020val_dataloader = DataLoader(imw2020val, batch_size=1, pin_memory=not debug, num_workers=num_workers)

    # ========================================= logger
    log_name = 'debug' if debug else 'train'
    version = time.strftime("Version-%m%d-%H%M%S", time.localtime())

    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name, version=version, default_hp_metric=False)
    logging.info(f'>>>>>>>>>>>>>>>>> log dir: {logger.log_dir}')

    # ========================================= trainer
    trainer = pl.Trainer(accelerator="gpu", devices=[1],
                         # resume_from_checkpoint='/mnt/data/zxm/document/ALIKE/training/log_train/train/Version-0715-191154/checkpoints/last.ckpt',
                         # resume_from_checkpoint='/mnt/data/zxm/document/ALIKE/training/log_train/train/Version-0702-195918/checkpoints/last.ckpt',
                         fast_dev_run=False,
                         accumulate_grad_batches=accumulate_grad_batches,
                         num_sanity_val_steps=num_sanity_val_steps,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         max_epochs=max_epochs,
                         logger=logger,
                         reload_dataloaders_every_n_epochs=reload_dataloaders_every_epoch,
                         callbacks=[
                             ModelCheckpoint(monitor='val_metrics/mean', save_top_k=3,
                                             mode='max', save_last=True,
                                             dirpath=logger.log_dir + '/checkpoints',
                                             auto_insert_metric_name=False,
                                             filename='epoch={epoch}-mean_metric={val_metrics/mean:.4f}'),
                             LearningRateMonitor(logging_interval='step'),
                             RebuildDatasetCallback()
                         ]
                         )

    trainer.fit(model, train_dataloaders=[train_loader],
                val_dataloaders=[hpatch_i_dataloader, hpatch_v_dataloader])
