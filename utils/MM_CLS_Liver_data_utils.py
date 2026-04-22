# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch
import random

from monai import data, transforms
from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args,train_modality='crop'):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    crop_train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["img_T2WI", "mask_T2WI"],image_only=False),
            transforms.EnsureChannelFirstd(keys=["img_T2WI", "mask_T2WI"], channel_dim="no_channel"),
            transforms.Orientationd(keys=["img_T2WI", "mask_T2WI"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["img_T2WI", "mask_T2WI"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["img_T2WI"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["img_T2WI", "mask_T2WI"], source_key="img_T2WI", select_fn=lambda x: x > 0, margin=0), # 裁剪organ
            transforms.SpatialPadd(keys=["img_T2WI", "mask_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode='constant'),
            transforms.RandCropByPosNegLabeld(
                keys=["img_T2WI", "mask_T2WI"],
                label_key="mask_T2WI",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="img_T2WI",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["img_T2WI", "mask_T2WI"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["img_T2WI", "mask_T2WI"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["img_T2WI", "mask_T2WI"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["img_T2WI", "mask_T2WI"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys=["img_T2WI"], factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys=["img_T2WI"], offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["img_T2WI", "mask_T2WI","label"]),
        ]
    )
    crop_val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["img_T2WI", "mask_T2WI"],image_only=False),
            transforms.EnsureChannelFirstd(keys=["img_T2WI", "mask_T2WI"], channel_dim="no_channel"),
            transforms.Orientationd(keys=["img_T2WI", "mask_T2WI"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["img_T2WI", "mask_T2WI"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["img_T2WI"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["img_T2WI", "mask_T2WI"], source_key="img_T2WI", select_fn=lambda x: x > 0, margin=0), # 裁剪organ
            transforms.Resized(keys=["img_T2WI", "mask_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                               mode=("bilinear", "nearest")),  # selected
            # transforms.SpatialPadd(keys=["img_T2WI", "mask_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #                        mode='constant'),

            transforms.ToTensord(keys=["img_T2WI", "mask_T2WI","label"]),
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["img_T2WI", "mask_T2WI"],image_only=False),
            transforms.EnsureChannelFirstd(keys=["img_T2WI", "mask_T2WI"], channel_dim="no_channel"),
            transforms.Orientationd(keys=["img_T2WI", "mask_T2WI"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["img_T2WI", "mask_T2WI"], pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest")
            ),
            transforms.NormalizeIntensityd(keys="img_T2WI", nonzero=True, channel_wise=True),
            # transforms.CropForegroundd(keys=["img_T2WI", "mask_T2WI"], source_key="img_T2WI", select_fn=lambda x: x > 0, margin=0), # 裁剪organ
            transforms.SpatialPadd(keys=["img_T2WI", "mask_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            transforms.RandCropByPosNegLabeld(
                keys=["img_T2WI", "mask_T2WI"],
                label_key="mask_T2WI",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="img_T2WI",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["img_T2WI", "mask_T2WI"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["img_T2WI", "mask_T2WI"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["img_T2WI", "mask_T2WI"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["img_T2WI", "mask_T2WI"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys=["img_T2WI"], factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys=["img_T2WI"], offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["img_T2WI", "mask_T2WI","label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["img_T2WI", "mask_T2WI"],image_only=False),
            transforms.EnsureChannelFirstd(keys=["img_T2WI", "mask_T2WI"], channel_dim="no_channel"),
            transforms.Orientationd(keys=["img_T2WI", "mask_T2WI"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["img_T2WI", "mask_T2WI"], pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest")
            ),
            transforms.NormalizeIntensityd(keys="img_T2WI", nonzero=True, channel_wise=True),
            # transforms.CropForegroundd(keys=["img_T2WI", "mask_T2WI"], source_key="img_T2WI"),
            transforms.Resized(keys=["img_T2WI", "mask_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                               mode=("bilinear")),  # selected
            # transforms.SpatialPadd(keys=["img_T2WI", "mask_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            #                        mode='constant'),
            transforms.ToTensord(keys=["img_T2WI", "mask_T2WI","label"]),
        ]
    )
    crop_test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["img_T2WI", "mask_T2WI"],image_only=False),
            transforms.EnsureChannelFirstd(keys=["img_T2WI", "mask_T2WI"], channel_dim="no_channel"),
            transforms.Orientationd(keys=["img_T2WI", "mask_T2WI"], axcodes="RAS"),
            transforms.Spacingd(keys=["img_T2WI","mask_T2WI"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("trilinear","nearest")),

            transforms.ScaleIntensityRanged(
                keys=["img_T2WI"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["img_T2WI", "mask_T2WI"], source_key="img_T2WI", select_fn=lambda x: x > 0, margin=0), # 裁剪organ
            transforms.Resized(keys=["img_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                               mode=("bilinear")),
            transforms.ToTensord(keys=["img_T2WI", "mask_T2WI"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["img_T2WI", "mask_T2WI"],image_only=False),
            transforms.EnsureChannelFirstd(keys=["img_T2WI", "mask_T2WI"], channel_dim="no_channel"),
            transforms.Orientationd(keys=["img_T2WI", "mask_T2WI"], axcodes="RAS"),
            transforms.Spacingd(keys=["img_T2WI","mask_T2WI"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear","nearest")),
            # transforms.NormalizeIntensityd(keys="img_T2WI", nonzero=True, channel_wise=True),
            transforms.ScaleIntensityRanged(
                keys=["img_T2WI"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # transforms.CropForegroundd(keys=["img_T2WI","mask_T2WI"], source_key="img_T2WI"),
            transforms.Resized(keys=["img_T2WI"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                               mode=("bilinear")),  # selected
            transforms.ToTensord(keys=["img_T2WI", "mask_T2WI"]),
        ]
    )

    if train_modality == 'crop':
        train_transform = crop_train_transform
        val_transform = crop_val_transform
        test_transform = crop_test_transform
    elif train_modality == 'nocrop':
        train_transform = train_transform
        val_transform = val_transform
        test_transform = test_transform
    elif train_modality == 'T2WI_unlabeled':
        train_transform = crop_train_transform
    elif train_modality == 'T1IP_unlabeled':
        train_transform = crop_train_transform

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "testing", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        if train_modality == 't1_unlabeled' or train_modality == 't1c_unlabeled':
            datalist = load_decathlon_datalist(datalist_json, True, "unlabeled_training", base_dir=data_dir)
        else:
            datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        if train_modality == 't1_unlabeled' or train_modality == 't1c_unlabeled':
            loader = train_loader
        else:
            val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
            val_ds = data.Dataset(data=val_files, transform=val_transform)
            val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
            val_loader = data.DataLoader(
                val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
            )
            loader = [train_loader, val_loader]

    return loader
