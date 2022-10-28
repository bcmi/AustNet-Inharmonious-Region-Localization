import random
import numpy as np
import torch.utils.data as data

import cv2

from albumentations import HorizontalFlip, RandomResizedCrop, Compose, DualTransform
import albumentations.augmentations.transforms as transforms
from albumentations.augmentations.geometric.resize import Resize


import os.path
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
import copy


class HCompose(Compose):
    def __init__(self, transforms, *args, additional_targets=None, no_nearest_for_masks=True, **kwargs):
        if additional_targets is None:
            additional_targets = {
                'real': 'image',
                'mask': 'mask',
                'yuv': 'image',
            }
        self.additional_targets = additional_targets
        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)
        if no_nearest_for_masks:
            for t in transforms:
                if isinstance(t, DualTransform):
                    t._additional_targets['mask'] = 'image'
                    # t._additional_targets['edge'] = 'image'


def get_transform(opt, params=None, grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.ToGray())
    if opt.preprocess == 'resize_and_crop':
        if params is None:
            transform_list.append(RandomResizedCrop(opt.crop_size, opt.crop_size, scale=(0.9, 1.0))) # 0.5,1.0
    elif opt.preprocess == 'resize':
        transform_list.append(Resize(opt.crop_size, opt.crop_size))
    elif opt.preprocess == 'none':
        return HCompose(transform_list)

    if not opt.no_flip:
        if params is None:
            # print("flip")
            transform_list.append(HorizontalFlip())

    return HCompose(transform_list)


class iH4Dataset(data.Dataset):
    def __init__(self, opt):
        self.opt = copy.copy(opt)
        self.root = opt.dataset_root

        self.image_paths = []
        self.phase = opt.phase



        if opt.phase=='train':
            # print('loading training file: ')
            self.trainfile = os.path.join(opt.dataset_root,'le50_train.txt')
            self.keep_background_prob = 0.05 # 0.05
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,line.rstrip()))
        elif opt.phase == 'val' or opt.phase == 'test':
            print('loading {} file'.format(opt.phase))
            self.keep_background_prob = -1
            self.trainfile = os.path.join(opt.dataset_root,'le50_{}.txt'.format(opt.phase))
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,line.rstrip()))
                    
        self.transform = get_transform(opt)
        self.input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.46962251, 0.4464104,  0.40718787),
                    (0.27469736, 0.27012361, 0.28515933),
                )
            ])

        self.input_transform_yuv = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.inharmonious_threshold = 1e-2
        self.fg_upper_bound = 0.5

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        comp = self.input_transform(sample['image'])
        real = self.input_transform(sample['real'])
        mask = sample['mask'][np.newaxis, ...].astype(np.float32)
        mask_2 = np.expand_dims(cv2.resize(mask.transpose(1, 2, 0), dsize=(112, 112)), axis = 0)
        mask_4 = np.expand_dims(cv2.resize(mask.transpose(1, 2, 0), dsize=(56, 56)), axis = 0)
        mask_8 = np.expand_dims(cv2.resize(mask.transpose(1, 2, 0), dsize=(28, 28)), axis = 0)
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)
        mask_2 = np.where(mask_2 > 0.5, 1, 0).astype(np.uint8)
        mask_4 = np.where(mask_4 > 0.5, 1, 0).astype(np.uint8)
        mask_8 = np.where(mask_8 > 0.5, 1, 0).astype(np.uint8)

        yuv = self.input_transform_yuv(sample['yuv'])

        output = {
            'comp': comp,
            'mask': mask,
            'mask_2': mask_2,
            'mask_4': mask_4,
            'mask_8': mask_8,
            'real': real,
            'yuv': yuv,
            'img_path':sample['img_path']
        }
        return output



    def augment_sample(self, sample):
        if self.transform is None:
            return sample
        #print(self.transform.additional_targets.keys())
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.transform(image=sample['comp'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            #print(target_name,transformed_target.shape)
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        # return aug_output['mask'].sum() > 1.0
        return aug_output['mask'].sum() > 10

    def check_sample_types(self, sample):
        assert sample['comp'].dtype == 'uint8'
        if 'real' in sample:
            assert sample['real'].dtype == 'uint8'


    def get_sample(self, index):
        path = self.image_paths[index]

        name_parts=path.split('_')

        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_images','real_images')
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')

        comp = cv2.imread(path)
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        real = cv2.imread(target_path)
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32) / 255.

        yuv = cv2.cvtColor(comp, cv2.COLOR_RGB2YCrCb)
       
        return {'comp': comp, 'mask': mask, 'real': real, 'yuv': yuv, 'img_path':path}