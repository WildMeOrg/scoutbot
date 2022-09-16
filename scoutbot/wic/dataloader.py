# -*- coding: utf-8 -*-
import os

import numpy as np
import PIL
import torch
import torchvision
import utool as ut

BATCH_SIZE = int(os.getenv('WIC_BATCH_SIZE', 256))
INPUT_SIZE = 224


class ImageFilePathList(torch.utils.data.Dataset):
    def __init__(self, filepaths, targets=None, transform=None, target_transform=None):
        from torchvision.datasets.folder import default_loader

        self.targets = targets is not None

        args = (filepaths, targets) if self.targets else (filepaths,)
        self.samples = list(zip(*args))

        if self.targets:
            self.classes = sorted(set(ut.take_column(self.samples, 1)))
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            self.classes, self.class_to_idx = None, None

        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index]

        if self.targets:
            path, target = sample
        else:
            path = sample[0]
            target = None

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        result = (sample, target) if self.targets else (sample,)

        return result

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{}{}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        tmp = '    Target Transforms (if any): '
        fmt_str += '{}{}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        )
        return fmt_str


class Augmentations(object):
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class TestAugmentations(Augmentations):
    def __init__(self, **kwargs):
        from imgaug import augmenters as iaa

        self.aug = iaa.Sequential([iaa.Resize((INPUT_SIZE, INPUT_SIZE))])


def _init_transforms(**kwargs):
    transform = torchvision.transforms.Compose(
        [
            TestAugmentations(**kwargs),
            torchvision.transforms.Lambda(PIL.Image.fromarray),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    return transform
