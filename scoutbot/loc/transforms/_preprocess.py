# -*- coding: utf-8 -*-
#
#   Image and annotations preprocessing for lightnet networks
#   The image transformations work with both Pillow and OpenCV images
#   The annotation transformations work with brambox.annotations.Annotation objects
#   Copyright EAVISE
#
import collections
import logging

import numpy as np
from PIL import Image, ImageOps

from scoutbot.loc.transforms.util import BaseMultiTransform

log = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    log.warn('OpenCV is not installed and cannot be used')
    cv2 = None

__all__ = ['Letterbox']


class Letterbox(BaseMultiTransform):
    """Transform images and annotations to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """

    def __init__(self, dimension=None, dataset=None):
        super().__init__(dimension=dimension, dataset=dataset)
        if self.dimension is None and self.dataset is None:
            raise ValueError(
                'This transform either requires a dimension or a dataset to infer the dimension'
            )

        self.pad = None
        self.scale = None
        self.fill_color = 127

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.abc.Sequence):
            return self._tf_anno(data)
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log.error(
                f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]'
            )
            return data

    def _tf_pil(self, img):
        """Letterbox an image to fit in the network"""
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            bands = img.split()
            bands = [
                b.resize((int(self.scale * im_w), int(self.scale * im_h))) for b in bands
            ]
            img = Image.merge(img.mode, bands)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w + 0.5), int(pad_h + 0.5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,) * channels)
        return img

    def _tf_cv(self, img):
        """Letterbox and image to fit in the network"""
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            img = cv2.resize(
                img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC
            )
            im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        # channels = img.shape[2] if len(img.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w + 0.5), int(pad_h + 0.5))
        img = cv2.copyMakeBorder(
            img,
            self.pad[1],
            self.pad[3],
            self.pad[0],
            self.pad[2],
            cv2.BORDER_CONSTANT,
            value=self.fill_color,
        )
        return img

    def _tf_anno(self, annos):
        """Change coordinates of an annotation, according to the previous letterboxing"""
        for anno in annos:
            if self.scale is not None:
                anno.x_top_left *= self.scale
                anno.y_top_left *= self.scale
                anno.width *= self.scale
                anno.height *= self.scale
            if self.pad is not None:
                anno.x_top_left += self.pad[0]
                anno.y_top_left += self.pad[1]
        return annos
