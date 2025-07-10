import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from typing import Optional

from . import dla as dla_modules
from .herdnet_lmds import HerdNetLMDS
from typing import List, Optional, Union, Dict, Tuple

class HerdNet(nn.Module):
    ''' HerdNet architecture '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64
        ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            num_classes (int, optional): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16. 
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end 
                of decoder. Defaults to 64.
        '''

        super(HerdNet, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        base_name = 'dla{}'.format(num_layers)

        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.head_conv = head_conv

        self.first_level = int(np.log2(down_ratio))

        # backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)

        channels = self.channels_0

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)

        # bottleneck conv
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1], 
            kernel_size=1, stride=1, 
            padding=0, bias=True
        )

        # localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, 1, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                ),
            nn.Sigmoid()
            )

        self.loc_head[-2].bias.data.fill_(0.00)

        # classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(channels[-1], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_conv, self.num_classes, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                )
            )

        self.cls_head[-1].bias.data.fill_(0.00)

        # Local Maxima Detection Strategy
        lmds_kwargs: dict = {'kernel_size': (3, 3), 'adapt_ts': 0.2, 'neg_ts': 0.1}
        self.lmds = HerdNetLMDS(up=False, **lmds_kwargs)
        
    def forward(self, input: torch.Tensor):
        encode = self.base_0(input)    
        bottleneck = self.bottleneck_conv(encode[-1])
        encode[-1] = bottleneck
        decode_hm = self.dla_up(encode[self.first_level:])
        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)

        return heatmap, clsmap
    
    def freeze(self, layers: list) -> None:
        ''' Freeze all layers mentioned in the input list '''
        for layer in layers:
            self._freeze_layer(layer)
    
    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
    
    def reshape_classes(self, num_classes: int) -> None:
        ''' Reshape architecture according to a new number of classes.

        Arg:
            num_classes (int): new number of classes
        '''
        
        self.cls_head[-1] = nn.Conv2d(
                self.head_conv, num_classes, 
                kernel_size=1, stride=1, 
                padding=0, bias=True
                )

        self.cls_head[-1].bias.data.fill_(0.00)

        self.num_classes = num_classes
    
    @torch.no_grad()
    def batch_image_detection(self, images: List[np.ndarray], transforms: T.Compose, batch_size: int = 1, device: str = 'cuda:0'):
        self.eval()
        self.device = device
        self.to(self.device)
        # convert images to a tensor of shape [len(images), C, H, W]
        images = torch.stack([transforms(image) for image in images])
        dataset = TensorDataset(images)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
        counts, locs, labels, scores, dscores = [], [], [], [], []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            outputs = self(patch)
            heatmap = outputs[0]
            clsmap = nn.functional.interpolate(outputs[1], scale_factor=16, mode='nearest')
            outmaps = torch.cat([heatmap, clsmap], dim=1)
            # (Upsample)
            outmaps = nn.functional.interpolate(outmaps, scale_factor=2, mode='bilinear', align_corners=True)
            heatmap, clsmap = outmaps[:,:1,:,:], outmaps[:,1:,:,:]
            # Local Maxima Detection Strategy (LMDS)
            counts_patch, locs_patch, labels_patch, scores_patch, dscores_patch = self.lmds((heatmap, clsmap))
            counts.append(counts_patch)
            locs.append(locs_patch)
            labels.append(labels_patch)
            scores.append(scores_patch)
            dscores.append(dscores_patch)

        return counts, locs, labels, scores, dscores
            
