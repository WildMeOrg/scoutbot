# -*- coding: utf-8 -*-
import lightnet as ln
import torch

__all__ = ['params']


params = ln.engine.HyperParameters(
    # Network
    class_label_map=[
        'buffalo',
        'camel',
        'canoe',
        'car',
        'cow',
        'crocodile',
        'dead_animalwhite_bones',
        'deadbones',
        'eland',
        'elecarcass_old',
        'elephant',
        'gazelle_gr',
        'gazelle_grants',
        'gazelle_th',
        'gazelle_thomsons',
        'gerenuk',
        'giant_forest_hog',
        'giraffe',
        'goat',
        'hartebeest',
        'hippo',
        'impala',
        'kob',
        'kudu',
        'motorcycle',
        'oribi',
        'oryx',
        'ostrich',
        'roof_grass',
        'roof_mabati',
        'sheep',
        'test',
        'topi',
        'vehicle',
        'warthog',
        'waterbuck',
        'white_bones',
        'wildebeest',
        'zebra',
    ],
    input_dimension=(416, 416),
    batch_size=1024,
    mini_batch_size=512,
    max_batches=30000,
    # Dataset
    _train_set='/data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/data/train.pkl',
    _valid_set=None,
    _test_set='/data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/data/test.pkl',
    _filter_anno='ignore',
    # Data Augmentation
    jitter=0.3,
    flip=0.5,
    hue=0.1,
    saturation=1.5,
    value=1.5,
)


# Network
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')


params.network = ln.models.Yolo(
    len(params.class_label_map),
    conf_thresh=0.001,
    nms_thresh=0.5,
)
params.network.postprocess.append(
    ln.data.transform.TensorToBrambox(params.input_dimension, params.class_label_map)
)
params.network.apply(init_weights)

# Optimizers
params.add_optimizer(
    torch.optim.SGD(
        params.network.parameters(),
        lr=0.001 / params.batch_size,
        momentum=0.9,
        weight_decay=0.0005 * params.batch_size,
        dampening=0,
    )
)

# Schedulers
burn_in = torch.optim.lr_scheduler.LambdaLR(
    params.optimizers[0],
    lambda b: (b / 1000) ** 4,
)
step = torch.optim.lr_scheduler.MultiStepLR(
    params.optimizers[0],
    milestones=[20000, 40000],
    gamma=0.1,
)
params.add_scheduler(
    ln.engine.SchedulerCompositor(
        #   batch   scheduler
        (0, burn_in),
        (1000, step),
    )
)
