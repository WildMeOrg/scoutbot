# -*- coding: utf-8 -*-
"""
pip install torch torchvision onnx onnxruntime-gpu tqdm wbia-utool scikit-learn numpy
"""
import random
import time
from os.path import exists, join, split, splitext

import cv2
import lightnet as ln
import numpy as np
import onnx
import onnxruntime as ort
import sklearn
import torch
import torchvision
import tqdm
import utool as ut
import vtool as vt
import wbia

WITH_GPU = False
BATCH_SIZE = 16


ibs = wbia.opendb(dbdir='/data/db')


pkl_path = 'scout.pkl'
if not exists(pkl_path):
    if False:
        pass
        # tids = ibs.get_valid_gids(is_tile=True)
    else:
        imageset_text_list = ['TEST_SET']
        imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(imageset_text_list)
        gids_list = ibs.get_imageset_gids(imageset_rowid_list)
        gids = ut.flatten(gids_list)
        flags = ibs.get_tile_flags(gids)
        test_gids = ut.filterfalse_items(gids, flags)
        assert sum(ibs.get_tile_flags(test_gids)) == 0
        tids = ibs.scout_get_valid_tile_rowids(gid_list=test_gids)

    random.shuffle(tids)
    positive, negative = [], []
    for chunk_tids in tqdm.tqdm(ut.ichunks(tids, 1000)):
        _, _, chunk_flags = ibs.scout_tile_positive_cumulative_area(chunk_tids)
        chunk_filepaths = ibs.get_image_paths(chunk_tids)
        for index, (tid, flag, filepath) in enumerate(
            zip(chunk_tids, chunk_flags, chunk_filepaths)
        ):
            if not exists(filepath):
                continue
            if flag:
                positive.append(tid)
            else:
                negative.append(tid)
        if len(positive) >= 100 and len(negative) >= 100:
            break
        print(len(positive), len(negative))

    random.shuffle(positive)
    random.shuffle(negative)
    positive = positive[:100]
    negative = negative[:100]
    data = positive + negative
    filepaths = ibs.get_image_paths(data)
    labels = [True] * len(positive) + [False] * len(negative)
    ut.save_cPkl(pkl_path, (data, labels))

    OUTPUT_PATH = '/data/db/checks'
    ut.delete(OUTPUT_PATH)
    ut.ensuredir(OUTPUT_PATH)
    for filepath, label in zip(filepaths, labels):
        path, filename = split(filepath)
        name, ext = splitext(filename)
        tag = 'true' if label else 'false'
        filename_ = f'{name}.{tag}{ext}'
        filepath_ = join(OUTPUT_PATH, filename_)
        if not exists(filepath_):
            ut.copy(filepath, filepath_)

assert exists(pkl_path)
data, labels = ut.load_cPkl(pkl_path)

filepaths = ibs.get_image_paths(data)
orients = ibs.get_image_orientation(data)

assert len(data) == len(set(data))
assert set(ibs.get_image_sizes(data)) == {(256, 256)}
assert sum(map(exists, filepaths)) == len(filepaths)
assert sum(orients) == 0

##########

INDEX = 1

config_path = f'/cache/lightnet/detect.lightnet.scout.5fbfff26.v{INDEX}.py'
weights_path = f'/cache/lightnet/detect.lightnet.scout.5fbfff26.v{INDEX}.weights'
conf_thresh = 0.0
nms_thresh = 0.2

assert exists(config_path)
assert exists(weights_path)

params = ln.engine.HyperParameters.from_file(config_path)
params.load(weights_path)

model = params.network

# Update conf_thresh and nms_thresh in postpsocess
model.postprocess[0].conf_thresh = conf_thresh
model.postprocess[1].nms_thresh = nms_thresh

if WITH_GPU:
    model = model.cuda()
model.eval()

INPUT_SIZE = params.input_dimension
INPUT_SIZE_H, INPUT_SIZE_W = INPUT_SIZE

#############

dataloader = list(zip(filepaths, orients, labels))

transform = torchvision.transforms.ToTensor()

time_pytorch = 0.0
inputs = []
sizes = []
outputs = []
targets = []
for chunk in ut.ichunks(dataloader, BATCH_SIZE):

    filepaths_ = ut.take_column(chunk, 0)
    orients_ = ut.take_column(chunk, 1)
    targets_ = ut.take_column(chunk, 2)

    inputs_ = []
    sizes_ = []
    for filepath, orient in zip(filepaths_, orients_):
        img = vt.imread(filepath, orient=orient)
        size = img.shape[:2][::-1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ln.data.transform.Letterbox.apply(img, dimension=INPUT_SIZE)
        img = transform(img)

        inputs_.append(img)
        sizes_.append(size)
    inputs_ = torch.stack(inputs_)

    if WITH_GPU:
        inputs_ = inputs_.cuda()

    time_start = time.time()
    with torch.set_grad_enabled(False):
        output_ = model(inputs_)
    time_end = time.time()
    time_pytorch += time_end - time_start

    output_transform_ = []
    for out_, size_ in zip(output_, sizes_):
        out_transform_ = ln.data.transform.ReverseLetterbox.apply(
            [out_], INPUT_SIZE, size_
        )
        output_transform_.append(out_transform_[0])

    inputs += inputs_.tolist()
    sizes += sizes_
    outputs += output_transform_
    targets += targets_

predictions_pytorch = outputs

#############

threshs = list(np.arange(0.0, 1.01, 0.01))
best_thresh = None
best_accuracy = 0.0
best_confusion = None
for thresh in tqdm.tqdm(threshs):
    globals().update(locals())
    values = [
        [prediction for prediction in predictions if prediction.confidence >= thresh]
        for predictions in predictions_pytorch
    ]
    values = [len(value) > 0 for value in values]
    accuracy = sklearn.metrics.accuracy_score(targets, values)
    confusion = sklearn.metrics.confusion_matrix(targets, values)
    if accuracy > best_accuracy:
        best_thresh = thresh
        best_accuracy = accuracy
        best_confusion = confusion

tn, fp, fn, tp = best_confusion.ravel()
print(f'Thresh:    {best_thresh}')
print(f'Accuracy:  {best_accuracy}')
print(f'TP:        {tp}')
print(f'TN:        {tn}')
print(f'FP:        {fp}')
print(f'FN:        {fn}')

# Thresh:    0.25
# Accuracy:  0.93
# TP:        88
# TN:        98
# FP:        2
# FN:        12

# Thresh:    0.35
# Accuracy:  0.925
# TP:        85
# TN:        100
# FP:        0
# FN:        15

#############

dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE_H, INPUT_SIZE_W, device='cpu')
input_names = ['input']
output_names = ['output']

model.onnx = True
onnx_filename = f'scout.loc.5fbfff26.{INDEX}.onnx'
output = torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        'input': {0: 'batch_size'},  # variable length axes
        'output': {0: 'batch_size'},
    },
)

###########

model = onnx.load(onnx_filename)
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

###########

ort_session = ort.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

num_classes = params.network.num_classes
anchors = params.network.anchors
network_size = (INPUT_SIZE_H, INPUT_SIZE_W, 3)
class_label_map = params.class_label_map
conf_thresh = 0.0
nms_thresh = 0.2

postprocess = ln.data.transform.Compose(
    [
        ln.data.transform.GetBoundingBoxes(num_classes, anchors, conf_thresh),
        ln.data.transform.NonMaxSupression(nms_thresh),
        ln.data.transform.TensorToBrambox(network_size, class_label_map),
    ]
)

zipped = list(zip(inputs, sizes))

time_onnx = 0.0
outputs = []
for chunk in ut.ichunks(zipped, BATCH_SIZE):

    imgs = ut.take_column(chunk, 0)
    sizes_ = ut.take_column(chunk, 1)

    trim = len(imgs)
    while (len(imgs)) < BATCH_SIZE:
        imgs.append(np.random.randn(3, INPUT_SIZE_H, INPUT_SIZE_W).astype(np.float32))
        sizes_.append(INPUT_SIZE)
    input_ = np.array(imgs, dtype=np.float32)

    time_start = time.time()
    outputs_ = ort_session.run(
        None,
        {'input': input_},
    )
    output_ = postprocess(torch.tensor(outputs_[0]))
    time_end = time.time()
    time_onnx += time_end - time_start

    output_transform_ = []
    for out_, size_ in zip(output_, sizes_):
        out_transform_ = ln.data.transform.ReverseLetterbox.apply(
            [out_], INPUT_SIZE, size_
        )
        output_transform_.append(out_transform_[0])

    outputs += output_transform_[:trim]

predictions_onnx = outputs

###########

globals().update(locals())
values_pytorch = [
    [prediction for prediction in predictions if prediction.confidence >= best_thresh]
    for predictions in predictions_pytorch
]
values_onnx = [
    [prediction for prediction in predictions if prediction.confidence >= best_thresh]
    for predictions in predictions_onnx
]

deviations = []
for value_pytorch, value_onnx in zip(values_pytorch, values_onnx):
    assert len(value_pytorch) == len(value_onnx)
    for value_p, value_o in zip(value_pytorch, value_onnx):
        assert value_p.class_label == value_o.class_label
        for attr in ['x_top_left', 'y_top_left', 'width', 'height', 'confidence']:
            deviation = abs(getattr(value_p, attr) - getattr(value_o, attr))
            deviations.append(deviation)

print(f'Min:  {np.min(deviations):0.08f}')
print(f'Max:  {np.max(deviations):0.08f}')
print(f'Mean: {np.mean(deviations):0.08f} +/- {np.std(deviations):0.08f}')
print(f'Time Pytorch: {time_pytorch:0.02f} sec.')
print(f'Time ONNX:    {time_onnx:0.02f} sec.')

values = [
    [prediction for prediction in predictions if prediction.confidence >= best_thresh]
    for predictions in predictions_onnx
]
values = [len(value) > 0 for value in values]
accuracy = sklearn.metrics.accuracy_score(targets, values)
confusion = sklearn.metrics.confusion_matrix(targets, values)
tn, fp, fn, tp = best_confusion.ravel()

print(f'Thresh:    {best_thresh}')
print(f'Accuracy:  {best_accuracy}')
print(f'TP:        {tp}')
print(f'TN:        {tn}')
print(f'FP:        {fp}')
print(f'FN:        {fn}')

# Min:  0.00000000
# Max:  0.00017841
# Mean: 0.00000904 +/- 0.00001550
# Time Pytorch: 18.18 sec.
# Time ONNX:    9.77 sec.
# Thresh:    0.25
# Accuracy:  0.93
# TP:        88
# TN:        98
# FP:        2
# FN:        12

# Min:  0.00000000
# Max:  0.00011268
# Mean: 0.00000845 +/- 0.00001284
# Time Pytorch: 18.75 sec.
# Time ONNX:    9.72 sec.
# Thresh:    0.35000000000000003
# Accuracy:  0.925
# TP:        85
# TN:        100
# FP:        0
# FN:        15
