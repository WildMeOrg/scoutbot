# -*- coding: utf-8 -*-
"""

pip install torch torchvision onnx onnxruntime-gpu tqdm wbia-utool scikit-learn numpy

"""
import random
import time
from collections import OrderedDict
from os.path import exists, join, split, splitext

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
import torch
import torch.nn as nn
import torchvision
import tqdm
import utool as ut
import wbia
from wbia.algo.detect.densenet import INPUT_SIZE, ImageFilePathList, _init_transforms

WITH_GPU = False
BATCH_SIZE = 128


ibs = wbia.opendb(dbdir='/data/db')


pkl_path = 'scout.pkl'
if not exists(pkl_path):
    if False:
        tids = ibs.get_valid_gids(is_tile=True)
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

assert len(data) == len(set(data))
assert set(ibs.get_image_sizes(data)) == {(256, 256)}
assert sum(map(exists, filepaths)) == len(filepaths)

##########

INDEX = 0

weights_path = f'/cache/wbia/classifier2.scout.5fbfff26.3/classifier2.vulcan.5fbfff26.3/classifier.{INDEX}.weights'

assert exists(weights_path)
weights = torch.load(weights_path, map_location='cpu')
state = weights['state']
classes = weights['classes']

# Initialize the model for this run
model = torchvision.models.densenet201()
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(classes))

# Convert any weights to non-parallel version
new_state = OrderedDict()
for k, v in state.items():
    k = k.replace('module.', '')
    new_state[k] = v

# Load state without parallel
model.load_state_dict(new_state)

# Add softmax
model.classifier = nn.Sequential(model.classifier, nn.LogSoftmax(), nn.Softmax())
if WITH_GPU:
    model = model.cuda()
model.eval()

#############

transforms = _init_transforms()
transform = transforms['test']
dataset = ImageFilePathList(filepaths, labels, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False
)

time_pytorch = 0.0
inputs = []
outputs = []
targets = []
for (inputs_, targets_) in tqdm.tqdm(dataloader, desc='test'):
    if WITH_GPU:
        inputs_ = inputs_.cuda()

    time_start = time.time()
    with torch.set_grad_enabled(False):
        output_ = model(inputs_)
    time_end = time.time()
    time_pytorch += time_end - time_start

    inputs += inputs_.tolist()
    outputs += output_.tolist()
    targets += targets_.tolist()

inputs = np.array(inputs, dtype=np.float32)
globals().update(locals())
predictions_pytorch = [dict(zip(classes, output)) for output in outputs]

#############

threshs = list(np.arange(0.0, 1.01, 0.01))
best_thresh = None
best_accuracy = 0.0
best_confusion = None
for thresh in tqdm.tqdm(threshs):
    globals().update(locals())
    values = [prediction['positive'] >= thresh for prediction in predictions_pytorch]
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

# Thresh:    0.01
# Accuracy:  0.895
# TP:        83
# TN:        96
# FP:        4
# FN:        17

# Thresh:    0.06
# Accuracy:  0.91
# TP:        85
# TN:        97
# FP:        3
# FN:        15

# Thresh:    0.01
# Accuracy:  0.905
# TP:        83
# TN:        98
# FP:        2
# FN:        17

#############

dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE, device='cpu')
input_names = ['input']
output_names = ['output']

onnx_filename = f'scout.wic.5fbfff26.3.{INDEX}.onnx'
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

time_onnx = 0.0
outputs = []
for chunk in ut.ichunks(inputs, BATCH_SIZE):
    trim = len(chunk)
    while (len(chunk)) < BATCH_SIZE:
        chunk.append(np.random.randn(3, INPUT_SIZE, INPUT_SIZE).astype(np.float32))
    input_ = np.array(chunk, dtype=np.float32)

    time_start = time.time()
    output_ = ort_session.run(
        None,
        {'input': input_},
    )
    time_end = time.time()
    time_onnx += time_end - time_start

    outputs += output_[0].tolist()[:trim]

predictions_onnx = [dict(zip(classes, output)) for output in outputs]

###########

values_pytorch = [
    prediction_pytorch['positive'] for prediction_pytorch in predictions_pytorch
]
values_onnx = [prediction_onnx['positive'] for prediction_onnx in predictions_onnx]
deviations = [
    abs(value_pytorch - value_onnx)
    for value_pytorch, value_onnx in zip(values_pytorch, values_onnx)
]

print(f'Min:  {np.min(deviations):0.08f}')
print(f'Max:  {np.max(deviations):0.08f}')
print(f'Mean: {np.mean(deviations):0.08f} +/- {np.std(deviations):0.08f}')
print(f'Time Pytorch: {time_pytorch:0.02f} sec.')
print(f'Time ONNX:    {time_onnx:0.02f} sec.')

globals().update(locals())
values = [prediction['positive'] >= best_thresh for prediction in predictions_onnx]
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
# Max:  0.00000143
# Mean: 0.00000003 +/- 0.00000013
# Time Pytorch: 9.64 sec.
# Time ONNX:    3.17 sec.
# Thresh:    0.01
# Accuracy:  0.895
# TP:        83
# TN:        96
# FP:        4
# FN:        17

# Min:  0.00000000
# Max:  0.00000113
# Mean: 0.00000004 +/- 0.00000013
# Time Pytorch: 9.42 sec.
# Time ONNX:    3.54 sec.
# Thresh:    0.06
# Accuracy:  0.91
# TP:        85
# TN:        97
# FP:        3
# FN:        15


# Min:  0.00000000
# Max:  0.00000209
# Mean: 0.00000004 +/- 0.00000019
# Time Pytorch: 9.98 sec.
# Time ONNX:    3.45 sec.
# Thresh:    0.01
# Accuracy:  0.905
# TP:        83
# TN:        98
# FP:        2
# FN:        17
