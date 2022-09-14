# -*- coding: utf-8 -*-
'''
2022 Wild Me
'''
from os.path import exists, join
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pooch
import torch
import torchvision
import utool as ut

from scoutbot.loc.transforms import (
    Compose,
    GetBoundingBoxes,
    Letterbox,
    NonMaxSupression,
    ReverseLetterbox,
    TensorToBrambox,
)

PWD = Path(__file__).absolute().parent

BATCH_SIZE = 16
INPUT_SIZE = (416, 416)
INPUT_SIZE_H, INPUT_SIZE_W = INPUT_SIZE
NETWORK_SIZE = (INPUT_SIZE_H, INPUT_SIZE_W, 3)

NUM_CLASSES = 1
ANCHORS = [
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
]
CLASS_LABEL_MAP = ['elephant_savanna']
CONF_THRESH = 0.4
NMS_THRESH = 0.8

ONNX_MODEL = 'scout.loc.5fbfff26.0.onnx'
ONNX_MODEL_PATH = join(PWD, 'models', 'onnx', ONNX_MODEL)
ONNX_MODEL_HASH = '85a9378311d42b5143f74570136f32f50bf97c548135921b178b46ba7612b216'


def fetch():
    if exists(ONNX_MODEL_PATH):
        onnx_model = ONNX_MODEL_PATH
    else:
        onnx_model = pooch.retrieve(
            url=f'https://wildbookiarepository.azureedge.net/models/{ONNX_MODEL}',
            known_hash=ONNX_MODEL_HASH,
            progressbar=True,
        )
        assert exists(onnx_model)

    return onnx_model


def pre(inputs):
    transform = torchvision.transforms.ToTensor()

    data = []
    sizes = []
    for filepath in inputs:
        img = cv2.imread(filepath)
        size = img.shape[:2][::-1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Letterbox.apply(img, dimension=INPUT_SIZE)
        img = transform(img)

        data.append(img.tolist())
        sizes.append(size)

    return data, sizes


def predict(data, fill=True):
    onnx_model = fetch()

    ort_session = ort.InferenceSession(
        onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    preds = []
    for chunk in ut.ichunks(data, BATCH_SIZE):
        trim = len(chunk)
        if fill:
            while (len(chunk)) < BATCH_SIZE:
                chunk.append(
                    np.random.randn(3, INPUT_SIZE_H, INPUT_SIZE_W).astype(np.float32)
                )
        input_ = np.array(chunk, dtype=np.float32)

        pred_ = ort_session.run(
            None,
            {'input': input_},
        )
        preds += pred_[0].tolist()[:trim]

    return preds


def post(preds, sizes, loc_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    postprocess = Compose(
        [
            GetBoundingBoxes(NUM_CLASSES, ANCHORS, loc_thresh),
            NonMaxSupression(nms_thresh),
            TensorToBrambox(NETWORK_SIZE, CLASS_LABEL_MAP),
        ]
    )

    preds = postprocess(torch.tensor(preds))

    outputs = []
    for pred, size in zip(preds, sizes):
        output = ReverseLetterbox.apply([pred], INPUT_SIZE, size)
        outputs.append(output[0])

    return outputs
