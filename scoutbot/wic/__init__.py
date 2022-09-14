# -*- coding: utf-8 -*-
'''
2022 Wild Me
'''
from os.path import exists, join
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pooch
import torch
import utool as ut

from scoutbot.wic.dataloader import (
    BATCH_SIZE,
    INPUT_SIZE,
    ImageFilePathList,
    _init_transforms,
)

PWD = Path(__file__).absolute().parent

ONNX_MODEL = 'scout.wic.5fbfff26.3.0.onnx'
ONNX_MODEL_PATH = join(PWD, 'models', 'onnx', ONNX_MODEL)
ONNX_MODEL_HASH = 'cbc7f381fa58504e03b6510245b6b2742d63049429337465d95663a6468df4c1'
ONNX_CLASSES = ['negative', 'positive']


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
    transform = _init_transforms()
    dataset = ImageFilePathList(inputs, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False
    )

    data = []
    for (data_,) in dataloader:
        data += data_.tolist()

    return data


def predict(data, fill=False):
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
                    np.random.randn(3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
                )
        input_ = np.array(chunk, dtype=np.float32)

        pred_ = ort_session.run(
            None,
            {'input': input_},
        )
        preds += pred_[0].tolist()[:trim]

    return preds


def post(preds):
    outputs = [dict(zip(ONNX_CLASSES, pred)) for pred in preds]
    return outputs
