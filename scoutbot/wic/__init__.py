# -*- coding: utf-8 -*-
'''
2022 Wild Me
'''
from os.path import join
import onnxruntime as ort
from pathlib import Path
from scoutbot.wic.dataloader import _init_transforms, ImageFilePathList, BATCH_SIZE, INPUT_SIZE
import numpy as np
import utool as ut
import torch


PWD = Path(__file__).absolute().parent

ONNX_MODEL = join(PWD, 'models', 'onnx', 'scout.wic.5fbfff26.3.0.onnx')
ONNX_CLASSES = ['negative', 'positive']


def pre(inputs):
    transform = _init_transforms()
    dataset = ImageFilePathList(inputs, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False
    )

    data = []
    for data_, in dataloader:
        data += data_.tolist()

    return data


def predict(data):
    ort_session = ort.InferenceSession(
        ONNX_MODEL, 
        providers=['CPUExecutionProvider']
    )

    preds = []
    for chunk in ut.ichunks(data, BATCH_SIZE):
        trim = len(chunk)
        while(len(chunk)) < BATCH_SIZE:
            chunk.append(np.random.randn(3, INPUT_SIZE, INPUT_SIZE).astype(np.float32))
        input_ = np.array(chunk, dtype=np.float32)

        pred_ = ort_session.run(
            None,
            {'input': input_},
        )
        preds += pred_[0].tolist()[:trim]

    return preds


def post(preds):
    outputs = [
        dict(zip(ONNX_CLASSES, pred)) 
        for pred in preds
    ]
    return outputs
