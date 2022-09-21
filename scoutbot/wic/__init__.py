# -*- coding: utf-8 -*-
'''The Whole Image Classifier (WIC) returns confidence scores for image tiles.

This module defines how WIC models are downloaded from an external CDN,
how to load an image and prepare it for inference, demonstrates how to run the
WIC ONNX model on this input, and finally how to convert this raw CNN output
into usable confidence scores.
'''
from os.path import exists, join
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pooch
import torch
import tqdm
import utool as ut

from scoutbot import log
from scoutbot.wic.dataloader import (  # NOQA
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

WIC_THRESH = 0.2


def fetch(pull=False):
    """
    Fetch the WIC ONNX model file from a CDN if it does not exist locally.

    This function will throw an AssertionError if the download fails or the
    file otherwise does not exists locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, use a downloaded version stored in
            sthe local system's cache.  Defaults to :obj:`False`.

    Returns:
        str: local ONNX model file path.

    Raises:
        AssertionError: If the model cannot be fetched.
    """
    if not pull and exists(ONNX_MODEL_PATH):
        onnx_model = ONNX_MODEL_PATH
    else:
        onnx_model = pooch.retrieve(
            url=f'https://wildbookiarepository.azureedge.net/models/{ONNX_MODEL}',
            known_hash=ONNX_MODEL_HASH,
            progressbar=True,
        )
        assert exists(onnx_model)

    log.info(f'WIC Model: {onnx_model}')

    return onnx_model


def pre(inputs, batch_size=BATCH_SIZE):
    """
    Load a list of filepaths and return a corresponding list of the image
    data as a 4-D list of floats.  The image data is loaded from disk, transformed
    as needed, and is normalized to the input ranges that the WIC ONNX model
    expects.

    This function will throw an error if any of the filepaths do not exist.

    Args:
        inputs (list(str)): list of tile image filepaths (relative or absolute)

    Returns:
        generator ( list ( list ( list ( list ( float ) ) ) ) ) : generator ->
        list of transformed image data
    """
    assert len(inputs) > 0

    log.info(f'Preprocessing {len(inputs)} WIC inputs in batches of {batch_size}')

    transform = _init_transforms()
    dataset = ImageFilePathList(inputs, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, pin_memory=False
    )

    for (data,) in dataloader:
        yield data.numpy().astype(np.float32)


def predict(gen):
    """
    Run neural network inference using the WIC's ONNX model on preprocessed data.

    Args:
        gen (generator): generator of batches of transformed image data, the
            return of :meth:`scoutbot.wic.pre`

    Returns:
        generator ( list ( list ( float ) ) ): generator -> list of raw ONNX
        model outputs
    """
    onnx_model = fetch()

    log.info('Running WIC inference')

    ort_session = ort.InferenceSession(
        onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    for chunk in tqdm.tqdm(gen):
        if len(chunk) == 0:
            preds = []
        else:
            pred = ort_session.run(
                None,
                {'input': chunk},
            )
            preds = pred[0]
        yield preds


def post(gen):
    """
    Apply a post-processing normalization of the raw ONNX network outputs.

    The final output is a dictionary where the key values are the predicted labels
    and the values are their corresponding confidence values.

    Args:
        gen (generator): generator of batches of raw ONNX model
            outputs, the return of :meth:`scoutbot.wic.predict`

    Returns:
        list ( dict ): list of WIC predictions
    """
    # Exhaust generator and format output
    log.info('Postprocessing WIC outputs')

    outputs = [dict(zip(ONNX_CLASSES, pred.tolist())) for pred in ut.flatten(gen)]
    return outputs
