# -*- coding: utf-8 -*-
'''The Whole Image Classifier (WIC) returns confidence scores for image tiles.

This module defines how WIC models are downloaded from an external CDN,
how to load an image and prepare it for inference, demonstrates how to run the
WIC ONNX model on this input, and finally how to convert this raw CNN output
into usable confidence scores.
'''
import os
from os.path import exists, join
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pooch
import torch
import tqdm

from scoutbot import log
from scoutbot.wic.dataloader import (  # NOQA
    BATCH_SIZE,
    INPUT_SIZE,
    ImageFilePathList,
    _init_transforms,
)

PWD = Path(__file__).absolute().parent


DEFAULT_CONFIG = os.getenv('CONFIG', 'phase1').strip().lower()
CONFIGS = {
    'phase1': {
        'name': 'scout.wic.5fbfff26.3.0.onnx',
        'path': join(PWD, 'models', 'onnx', 'scout.wic.5fbfff26.3.0.onnx'),
        'hash': 'cbc7f381fa58504e03b6510245b6b2742d63049429337465d95663a6468df4c1',
        'classes': ['negative', 'positive'],
        'thresh': 0.2,
    },
    'mvp': {
        'name': 'scout.wic.mvp.2.0.onnx',
        'path': join(PWD, 'models', 'onnx', 'scout.wic.mvp.2.0.onnx'),
        'hash': '3ff3a192803e53758af5e112526ba9622f1dedc55e2fa88850db6f32af160f32',
        'classes': ['negative', 'positive'],
        'thresh': 0.07,
    },
}
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]
assert DEFAULT_CONFIG in CONFIGS


def fetch(pull=False, config=DEFAULT_CONFIG):
    """
    Fetch the WIC ONNX model file from a CDN if it does not exist locally.

    This function will throw an AssertionError if the download fails or the
    file otherwise does not exists locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, force using the downloaded versions
            stored in the local system's cache.  Defaults to :obj:`False`.
        config (str or None, optional): the configuration to use, one of ``phase1``
            or ``mvp``.  Defaults to :obj:`None` (the ``phase1`` model).

    Returns:
        str: local ONNX model file path.

    Raises:
        AssertionError: If the model cannot be fetched.
    """
    model_name = CONFIGS[config]['name']
    model_path = CONFIGS[config]['path']
    model_hash = CONFIGS[config]['hash']

    if not pull and exists(model_path):
        onnx_model = model_path
    else:
        onnx_model = pooch.retrieve(
            url=f'https://wildbookiarepository.azureedge.net/models/{model_name}',
            known_hash=model_hash,
            progressbar=True,
        )
        assert exists(onnx_model)

    log.info(f'WIC Model: {onnx_model}')

    return onnx_model


def pre(inputs, batch_size=BATCH_SIZE, config=DEFAULT_CONFIG):
    """
    Load a list of filepaths and return a corresponding list of the image
    data as a 4-D list of floats.  The image data is loaded from disk, transformed
    as needed, and is normalized to the input ranges that the WIC ONNX model
    expects.

    This function will throw an error if any of the filepaths do not exist.

    Args:
        inputs (list(str)): list of tile image filepaths (relative or absolute)
        batch_size (int, optional): the maximum number of images to load in a
            single batch.  Defaults to the environment variable ``WIC_BATCH_SIZE``.
        config (str or None, optional): the configuration to use, one of ``phase1``
            or ``mvp``.  Defaults to :obj:`None` (the ``phase1`` model).

    Returns:
        generator ( np.ndarray<np.float32>, str ):
            - generator ->
            - - list of transformed image data with shape ``(b, c, w, h)``
            - - model configuration
    """
    if len(inputs) == 0:
        return [], config

    log.info(f'Preprocessing {len(inputs)} WIC inputs in batches of {batch_size}')

    transform = _init_transforms()
    dataset = ImageFilePathList(inputs, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=False
    )

    for (data,) in dataloader:
        yield data.numpy().astype(np.float32), config


def predict(gen):
    """
    Run neural network inference using the WIC's ONNX model on preprocessed data.

    Args:
        gen (generator): generator of batches of transformed image data, the
            return of :meth:`scoutbot.wic.pre`

    Returns:
        generator ( np.ndarray<np.float32>, str ):
            - generator ->
            - - list of raw ONNX model outputs as shape ``(b, n)``
            - - model configuration
    """
    log.info('Running WIC inference')

    ort_sessions = {}

    for chunk, config in tqdm.tqdm(gen):

        ort_session = ort_sessions.get(config)
        if ort_session is None:
            onnx_model = fetch(config=config)

            ort_session = ort.InferenceSession(
                onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            ort_sessions[config] = ort_session

        if len(chunk) == 0:
            preds = []
        else:
            pred = ort_session.run(
                None,
                {'input': chunk},
            )
            preds = pred[0]
        yield preds, config


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

    outputs = []
    for preds, config in gen:
        classes = CONFIGS[config]['classes']
        for pred in preds:
            output = dict(zip(classes, pred.tolist()))
            outputs.append(output)

    return outputs
