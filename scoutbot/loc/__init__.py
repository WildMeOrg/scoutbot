# -*- coding: utf-8 -*-
'''The localizer (LOC) returns bounding box detections on image tiles.

This module defines how Localizer models are downloaded from an external CDN,
how to load an image and prepare it for inference, demonstrates how to run the
Localization ONNX model on this input, and finally how to convert this raw CNN
output into usable detection bounding boxes with class labels and confidence
scores.
'''
import os
from os.path import exists, join
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pooch
import torch
import torchvision
import tqdm
import utool as ut

from scoutbot import log
from scoutbot.loc.transforms import (
    Compose,
    GetBoundingBoxes,
    Letterbox,
    NonMaxSupression,
    ReverseLetterbox,
    TensorToBrambox,
)

PWD = Path(__file__).absolute().parent

INPUT_SIZE = (416, 416)
INPUT_SIZE_H, INPUT_SIZE_W = INPUT_SIZE
NETWORK_SIZE = (INPUT_SIZE_H, INPUT_SIZE_W, 3)

DEFAULT_CONFIG = os.getenv('CONFIG', 'phase1').strip().lower()
CONFIGS = {
    'phase1': {
        'batch': 16,
        'name': 'scout.loc.5fbfff26.0.onnx',
        'path': join(PWD, 'models', 'onnx', 'scout.loc.5fbfff26.0.onnx'),
        'hash': '85a9378311d42b5143f74570136f32f50bf97c548135921b178b46ba7612b216',
        'classes': ['elephant_savanna'],
        'thresh': 0.4,
        'nms': 0.8,
        'anchors': [
            (1.3221, 1.73145),
            (3.19275, 4.00944),
            (5.05587, 8.09892),
            (9.47112, 4.84053),
            (11.2364, 10.0071),
        ],
    },
    'mvp': {
        'batch': 32,
        'name': 'scout.loc.mvp.0.onnx',
        'path': join(PWD, 'models', 'onnx', 'scout.loc.mvp.0.onnx'),
        'hash': 'AAA',
        'classes': [
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
        'thresh': 0.4,
        'nms': 0.8,
        'anchors': [
            (1.3221, 1.73145),
            (3.19275, 4.00944),
            (5.05587, 8.09892),
            (9.47112, 4.84053),
            (11.2364, 10.0071),
        ],
    },
}
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]
assert DEFAULT_CONFIG in CONFIGS


def fetch(pull=False, config=DEFAULT_CONFIG):
    """
    Fetch the Localizer ONNX model file from a CDN if it does not exist locally.

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

    log.info(f'LOC Model: {onnx_model}')

    return onnx_model


def pre(inputs, config=DEFAULT_CONFIG):
    """
    Load a list of filepaths and return a corresponding list of the image
    data as a 4-D list of floats.  The image data is loaded from disk, transformed
    as needed, and is normalized to the input ranges that the Localizer ONNX model
    expects.

    This function will throw an error if any of the filepaths do not exist.

    Args:
        inputs (list(str)): list of tile image filepaths (relative or absolute)
        config (str or None, optional): the configuration to use, one of ``phase1``
            or ``mvp``.  Defaults to :obj:`None` (the ``phase1`` model).

    Returns:
        generator ( np.ndarray<np.float32>, list ( tuple ( int ) ), int, str ):
            - generator ->
            - - list of transformed image data with shape ``(b, c, w, h)``
            - - list of each tile's original size
            - - trim index
            - - model configuration
    """
    if len(inputs) == 0:
        return [], config

    batch_size = CONFIGS[config]['batch']
    log.info(f'Preprocessing {len(inputs)} LOC inputs in batches of {batch_size}')

    transform = torchvision.transforms.ToTensor()

    for filepaths in ut.ichunks(inputs, batch_size):
        data = np.zeros((batch_size, 3, INPUT_SIZE_H, INPUT_SIZE_W), dtype=np.float32)
        sizes = []
        trim = len(filepaths)

        for index, filepath in enumerate(filepaths):
            img = cv2.imread(filepath)
            size = img.shape[:2][::-1]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Letterbox.apply(img, dimension=INPUT_SIZE)
            img = transform(img)
            img = img.numpy().astype(np.float32)

            data[index] = img
            sizes.append(size)

        while len(sizes) < batch_size:
            sizes.append((0, 0))

        yield data, sizes, trim, config


def predict(gen):
    """
    Run neural network inference using the Localizer's ONNX model on preprocessed data.

    Args:
        gen (generator): generator of batches of transformed image data, the return of
            :meth:`scoutbot.loc.pre`

    Returns:
        generator ( np.ndarray<np.float32>, list ( tuple ( int ) ), str ):
            - generator ->
            - - list of raw ONNX model outputs as shape ``(b, n)``
            - - list of each tile's original size
            - - model configuration
    """
    log.info('Running LOC inference')

    ort_sessions = {}

    for chunk, sizes, trim, config in tqdm.tqdm(gen):
        assert len(chunk) == len(sizes)

        if len(chunk) == 0:
            preds = []
            sizes = []
        else:
            ort_session = ort_sessions.get(config)
            if ort_session is None:
                onnx_model = fetch(config=config)

                ort_session = ort.InferenceSession(
                    onnx_model,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                ort_sessions[config] = ort_session

            assert trim <= len(chunk)

            pred = ort_session.run(
                None,
                {'input': chunk},
            )
            preds = pred[0]

            preds = preds[:trim]
            sizes = sizes[:trim]

        yield preds, sizes, config


def post(gen, loc_thresh=None, nms_thresh=None):
    """
    Apply a post-processing normalization of the raw ONNX network outputs.

    The final output is a list of lists of dictionaries, each representing a single
    detection.  Each dictionary has a structure with the following keys:

        ::

            {
                'l': class_label (str)
                'c': confidence (float)
                'x': x_top_left (float)
                'y': y_top_left (float)
                'w': width (float)
                'h': height (float)
            }

    The ``l`` label is the string class as used when the original
    ONNX model was trained.

    The ``c`` confidence value is a bounded float between ``0.0`` and
    ``1.0`` (inclusive), but should not be treated as a probability.

    The ``x``, ``y``, ``w``, ``h`` bounding box keys are in real pixel values.

    Args:
        gen (generator): generator of batches of raw ONNX model outputs and sizes,
            the return of :meth:`scoutbot.loc.predict`
        loc_thresh (float or None, optional): the confidence threshold for the localizer's
            predictions.  Defaults to None.  Defaults to :obj:`None`
            (the ``phase1`` model).
        nms_thresh (float or None, optional): the non-maximum suppression (NMS) threshold
            for the localizer's predictions.  Defaults to :obj:`None`
            (the ``phase1`` model).

    Returns:
        list ( list ( dict ) ): nested list of Localizer predictions
    """
    log.info('Postprocessing LOC outputs')

    # Exhaust generator and format output
    outputs = []
    for preds, sizes, config in gen:
        assert len(preds) == len(sizes)
        if len(preds) == 0:
            continue

        anchors = CONFIGS[config]['anchors']
        classes = CONFIGS[config]['classes']
        if loc_thresh is None:
            loc_thresh = CONFIGS[config]['thresh']
        if nms_thresh is None:
            nms_thresh = CONFIGS[config]['nms']

        postprocess = Compose(
            [
                GetBoundingBoxes(len(classes), anchors, loc_thresh),
                NonMaxSupression(nms_thresh),
                TensorToBrambox(NETWORK_SIZE, classes),
            ]
        )

        preds = postprocess(torch.tensor(preds))

        for pred, size in zip(preds, sizes):
            output = ReverseLetterbox.apply([pred], INPUT_SIZE, size)
            output = output[0]
            output = [
                {
                    'l': detect.class_label,
                    'c': detect.confidence,
                    'x': detect.x_top_left,
                    'y': detect.y_top_left,
                    'w': detect.width,
                    'h': detect.height,
                }
                for detect in output
            ]
            outputs.append(output)

    return outputs
