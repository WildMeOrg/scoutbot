# -*- coding: utf-8 -*-
'''The localizer (LOC) returns bounding box detections on image tiles.

This module defines how Localizer models are downloaded from an external CDN,
how to load an image and prepare it for inference, demonstrates how to run the
Localization ONNX model on this input, and finally how to convert this raw CNN
output into usable detection bounding boxes with class labels and confidence
scores.
'''
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
LOC_THRESH = 0.4
NMS_THRESH = 0.8

ONNX_MODEL = 'scout.loc.5fbfff26.0.onnx'
ONNX_MODEL_PATH = join(PWD, 'models', 'onnx', ONNX_MODEL)
ONNX_MODEL_HASH = '85a9378311d42b5143f74570136f32f50bf97c548135921b178b46ba7612b216'


def fetch(pull=False):
    """
    Fetch the Localizer ONNX model file from a CDN if it does not exist locally.

    This function will throw an AssertionError if the download fails or the
    file otherwise does not exists locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, use a downloaded version stored in
            the local system's cache.  Defaults to :obj:`False`.

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
    log.info(f'LOC Model: {onnx_model}')

    return onnx_model


def pre(inputs):
    """
    Load a list of filepaths and return a corresponding list of the image
    data as a 4-D list of floats.  The image data is loaded from disk, transformed
    as needed, and is normalized to the input ranges that the Localizer ONNX model
    expects.

    This function will throw an error if any of the filepaths do not exist.

    Args:
        inputs (list(str)): list of tile image filepaths (relative or absolute)

    Returns:
        generator ( tuple ( list ( list ( list ( list ( float ) ) ) ), list ( tuple ( int ) ) ) ):
            - generator ->
            - - list of transformed image data.
            - - list of each tile's original size.
    """
    if len(inputs) == 0:
        return []

    log.info(f'Preprocessing {len(inputs)} LOC inputs in batches of {BATCH_SIZE}')

    transform = torchvision.transforms.ToTensor()

    for filepaths in ut.ichunks(inputs, BATCH_SIZE):
        data = np.zeros((BATCH_SIZE, 3, INPUT_SIZE_H, INPUT_SIZE_W), dtype=np.float32)
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

        while len(sizes) < BATCH_SIZE:
            sizes.append((0, 0))

        yield data, sizes, trim


def predict(gen):
    """
    Run neural network inference using the Localizer's ONNX model on preprocessed data.

    Args:
        gen (generator): generator of batches of transformed image data, the return of
            :meth:`scoutbot.loc.pre`

    Returns:
        generator ( list ( list ( float ) ), list ( tuple ( int ) ) ) ):
            - generator ->
            - - list of raw ONNX model outputs.
            - - list of each tile's original size.
    """
    onnx_model = fetch()

    log.info('Running LOC inference')

    ort_session = ort.InferenceSession(
        onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    for chunk, sizes, trim in tqdm.tqdm(gen):
        assert len(chunk) == len(sizes)

        if len(chunk) == 0:
            preds = []
            sizes = []
        else:
            assert trim <= len(chunk)

            pred = ort_session.run(
                None,
                {'input': chunk},
            )
            preds = pred[0]

            preds = preds[:trim]
            sizes = sizes[:trim]

        yield preds, sizes


def post(gen, loc_thresh=LOC_THRESH, nms_thresh=NMS_THRESH):
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

    Returns:
        list ( list ( dict ) ): nested list of Localizer predictions
    """
    log.info('Postprocessing LOC outputs')

    postprocess = Compose(
        [
            GetBoundingBoxes(NUM_CLASSES, ANCHORS, loc_thresh),
            NonMaxSupression(nms_thresh),
            TensorToBrambox(NETWORK_SIZE, CLASS_LABEL_MAP),
        ]
    )

    # Exhaust generator and format output
    outputs = []
    for preds, sizes in gen:
        assert len(preds) == len(sizes)
        if len(preds) == 0:
            continue

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
