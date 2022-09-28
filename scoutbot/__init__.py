# -*- coding: utf-8 -*-
'''
The above components must be run in the correct order, but ScoutbBot also offers a single pipeline.

All of the ML models can be pre-downloaded and fetched in a single call to :func:`scoutbot.fetch` and
the unified pipeline -- which uses the 4 components correctly -- can be run by the function
:func:`scoutbot.pipeline`.  Below is example code for how these components interact.

Furthermore, there are two application demo files (``app.py`` and ``app2.py``) that shows
how the entire pipeline can be run on tiles or images, respectively.

.. code-block:: python

    # Get image filepath
    filepath = '/path/to/image.ext'
    config = 'mvp'

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(filepath)

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths, config=config)))

    # Threshold for WIC
    flags = [wic_output.get('positive') >= wic_thresh for wic_output in wic_outputs]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)

    # Run localizer
    loc_outputs = loc.post(
        loc.predict(
            loc.pre(loc_tile_filepaths, config=config)
        ),
        loc_thresh=loc_thresh,
        nms_thresh=loc_nms_thresh
    )

    # Run Aggregation and get final detections
    detects = agg.compute(
        img_shape,
        loc_tile_grids,
        loc_outputs,
        config=config,
        agg_thresh=agg_thresh,
        nms_thresh=agg_nms_thresh,
    )
'''
from os.path import exists

import pooch
import utool as ut

from scoutbot import utils

log = utils.init_logging()


from scoutbot import agg, loc, tile, wic  # NOQA

VERSION = '0.1.15'
version = VERSION
__version__ = VERSION


def fetch(pull=False, config=None):
    """
    Fetch the WIC and Localizer ONNX model files from a CDN if they do not exist locally.

    This function will throw an AssertionError if either download fails or the
    files otherwise do not exist locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, force using the downloaded versions
            stored in the local system's cache.  Defaults to :obj:`False`.
        config (str or None, optional): the configuration to use, one of ``phase1``
            or ``mvp``.  Defaults to :obj:`None` (the ``phase1`` model).

    Returns:
        None

    Raises:
        AssertionError: If any model cannot be fetched.
    """
    wic.fetch(pull=pull, config=None)
    loc.fetch(pull=pull, config=None)


def pipeline(
    filepath,
    config=None,
    wic_thresh=wic.CONFIGS[None]['thresh'],
    loc_thresh=loc.CONFIGS[None]['thresh'],
    loc_nms_thresh=loc.CONFIGS[None]['nms'],
    agg_thresh=agg.CONFIGS[None]['thresh'],
    agg_nms_thresh=agg.CONFIGS[None]['nms'],
    clean=True,
):
    """
    Run the ML pipeline on a given image filepath and return the detections

    The final output is a list of dictionaries, each representing a single detection.
    Each dictionary has a structure with the following keys:

        ::

            {
                'l': class_label (str)
                'c': confidence (float)
                'x': x_top_left (float)
                'y': y_top_left (float)
                'w': width (float)
                'h': height (float)
            }

    Args:
        filepath (str): image filepath (relative or absolute)
        config (str or None, optional): the configuration to use, one of ``phase1``
            or ``mvp``.  Defaults to :obj:`None` (the ``phase1`` model).
        wic_thresh (float or None, optional): the confidence threshold for the WIC's
            predictions.  Defaults to the ``phase1`` configuration setting.
        loc_thresh (float or None, optional): the confidence threshold for the localizer's
            predictions.  Defaults to the ``phase1`` configuration setting.
        nms_thresh (float or None, optional): the non-maximum suppression (NMS) threshold
            for the localizer's predictions.  Defaults to the ``phase1`` configuration setting.
        agg_thresh (float or None, optional): the confidence threshold for the aggregated
            localizer predictions.  Defaults to the ``phase1`` configuration setting.
        agg_nms_thresh (float or None, optional): the non-maximum suppression (NMS) threshold
            for the aggregated localizer's predictions.  Defaults to the ``phase1``
            configuration setting.
        clean (bool, optional): a flag to clean up any on-disk tiles that were generated.
            Defaults to :obj:`True`.

    Returns:
        tuple ( float, list ( dict ) ): wic score, list of predictions
    """
    import utool as ut

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(filepath)

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths, config=config)))

    # Threshold for WIC
    wic_ = max(wic_output.get('positive') for wic_output in wic_outputs)
    flags = [wic_output.get('positive') >= wic_thresh for wic_output in wic_outputs]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)

    log.info(f'Filtered to {len(loc_tile_filepaths)} tiles')

    # Run localizer
    loc_outputs = loc.post(
        loc.predict(loc.pre(loc_tile_filepaths, config=config)),
        loc_thresh=loc_thresh,
        nms_thresh=loc_nms_thresh,
    )
    assert len(loc_tile_grids) == len(loc_outputs)

    # Run Aggregation
    detects = agg.compute(
        img_shape,
        loc_tile_grids,
        loc_outputs,
        config=config,
        agg_thresh=agg_thresh,
        nms_thresh=agg_nms_thresh,
    )

    if clean:
        for tile_filepath in tile_filepaths:
            if exists(tile_filepath):
                ut.delete(tile_filepath, verbose=False)

    return wic_, detects


def batch(
    filepaths,
    config=None,
    wic_thresh=wic.CONFIGS[None]['thresh'],
    loc_thresh=loc.CONFIGS[None]['thresh'],
    loc_nms_thresh=loc.CONFIGS[None]['nms'],
    agg_thresh=agg.CONFIGS[None]['thresh'],
    agg_nms_thresh=agg.CONFIGS[None]['nms'],
    clean=True,
):
    """
    Run the ML pipeline on a given batch of image filepaths and return the detections
    in a corresponding list.  The output is a list of outputs matching the output of
    :func:`scoutbot.pipeline`, except the processing is done in batch and is much faster.

    The final output is a list of lists of dictionaries, each representing a
    single detection.  Each dictionary has a structure with the following keys:

        ::

            {
                'l': class_label (str)
                'c': confidence (float)
                'x': x_top_left (float)
                'y': y_top_left (float)
                'w': width (float)
                'h': height (float)
            }

    Args:
        filepaths (list): list of str image filepath (relative or absolute)
        config (str or None, optional): the configuration to use, one of ``phase1``
            or ``mvp``.  Defaults to :obj:`None` (the ``phase1`` model).
        wic_thresh (float or None, optional): the confidence threshold for the WIC's
            predictions.  Defaults to the ``phase1`` configuration setting.
        loc_thresh (float or None, optional): the confidence threshold for the localizer's
            predictions.  Defaults to the ``phase1`` configuration setting.
        nms_thresh (float or None, optional): the non-maximum suppression (NMS) threshold
            for the localizer's predictions.  Defaults to the ``phase1`` configuration setting.
        agg_thresh (float or None, optional): the confidence threshold for the aggregated
            localizer predictions.  Defaults to the ``phase1`` configuration setting.
        agg_nms_thresh (float or None, optional): the non-maximum suppression (NMS) threshold
            for the aggregated localizer's predictions.  Defaults to the ``phase1``
            configuration setting.
        clean (bool, optional): a flag to clean up any on-disk tiles that were generated.
            Defaults to :obj:`True`.

    Returns:
        tuple ( list ( float ), list ( list ( dict ) ) : corresponding list of wic scores, corresponding list of lists of predictions
    """
    import utool as ut

    # Run tiling
    batch = {}
    for filepath in filepaths:
        img_shape, tile_grids, tile_filepaths = tile.compute(filepath)
        data = {
            'shape': img_shape,
            'grids': tile_grids,
            'filepaths': tile_filepaths,
            'loc': {
                'grids': [],
                'outputs': [],
            },
        }
        batch[filepath] = data

    # Run WIC
    tile_img_filepaths = []
    tile_grids = []
    tile_filepaths = []
    for filepath in filepaths:
        data = batch[filepath]
        batch_grids = data['grids']
        batch_filepaths = data['filepaths']
        assert len(batch_grids) == len(batch_filepaths)
        tile_img_filepaths += [filepath] * len(batch_grids)
        tile_grids += batch_grids
        tile_filepaths += batch_filepaths

    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths, config=config)))

    wic_dict = {}
    for tile_img_filepath, wic_output in zip(tile_img_filepaths, wic_outputs):
        wic_ = wic_output.get('positive')
        existing_wic_ = wic_dict.get(tile_img_filepath, None)
        if existing_wic_ is None:
            existing_wic_ = wic_
        wic_dict[tile_img_filepath] = max(existing_wic_, wic_)

    # Threshold for WIC
    flags = [wic_output.get('positive') >= wic_thresh for wic_output in wic_outputs]
    loc_tile_img_filepaths = ut.compress(tile_img_filepaths, flags)
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)

    log.info(f'Filtered to {len(loc_tile_filepaths)} tiles')

    # Run localizer
    loc_outputs = loc.post(
        loc.predict(loc.pre(loc_tile_filepaths, config=config)),
        loc_thresh=loc_thresh,
        nms_thresh=loc_nms_thresh,
    )
    assert len(loc_tile_grids) == len(loc_outputs)

    for filepath, loc_tile_grid, loc_output in zip(
        loc_tile_img_filepaths, loc_tile_grids, loc_outputs
    ):
        batch[filepath]['loc']['grids'].append(loc_tile_grid)
        batch[filepath]['loc']['outputs'].append(loc_output)

    # Run Aggregation
    wic_list = []
    detects_list = []
    for filepath in filepaths:
        data = batch[filepath]
        wic_ = wic_dict.get(filepath, None)

        img_shape = data['shape']
        loc_tile_grids = data['loc']['grids']
        loc_outputs = data['loc']['outputs']
        assert len(loc_tile_grids) == len(loc_outputs)

        detects = agg.compute(
            img_shape,
            loc_tile_grids,
            loc_outputs,
            config=config,
            agg_thresh=agg_thresh,
            nms_thresh=agg_nms_thresh,
        )

        wic_list.append(wic_)
        detects_list.append(detects)

    if clean:
        for tile_filepath in tile_filepaths:
            if exists(tile_filepath):
                ut.delete(tile_filepath, verbose=False)

    return wic_list, detects_list


def example():
    """
    Run the pipeline on an example image with the Phase 1 models
    """
    TEST_IMAGE = 'scout.example.jpg'
    TEST_IMAGE_HASH = (
        '786a940b062a90961f409539292f09144c3dbdbc6b6faa64c3e764d63d55c988'  # NOQA
    )

    img_filepath = pooch.retrieve(
        url=f'https://wildbookiarepository.azureedge.net/data/{TEST_IMAGE}',
        known_hash=TEST_IMAGE_HASH,
        progressbar=True,
    )
    assert exists(img_filepath)

    log.info(f'Running pipeline on image: {img_filepath}')

    wic_, detects = pipeline(img_filepath)

    log.info(ut.repr3(detects))
