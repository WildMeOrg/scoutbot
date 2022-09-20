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

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(filepath)

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths)))

    # Threshold for WIC
    flags = [wic_output.get('positive') >= wic_thresh for wic_output in wic_outputs]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)

    # Run localizer
    loc_data, loc_sizes = loc.pre(loc_tile_filepaths)
    loc_preds = loc.predict(loc_data)
    loc_outputs = loc.post(
        loc_preds,
        loc_sizes,
        loc_thresh=loc_thresh,
        nms_thresh=loc_nms_thresh
    )

    # Run Aggregation and get final detections
    detects = agg.compute(
        img_shape,
        loc_tile_grids,
        loc_outputs,
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

VERSION = '0.1.7'
version = VERSION
__version__ = VERSION


def fetch(pull=False):
    """
    Fetch the WIC and Localizer ONNX model files from a CDN if they do not exist locally.

    This function will throw an AssertionError if either download fails or the
    files otherwise do not exist locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, use the downloaded versions stored in
            the local system's cache.  Defaults to :obj:`False`.

    Returns:
        None

    Raises:
        AssertionError: If any model cannot be fetched.
    """
    wic.fetch(pull=pull)
    loc.fetch(pull=pull)


def pipeline(
    filepath,
    wic_thresh=wic.WIC_THRESH,
    loc_thresh=loc.LOC_THRESH,
    loc_nms_thresh=loc.NMS_THRESH,
    agg_thresh=agg.AGG_THRESH,
    agg_nms_thresh=agg.NMS_THRESH,
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

    Returns:
        list ( dict ): list of predictions
    """
    import utool as ut

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(filepath)

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths)))

    # Threshold for WIC
    flags = [wic_output.get('positive') >= wic_thresh for wic_output in wic_outputs]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)

    # Run localizer
    loc_data, loc_sizes = loc.pre(loc_tile_filepaths)
    loc_preds = loc.predict(loc_data)
    loc_outputs = loc.post(
        loc_preds, loc_sizes, loc_thresh=loc_thresh, nms_thresh=loc_nms_thresh
    )
    assert len(loc_tile_grids) == len(loc_outputs)

    # Run Aggregation
    detects = agg.compute(
        img_shape,
        loc_tile_grids,
        loc_outputs,
        agg_thresh=agg_thresh,
        nms_thresh=agg_nms_thresh,
    )

    return detects


def example():
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

    detects = pipeline(img_filepath)

    log.info(ut.repr3(detects))
