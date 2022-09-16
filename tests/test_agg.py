# -*- coding: utf-8 -*-
from os.path import abspath, join

import utool as ut

from scoutbot import agg, loc, tile, wic


def test_agg_compute():
    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(img_filepath)
    assert len(tile_filepaths) == 1252

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths)))
    assert len(wic_outputs) == len(tile_filepaths)

    # Threshold for WIC
    flags = [wic_output.get('positive') >= wic.WIC_THRESH for wic_output in wic_outputs]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)
    assert sum(flags) == 15

    # Run localizer
    loc_data, loc_sizes = loc.pre(loc_tile_filepaths)
    loc_preds = loc.predict(loc_data)
    loc_outputs = loc.post(
        loc_preds, loc_sizes, loc_thresh=loc.LOC_THRESH, nms_thresh=loc.NMS_THRESH
    )
    assert len(loc_tile_grids) == len(loc_outputs)

    # Aggregate
    detects = agg.compute(
        img_shape,
        loc_tile_grids,
        loc_outputs,
        agg_thresh=agg.AGG_THRESH,
        nms_thresh=agg.NMS_THRESH,
    )

    assert len(detects) == 3

    targets = [
        {'l': 'elephant_savanna', 'c': 0.9299, 'x': 4597, 'y': 2322, 'w': 72, 'h': 149},
        {'l': 'elephant_savanna', 'c': 0.8739, 'x': 4865, 'y': 2422, 'w': 97, 'h': 109},
        {'l': 'elephant_savanna', 'c': 0.7115, 'x': 4806, 'y': 2476, 'w': 66, 'h': 119},
    ]

    for output, target in zip(detects, targets):
        for key in target.keys():
            if key == 'l':
                assert output.get(key) == target.get(key)
            elif key == 'c':
                assert abs(output.get(key) - target.get(key)) < 1e-2
            else:
                assert abs(output.get(key) - target.get(key)) < 3
