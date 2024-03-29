# -*- coding: utf-8 -*-
from os.path import abspath, join

from scoutbot import agg, loc, tile, wic

import utool as ut


def test_agg_compute_phase1():
    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(img_filepath)
    assert len(tile_filepaths) == 1252

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths, config='phase1')))
    assert len(wic_outputs) == len(tile_filepaths)

    # Threshold for WIC
    flags = [
        wic_output.get('positive') >= wic.CONFIGS['phase1']['thresh']
        for wic_output in wic_outputs
    ]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)
    assert sum(flags) >= 10

    # Run localizer
    loc_outputs = loc.post(loc.predict(loc.pre(loc_tile_filepaths, config='phase1')))
    assert len(loc_tile_grids) == len(loc_outputs)

    # Aggregate
    detects = agg.compute(img_shape, loc_tile_grids, loc_outputs, config='phase1')

    assert len(detects) >= 3

    targets = [
        {'l': 'elephant', 'c': 0.9299, 'x': 4597, 'y': 2322, 'w': 72, 'h': 149},
        {'l': 'elephant', 'c': 0.8739, 'x': 4865, 'y': 2422, 'w': 97, 'h': 109},
        {'l': 'elephant', 'c': 0.7115, 'x': 4806, 'y': 2476, 'w': 66, 'h': 119},
        # {'l': 'elephant', 'c': 0.5236, 'x': 3511, 'y': 1228, 'w': 47, 'h': 78},
    ]

    for output, target in zip(detects, targets):
        for key in target.keys():
            if key == 'l':
                assert output.get(key) == target.get(key)
            elif key == 'c':
                assert abs(output.get(key) - target.get(key)) < 1e-2
            else:
                assert abs(output.get(key) - target.get(key)) < 3


def test_agg_compute_mvp():
    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(img_filepath)
    assert len(tile_filepaths) == 1252

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths, config='mvp')))
    assert len(wic_outputs) == len(tile_filepaths)

    # Threshold for WIC
    flags = [
        wic_output.get('positive') >= wic.CONFIGS['mvp']['thresh']
        for wic_output in wic_outputs
    ]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)
    assert sum(flags) in [123, 125]

    # Run localizer
    loc_outputs = loc.post(loc.predict(loc.pre(loc_tile_filepaths, config='mvp')))
    assert len(loc_tile_grids) == len(loc_outputs)

    # Aggregate
    detects = agg.compute(img_shape, loc_tile_grids, loc_outputs, config='mvp')

    assert len(detects) >= 6

    # fmt: off
    targets = [
        {'l': 'elephant', 'c': 0.6795, 'x': 4593, 'y': 2300, 'w': 78, 'h': 201},
        {'l': 'elephant', 'c': 0.6126, 'x': 4813, 'y': 2452, 'w': 54, 'h': 87},
        {'l': 'kob',      'c': 0.6058, 'x': 3391, 'y': 1076, 'w': 33, 'h': 32},
        {'l': 'elephant', 'c': 0.5933, 'x': 4873, 'y': 2428, 'w': 80, 'h': 99},
        {'l': 'kob',      'c': 0.4767, 'x': 1601, 'y': 1729, 'w': 53, 'h': 55},
        {'l': 'warthog',  'c': 0.4571, 'x': 4199, 'y': 2109, 'w': 31, 'h': 45},
        # {'l': 'kob',      'c': 0.4193, 'x': 1441, 'y': 3377, 'w': 30, 'h': 38},
        # {'l': 'elephant', 'c': 0.4178, 'x': 3891, 'y': 3641, 'w': 60, 'h': 84},
    ]
    # fmt: on

    for output, target in zip(detects, targets):
        for key in target.keys():
            if key == 'l':
                assert output.get(key) == target.get(key)
            elif key == 'c':
                assert abs(output.get(key) - target.get(key)) < 1e-2
            else:
                assert abs(output.get(key) - target.get(key)) < 3
