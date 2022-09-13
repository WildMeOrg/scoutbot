# -*- coding: utf-8 -*-
'''
2022 Wild Me
'''
from os.path import abspath, exists, join, split, splitext

import cv2
import numpy as np

TILE_WIDTH = 256
TILE_HEIGHT = 256
TILE_SIZE = (TILE_WIDTH, TILE_HEIGHT)
TILE_OVERLAP = 64
TILE_OFFSET = 0
TILE_BORDERS = True


def compute(img_filepath, grid1=True, grid2=True, ext=None, **kwargs):
    """Compute the tiles for a given input image"""
    assert exists(img_filepath)
    img = cv2.imread(img_filepath)

    grids = []
    if grid1:
        grids += tile_grid(img.shape)
    if grid2:
        grids += tile_grid(img.shape, offset=TILE_WIDTH // 2, borders=False)

    filepaths = [tile_filepath(img_filepath, grid, ext=ext) for grid in grids]
    for grid, filepath in zip(grids, filepaths):
        assert tile_write(img, grid, filepath)

    return filepaths


def tile_write(img, grid, filepath):
    if exists(filepath):
        return True

    x0 = grid.get('x')
    y0 = grid.get('y')
    w = grid.get('w')
    h = grid.get('h')
    y1 = y0 + h
    x1 = x0 + w

    tile = img[y0:y1, x0:x1]
    cv2.imwrite(filepath, tile)
    return exists(filepath)


def tile_filepath(img_filepath, grid, ext=None):
    x = grid.get('x')
    y = grid.get('y')
    w = grid.get('w')
    h = grid.get('h')

    assert exists(img_filepath)
    img_filepath = abspath(img_filepath)

    img_path, img_filename = split(img_filepath)
    img_name, img_ext = splitext(img_filename)

    img_ext = img_ext if ext is None else ext

    filepath = join(img_path, f'{img_name}_x_{x}_y_{y}_w_{w}_h_{h}{img_ext}')
    return filepath


def tile_grid(
    shape, size=TILE_SIZE, overlap=TILE_OVERLAP, offset=TILE_OFFSET, borders=TILE_BORDERS
):
    h_, w_ = shape[:2]
    w, h = size
    ol = overlap
    os = offset

    if borders:
        assert offset == 0, 'Cannot use an offset with borders turned on'

    y_ = int(np.floor((h_ - ol) / (h - ol)))
    x_ = int(np.floor((w_ - ol) / (w - ol)))
    iy = (h * y_) - (ol * (y_ - 1))
    ix = (w * x_) - (ol * (x_ - 1))
    oy = int(np.floor((h_ - iy) * 0.5))
    ox = int(np.floor((w_ - ix) * 0.5))

    miny = 0
    minx = 0
    maxy = h_ - h
    maxx = w_ - w

    ys = list(range(oy, h_ - h + 1, h - ol))
    yb = [False] * len(ys)
    xs = list(range(ox, w_ - w + 1, w - ol))
    xb = [False] * len(xs)

    if borders and oy > 0:
        ys = [miny] + ys + [maxy]
        yb = [True] + yb + [True]

    if borders and ox > 0:
        xs = [minx] + xs + [maxx]
        xb = [True] + xb + [True]

    outputs = []
    for y0, yb_ in zip(ys, yb):
        y0 += os
        y1 = y0 + h
        for x0, xb_ in zip(xs, xb):
            x0 += os
            x1 = x0 + w

            # Sanity, mostly to check for offset
            valid = True
            try:
                assert x1 - x0 == w, '%d, %d' % (
                    x1 - x0,
                    w,
                )
                assert y1 - y0 == h, '%d, %d' % (
                    y1 - y0,
                    h,
                )
                assert 0 <= x0 and x0 <= w_, '%d, %d' % (
                    x0,
                    w_,
                )
                assert 0 <= x1 and x1 <= w_, '%d, %d' % (
                    x1,
                    w_,
                )
                assert 0 <= y0 and y0 <= h_, '%d, %d' % (
                    y0,
                    h_,
                )
                assert 0 <= y1 and y1 <= h_, '%d, %d' % (
                    y1,
                    h_,
                )
            except AssertionError:
                valid = False

            if valid:
                outputs.append({'x': x0, 'y': y0, 'w': w, 'h': h, 'b': yb_ or xb_})

    return outputs
