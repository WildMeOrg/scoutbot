# -*- coding: utf-8 -*-
'''

'''
from os.path import abspath, exists, join, split, splitext

import cv2
import numpy as np

from scoutbot import log

TILE_WIDTH = 256
TILE_HEIGHT = 256
TILE_SIZE = (TILE_WIDTH, TILE_HEIGHT)
TILE_OVERLAP = 64
TILE_OFFSET = 0
TILE_BORDERS = True


def compute(img_filepath, grid1=True, grid2=True, ext=None, **kwargs):
    """
    Compute the tiles for a given input image and saves them to disk.

    If a given tile has already been rendered to disk, it will not be recomputed.

    Args:
        img_filepath (str): image filepath (relative or absolute) to compute tiles for.
        grid1 (bool, optional): If :obj:`True`, create a dense grid of tiles on the image.
            Defaults to :obj:`True`.
        grid2 (bool, optional): If :obj:`True`, create a secondary dense grid of tiles
            on the image with a 50% offset.  Defaults to :obj:`True`.
        ext (str, optional): The file extension of the resulting tile files.  If this value is
            not specified, it will use the same extension as `img_filepath`.  Passed as input
            to :meth:`scoutbot.tile.tile_filepath`.  Defaults to :obj:`None`.
        **kwargs: keyword arguments passed to :meth:`scoutbot.tile.tile_grid`

    Returns:
        tuple ( tuple ( int ), list ( dict ), list ( str ) ):
            - the original image's shape as ``(h, w, c)``.
            - list of grid coordinates as the output of :meth:`scoutbot.tile.tile_grid`.
            - list of tile filepaths as the output of :meth:`scoutbot.tile.tile_filepath`.
    """
    assert exists(img_filepath)
    img = cv2.imread(img_filepath)
    shape = img.shape

    log.info(f'Computing tiles (grid1={grid1}, grid2={grid2}) on {img_filepath}')

    grids = []
    if grid1:
        grids += tile_grid(shape, **kwargs)
    if grid2:
        grids += tile_grid(shape, offset=TILE_WIDTH // 2, borders=False, **kwargs)

    filepaths = [tile_filepath(img_filepath, grid, ext=ext) for grid in grids]
    for grid, filepath in zip(grids, filepaths):
        assert tile_write(img, grid, filepath)

    log.info(f'Rendered {len(filepaths)} tiles')

    return shape, grids, filepaths


def tile_write(img, grid, filepath):
    """
    Write a single image's tile to disk using its grid coordinates and an output path.

    Args:
        img (numpy.ndarray): 3-dimentional Numpy array, the return from :func:`cv2.imread`
        grid (dict): the grid coordinate dictionary, one of the returned dictionaries
            from :meth:`scoutbot.tile.tile_grid`
        filepath (str): the tile's full output filepath (relative or absolute)

    Returns:
        bool: returns :obj:`True` if the tile's filepath exists on disk.
    """
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
    """
    Returns a suggested filepath for a tile given the original image filepath and the tile's grid
    coordinates.

    Args:
        img_filepath (str): image filepath (relative or absolute)
        grid (dict): a dictionary of one grid coordinate, one output of
            :meth:`scoutbot.tile.tile_grid`
        ext (str, optional): The file extension of the resulting tile files.  If this value is
            not specified, it will use the same extension as `img_filepath`.  Defaults
            to :obj:`None`.

    Returns:
        str: the suggested absolute filepath to store the tile
    """
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
    """
    Calculates a grid of tile coordinates for a given image.

    The final output is a list of lists of dictionaries, each representing a single
    tile coordinate.  Each dictionary has a structure with the following keys:

        ::

            {
                'x': x_top_left (int)
                'y': y_top_left (int)
                'w': width (int)
                'h': height (int)
                'b': border (bool)
            }

    The ``x``, ``y``, ``w``, ``h`` bounding box keys are in real pixel values.

    The ``b`` key is :obj:`True` if the grid coordinate is on the border of the image.

    Args:
        shape (tuple): the image's shape as ``(h, w, c)`` or ``(h, w)``
        size (tuple): the tile's shape as ``(w, h)``
        overlap (int): The amount of pixel overlap between each tile, for both the x-axis
            and the y-axis.
        offset (int): The amount of pixel offset for the entire grid
        borders (bool): If :obj:`True`, include a set of border-only tiles.  Defaults to :obj:`True`.

    Returns:
        list ( dict ): a list of grid coordinate dictionaries
    """
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
