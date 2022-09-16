# -*- coding: utf-8 -*-
from os.path import abspath, exists, join, relpath

import cv2
import numpy as np
import utool as ut


def test_tile_grid():
    from scoutbot.tile import TILE_WIDTH, tile_grid

    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))
    assert exists(img_filepath)
    img = cv2.imread(img_filepath)

    assert img.shape == (4000, 6016, 3)

    grid1 = tile_grid(img.shape)
    grid2 = tile_grid(img.shape, offset=TILE_WIDTH // 2, borders=False)
    grid = grid1 + grid2

    assert len(grid1) == 682
    assert len(grid2) == 570
    assert len(set(map(str, grid))) == len(grid)

    assert grid1[0] == {'x': 0, 'y': 0, 'w': 256, 'h': 256, 'b': True}
    assert grid2[0] == {'x': 128, 'y': 176, 'w': 256, 'h': 256, 'b': False}


def test_tile_filepath():
    from scoutbot.tile import tile_filepath

    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))
    assert exists(img_filepath)

    grid = {'x': 0, 'y': 0, 'w': 256, 'h': 256, 'b': True}

    filepath = tile_filepath(img_filepath, grid)
    filepath = relpath(filepath)
    assert filepath == join(
        'examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41_x_0_y_0_w_256_h_256.jpg'
    )

    if exists(filepath):
        ut.delete(filepath, verbose=False)


def test_tile_write():
    from scoutbot.tile import tile_filepath, tile_write

    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))
    assert exists(img_filepath)
    img = cv2.imread(img_filepath)

    grid = {'x': 0, 'y': 0, 'w': 256, 'h': 256, 'b': True}
    filepath = tile_filepath(img_filepath, grid, ext='.png')  # Use lossless here
    assert tile_write(img, grid, filepath)

    assert exists(filepath)
    tile_img = cv2.imread(filepath)

    x0 = grid.get('x')
    y0 = grid.get('y')
    w = grid.get('w')
    h = grid.get('h')
    y1 = y0 + h
    x1 = x0 + w
    tile = img[y0:y1, x0:x1]

    assert tile_img.shape == (256, 256, 3)
    assert tile.shape == (256, 256, 3)

    tile = tile.astype(np.float32)
    tile_img = tile_img.astype(np.float32)

    assert np.all(np.abs(tile - tile_img) <= 0)
    ut.delete(filepath, verbose=False)


def test_tile_compute():
    from scoutbot.tile import compute

    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))
    shape, grids, filepaths = compute(img_filepath)

    assert len(filepaths) == 1252
    for filepath in filepaths:
        assert exists(filepath)
        ut.delete(filepath, verbose=False)
