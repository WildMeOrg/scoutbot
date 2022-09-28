# -*- coding: utf-8 -*-
from os.path import abspath, join

import scoutbot


def test_fetch():
    scoutbot.fetch(pull=False)
    scoutbot.fetch(pull=True)

    scoutbot.fetch(pull=False, config='phase1')
    scoutbot.fetch(pull=True, config='phase1')

    scoutbot.fetch(pull=False, config='mvp')
    scoutbot.fetch(pull=True, config='mvp')


def test_pipeline_phase1():
    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))

    wic_, detects = scoutbot.pipeline(img_filepath, config='phase1')

    assert abs(wic_ - 1.0) < 1e-2
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


def test_batch_phase1():
    img_filepath = abspath(join('examples', '1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg'))

    img_filepaths = [img_filepath]
    wic_list, detects_list = scoutbot.batch(img_filepaths, config='phase1')
    assert len(wic_list) == 1
    assert len(detects_list) == 1

    wic_ = wic_list[0]
    detects = detects_list[0]

    assert abs(wic_ - 1.0) < 1e-2
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


def test_example():
    scoutbot.example()
