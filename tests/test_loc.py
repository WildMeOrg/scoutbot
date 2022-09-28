# -*- coding: utf-8 -*-
from os.path import abspath, exists, join

import onnx


def test_loc_onnx_load_phase1():
    from scoutbot.loc import fetch

    onnx_model = fetch(config='phase1')
    model = onnx.load(onnx_model)
    assert exists(onnx_model)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 107


def test_loc_onnx_load_mvp():
    from scoutbot.loc import fetch

    onnx_model = fetch(config='mvp')
    model = onnx.load(onnx_model)
    assert exists(onnx_model)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 107


def test_loc_onnx_pipeline_phase1():
    from scoutbot.loc import CONFIGS, INPUT_SIZE, post, pre, predict

    inputs = [
        abspath(join('examples', '0d01a14e-311d-e153-356f-8431b6996b84.true.jpg')),
    ]

    assert exists(inputs[0])

    data = pre(inputs, config='phase1')
    batch_size = CONFIGS['phase1']['batch']

    temp, sizes, trim, config = next(data)
    assert temp.shape == (batch_size, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    assert len(temp) == len(sizes)
    assert sizes[0] == (256, 256)
    assert set(sizes[1:]) == {(0, 0)}
    assert config == 'phase1'

    data = pre(inputs, config='phase1')
    preds = predict(data)

    temp, sizes, config = next(preds)
    assert temp.shape == (1, 30, 13, 13)
    assert len(temp) == len(sizes)
    assert sizes == [(256, 256)]
    assert config == 'phase1'

    data = pre(inputs, config='phase1')
    preds = predict(data)
    outputs = post(preds)

    assert len(outputs) == 1
    assert len(outputs[0]) == 5
    # assert len(outputs[0]) == 7

    # fmt: off
    targets = [
        {
            'l': 'elephant_savanna',
            'c':   0.77065581,
            'x': 206.00893930,
            'y': 189.09138371,
            'w':  53.78145658,
            'h':  66.46106896,
        },
        {
            'l': 'elephant_savanna',
            'c':   0.61152166,
            'x': 216.61065204,
            'y': 193.30525090,
            'w':  42.83404541,
            'h':  62.44728440,
        },
        {
            'l': 'elephant_savanna',
            'c':   0.50862342,
            'x':  51.61210749,
            'y': 235.37819260,
            'w':  79.69709660,
            'h':  17.41258826,
        },
        {
            'l': 'elephant_savanna',
            'c':   0.44841822,
            'x':  57.47630427,
            'y': 236.92587515,
            'w':  94.69935960,
            'h':  16.03246718,
        },
        {
            'l': 'elephant_savanna',
            'c':   0.44012001,
            'x':  37.07233605,
            'y': 230.39122596,
            'w': 105.40560208,
            'h':  24.81017362,
        },
        # {
        #     'l': 'elephant_savanna',
        #     'c':   0.38498798,
        #     'x':  56.43274395,
        #     'y': 232.00978440,
        #     'w':  99.98320124,
        #     'h':  22.50272075,
        # },
        # {
        #     'l': 'elephant_savanna',
        #     'c':   0.37786528,
        #     'x': 202.67217548,
        #     'y': 178.77696814,
        #     'w':  58.69518573,
        #     'h':  71.09806941,
        # },
    ]
    # fmt: on

    for output, target in zip(outputs[0], targets):
        for key in target.keys():
            if key == 'l':
                assert output.get(key) == target.get(key)
            elif key == 'c':
                assert abs(output.get(key) - target.get(key)) < 1e-2
            else:
                assert abs(output.get(key) - target.get(key)) < 3


def test_loc_onnx_pipeline_mvp():
    from scoutbot.loc import CONFIGS, INPUT_SIZE, post, pre, predict

    inputs = [
        abspath(join('examples', '0d01a14e-311d-e153-356f-8431b6996b84.true.jpg')),
    ]

    assert exists(inputs[0])

    data = pre(inputs, config='mvp')
    batch_size = CONFIGS['mvp']['batch']

    temp, sizes, trim, config = next(data)
    assert temp.shape == (batch_size, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    assert len(temp) == len(sizes)
    assert sizes[0] == (256, 256)
    assert set(sizes[1:]) == {(0, 0)}
    assert config == 'mvp'

    data = pre(inputs, config='mvp')
    preds = predict(data)

    temp, sizes, config = next(preds)
    assert temp.shape == (1, 220, 13, 13)
    assert len(temp) == len(sizes)
    assert sizes == [(256, 256)]
    assert config == 'mvp'

    data = pre(inputs, config='mvp')
    preds = predict(data)
    outputs = post(preds)

    assert len(outputs) == 1
    assert len(outputs[0]) == 8

    # fmt: off
    targets = [
        {
            'l': 'elephant',
            'c':   0.78486251,
            'x': 205.34572190,
            'y': 198.39648437,
            'w':  52.55188457,
            'h':  56.18781456,
        },
        {
            'l': 'elephant',
            'c':   0.54303294,
            'x': 213.27392578,
            'y': 195.15114182,
            'w':  48.83143498,
            'h':  61.92804424,
        },
        {
            'l': 'elephant',
            'c':   0.25485479,
            'x':  39.34061373,
            'y': 227.89024939,
            'w':  99.23480694,
            'h':  26.51788095,
        },
        {
            'l': 'elephant',
            'c':   0.24082227,
            'x':  56.96651517,
            'y': 229.90174278,
            'w':  62.85778339,
            'h':  23.15211838,
        },
        {
            'l': 'elephant',
            'c':   0.22669222,
            'x': 213.39426832,
            'y': 200.48779296,
            'w':  36.94954974,
            'h':  57.41221266,
        },
        {
            'l': 'elephant',
            'c':   0.19940485,
            'x': 219.36613581,
            'y': 205.06403996,
            'w':  41.39131986,
            'h':  46.13519756,
        },
        {
            'l': 'kob',
            'c':   0.17925532,
            'x':   6.99571814,
            'y':   0.92224179,
            'w':  43.32685734,
            'h':  18.18345876,
        },
        {
            'l': 'elephant',
            'c':   0.15872234,
            'x': 160.69904972,
            'y': 235.63134765,
            'w':  51.77306659,
            'h':  19.74641535,
        }
    ]
    # fmt: on

    for output, target in zip(outputs[0], targets):
        for key in target.keys():
            if key == 'l':
                assert output.get(key) == target.get(key)
            elif key == 'c':
                assert abs(output.get(key) - target.get(key)) < 1e-2
            else:
                assert abs(output.get(key) - target.get(key)) < 3


def test_loc_onnx_pipeline_empty():
    from scoutbot.loc import post, pre, predict

    data = pre([])
    preds = predict(data)
    outputs = post(preds)
    assert len(outputs) == 0
