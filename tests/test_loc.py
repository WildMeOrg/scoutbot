# -*- coding: utf-8 -*-
from os.path import abspath, exists, join

import onnx


def test_loc_onnx_load():
    from scoutbot.loc import fetch

    onnx_model = fetch()
    model = onnx.load(onnx_model)
    assert exists(onnx_model)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 107


def test_loc_onnx_pipeline():
    from scoutbot.loc import INPUT_SIZE, post, pre, predict

    inputs = [
        abspath(join('examples', '0d01a14e-311d-e153-356f-8431b6996b84.true.jpg')),
    ]

    assert exists(inputs[0])

    data, sizes = pre(inputs)

    assert len(data) == 1
    assert len(data[0]) == 3
    assert len(data[0][0]) == INPUT_SIZE[0]
    assert len(data[0][0][0]) == INPUT_SIZE[1]
    assert sizes == [(256, 256)]

    preds = predict(data)

    assert len(preds) == 1
    assert len(preds[0]) == 30

    outputs = post(preds, sizes)

    assert len(outputs) == 1
    assert len(outputs[0]) == 5

    # fmt: off
    targets = [
        {
            'class_label': 'elephant_savanna',
            'x_top_left': 206.00893930,
            'y_top_left': 189.09138371,
            'width'     :  53.78145658,
            'height'    :  66.46106896,
            'confidence':   0.77065581,
        },
        {
            'class_label': 'elephant_savanna',
            'x_top_left': 216.61065204,
            'y_top_left': 193.30525090,
            'width'     :  42.83404541,
            'height'    :  62.44728440,
            'confidence':   0.61152166,
        },
        {
            'class_label': 'elephant_savanna',
            'x_top_left':  51.61210749,
            'y_top_left': 235.37819260,
            'width'     :  79.69709660,
            'height'    :  17.41258826,
            'confidence':   0.50862342,
        },
        {
            'class_label': 'elephant_savanna',
            'x_top_left':  57.47630427,
            'y_top_left': 236.92587515,
            'width'     :  94.69935960,
            'height'    :  16.03246718,
            'confidence':   0.44841822,
        },
        {
            'class_label': 'elephant_savanna',
            'x_top_left':  37.07233605,
            'y_top_left': 230.39122596,
            'width'     : 105.40560208,
            'height'    :  24.81017362,
            'confidence':   0.44012001,
        },
    ]
    # fmt: on

    for output, target in zip(outputs[0], targets):
        for key in target.keys():
            if key == 'class_label':
                assert getattr(output, key) == target.get(key)
            elif key == 'confidence':
                assert abs(getattr(output, key) - target.get(key)) < 1e-2
            else:
                assert abs(getattr(output, key) - target.get(key)) < 3
