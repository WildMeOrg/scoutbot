# -*- coding: utf-8 -*-
from os.path import abspath, exists, join

import onnx


def test_wic_onnx_load():
    from scoutbot.wic import fetch

    onnx_model = fetch()
    model = onnx.load(onnx_model)
    assert exists(onnx_model)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 1334


def test_wic_onnx_pipeline():
    from scoutbot.wic import INPUT_SIZE, ONNX_CLASSES, post, pre, predict

    inputs = [
        abspath(join('examples', '1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg')),
    ]

    assert exists(inputs[0])

    data = pre(inputs)

    temp = next(data)
    assert temp.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    data = pre(inputs)
    preds = predict(data)

    temp = next(preds)
    assert temp.shape == (1, 2)
    assert temp[0][1] > temp[0][0]
    assert abs(temp[0][0] - 0.00001503) < 1e-4
    assert abs(temp[0][1] - 0.99998497) < 1e-4

    data = pre(inputs)
    preds = predict(data)
    outputs = post(preds)

    assert len(outputs) == 1
    output = outputs[0]
    assert output.keys() == set(ONNX_CLASSES)
    assert output['positive'] > output['negative']
    assert abs(output['negative'] - 0.00001503) < 1e-4
    assert abs(output['positive'] - 0.99998497) < 1e-4
    assert isinstance(output['negative'], float)
    assert isinstance(output['positive'], float)

    data = pre([])
    preds = predict(data)
    outputs = post(preds)
    assert len(outputs) == 0
