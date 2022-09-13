# -*- coding: utf-8 -*-
import onnx
from os.path import exists, join, abspath


def test_wic_onnx_load():
    from scoutbot.wic import ONNX_MODEL

    model = onnx.load(ONNX_MODEL)
    assert exists(ONNX_MODEL)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 1334


def test_wic_onnx_pipeline():
    from scoutbot.wic import pre, predict, post, ONNX_CLASSES, INPUT_SIZE

    inputs = [
        abspath(join('examples', '1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg')),
    ]

    assert exists(inputs[0])

    data = pre(inputs)

    assert len(data) == 1
    assert len(data[0]) == 3
    assert len(data[0][0]) == INPUT_SIZE
    assert len(data[0][0][0]) == INPUT_SIZE

    preds = predict(data)

    assert len(preds) == 1
    assert len(preds[0]) == 2
    assert preds[0][1] > preds[0][0]
    assert abs(preds[0][0] - 0.00001503) < 1e-6, str(preds)
    assert abs(preds[0][1] - 0.99998497) < 1e-6

    outputs = post(preds)

    assert len(outputs) == 1
    output = outputs[0]
    assert output.keys() == set(ONNX_CLASSES)
    assert output['positive'] > output['negative']
    assert abs(output['negative'] - 0.00001503) < 1e-6
    assert abs(output['positive'] - 0.99998497) < 1e-6