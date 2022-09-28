# -*- coding: utf-8 -*-
from os.path import abspath, exists, join

import onnx


def test_wic_onnx_load_phase1():
    from scoutbot.wic import fetch

    onnx_model = fetch(config='phase1')
    model = onnx.load(onnx_model)
    assert exists(onnx_model)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 1334


def test_wic_onnx_load_mvp():
    from scoutbot.wic import fetch

    onnx_model = fetch(config='mvp')
    model = onnx.load(onnx_model)
    assert exists(onnx_model)

    onnx.checker.check_model(model)

    graph = onnx.helper.printable_graph(model.graph)
    assert graph.count('\n') == 237


def test_wic_onnx_pipeline_phase1():
    from scoutbot.wic import CONFIGS, INPUT_SIZE, post, pre, predict

    inputs = [
        abspath(join('examples', '1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg')),
    ]

    assert exists(inputs[0])

    data = pre(inputs, config='phase1')

    temp, config = next(data)
    assert temp.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert config == 'phase1'

    data = pre(inputs, config='phase1')
    preds = predict(data)

    temp, config = next(preds)
    assert temp.shape == (1, 2)
    assert temp[0][1] > temp[0][0]
    assert abs(temp[0][0] - 0.00001503) < 1e-4
    assert abs(temp[0][1] - 0.99998497) < 1e-4
    assert config == 'phase1'

    data = pre(inputs, config='phase1')
    preds = predict(data)
    outputs = post(preds)

    assert len(outputs) == 1
    output = outputs[0]
    classes = CONFIGS[None]['classes']
    assert output.keys() == set(classes)
    assert output['positive'] > output['negative']
    assert abs(output['negative'] - 0.00001503) < 1e-4
    assert abs(output['positive'] - 0.99998497) < 1e-4
    assert isinstance(output['negative'], float)
    assert isinstance(output['positive'], float)


def test_wic_onnx_pipeline_mvp():
    from scoutbot.wic import CONFIGS, INPUT_SIZE, post, pre, predict

    inputs = [
        abspath(join('examples', '1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg')),
    ]

    assert exists(inputs[0])

    data = pre(inputs, config='mvp')

    temp, config = next(data)
    assert temp.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert config == 'mvp'

    data = pre(inputs, config='mvp')
    preds = predict(data)

    temp, config = next(preds)
    assert temp.shape == (1, 2)
    assert temp[0][1] > temp[0][0]
    assert abs(temp[0][0] - 0.00000000) < 1e-4
    assert abs(temp[0][1] - 1.00000000) < 1e-4
    assert config == 'mvp'

    data = pre(inputs, config='mvp')
    preds = predict(data)
    outputs = post(preds)

    assert len(outputs) == 1
    output = outputs[0]
    classes = CONFIGS[None]['classes']
    assert output.keys() == set(classes)
    assert output['positive'] > output['negative']
    assert abs(output['negative'] - 0.00000000) < 1e-4
    assert abs(output['positive'] - 1.00000000) < 1e-4
    assert isinstance(output['negative'], float)
    assert isinstance(output['positive'], float)


def test_wic_onnx_pipeline_empty():
    from scoutbot.wic import post, pre, predict

    data = pre([])
    preds = predict(data)
    outputs = post(preds)
    assert len(outputs) == 0
