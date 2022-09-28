# -*- coding: utf-8 -*-
import time

import cv2
import gradio as gr
import numpy as np

import scoutbot


def predict(
    filepath, config, wic_thresh, loc_thresh, agg_thresh, loc_nms_thresh, agg_nms_thresh
):
    start = time.time()

    if config == 'MVP':
        config = 'mvp'
    elif config == 'Phase 1':
        config = 'phase1'
    else:
        raise ValueError()

    wic_thresh /= 100.0
    loc_thresh /= 100.0
    loc_nms_thresh /= 100.0
    agg_thresh /= 100.0
    agg_nms_thresh /= 100.0

    loc_nms_thresh = 1.0 - loc_nms_thresh
    agg_nms_thresh = 1.0 - agg_nms_thresh

    # Load data
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    pixels = h * w
    megapixels = pixels / 1e6

    wic_, detects = scoutbot.pipeline(
        filepath,
        config=config,
        wic_thresh=wic_thresh,
        loc_thresh=loc_thresh,
        loc_nms_thresh=loc_nms_thresh,
        agg_thresh=agg_thresh,
        agg_nms_thresh=agg_nms_thresh,
    )

    output = []
    for detect in detects:
        label = detect['l']
        conf = detect['c']
        if conf >= loc_thresh:
            point1 = (
                int(np.around(detect['x'])),
                int(np.around(detect['y'])),
            )
            point2 = (
                int(np.around(detect['x'] + detect['w'])),
                int(np.around(detect['y'] + detect['h'])),
            )
            color = (255, 0, 0)
            img = cv2.rectangle(img, point1, point2, color, 2)
            output.append(f'{label}: {conf:0.04f}')
    output = '\n'.join(output)

    end = time.time()
    duration = end - start
    speed = duration / megapixels
    speed = f'{speed:0.02f} seconds per megapixel (total: {megapixels:0.02f} megapixels, {duration:0.02f} seconds)'

    return img, speed, wic_, output


interface = gr.Interface(
    fn=predict,
    title='Wild Me Scout - Image ML Demo',
    inputs=[
        gr.Image(type='filepath'),
        gr.Radio(
            label='Model Configuration',
            type='value',
            choices=['Phase 1', 'MVP'],
            value='MVP',
        ),
        gr.Slider(label='WIC Confidence Threshold', value=7),
        gr.Slider(label='Localizer Confidence Threshold', value=14),
        gr.Slider(label='Aggregation Confidence Threshold', value=51),
        gr.Slider(label='Localizer NMS Threshold', value=80),
        gr.Slider(label='Aggregation NMS Threshold', value=80),
    ],
    outputs=[
        gr.Image(type='numpy'),
        gr.Textbox(label='Prediction Speed', interactive=False),
        gr.Number(label='Predicted WIC Confidence', precision=5, interactive=False),
        gr.Textbox(label='Predicted Detections', interactive=False),
    ],
    examples=[
        ['examples/0d4e4df2-7b69-91b1-1985-c8421f2f3253.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/18cef191-74ed-2b5e-55a5-f58bd3d483ff.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/1be4d40a-6fd0-42ce-da6c-294e45781f41.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/1d3c85e9-ee24-f290-e7e1-6e338f2eaebb.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/3e043302-af1c-75a7-4057-3a2f25c123bf.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/43ecc08d-502a-7a51-9d68-3e40a76439a2.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/479058af-e774-e6aa-a2b0-9a42dd6ff8b1.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/7c910b87-ae3a-f580-d431-03cd89793803.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/8fa04489-cd94-7d8f-7e2e-5f0fe2f7ae76.jpg', 'MVP', 7, 14, 51, 80, 80],
        ['examples/bb7b4345-b98a-c727-4c94-6090f0aa4355.jpg', 'MVP', 7, 14, 51, 80, 80],
    ],
    cache_examples=True,
    allow_flagging='never',
)

interface.launch(server_name='0.0.0.0')
