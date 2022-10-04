# -*- coding: utf-8 -*-
import time

import cv2
import gradio as gr
import numpy as np

from scoutbot import loc, wic

PHASE1 = [
    'Phase 1',
    int(wic.CONFIGS['phase1']['thresh'] * 100),
    int(loc.CONFIGS['phase1']['thresh'] * 100),
    int(loc.CONFIGS['phase1']['nms'] * 100),
]
MVP = [
    'MVP',
    int(wic.CONFIGS['mvp']['thresh'] * 100),
    int(loc.CONFIGS['mvp']['thresh'] * 100),
    int(loc.CONFIGS['mvp']['nms'] * 100),
]


def predict(filepath, config, wic_thresh, loc_thresh, nms_thresh):
    start = time.time()

    if config == MVP[0]:
        config = 'mvp'
    elif config == PHASE1[0]:
        config = 'phase1'
    else:
        raise ValueError()

    wic_thresh /= 100.0
    loc_thresh /= 100.0
    nms_thresh /= 100.0

    # Load data
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run WIC
    inputs = [filepath]
    outputs = wic.post(wic.predict(wic.pre(inputs, config=config)))

    # Get WIC confidence
    output = outputs[0]
    wic_confidence = output.get('positive')

    loc_detections = []
    if wic_confidence > wic_thresh:

        # Run Localizer
        outputs = loc.post(
            loc.predict(loc.pre(inputs, config=config)),
            loc_thresh=loc_thresh,
            nms_thresh=nms_thresh,
        )

        # Format and render results
        detects = outputs[0]
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
                loc_detections.append(f'{label}: {conf:0.04f}')
    loc_detections = '\n'.join(loc_detections)

    end = time.time()
    duration = end - start
    speed = f'{duration:0.02f} seconds)'

    return img, speed, wic_confidence, loc_detections


interface = gr.Interface(
    fn=predict,
    title='Wild Me Scout - Tile ML Demo',
    inputs=[
        gr.Image(type='filepath'),
        gr.Radio(
            label='Model Configuration',
            type='value',
            choices=[PHASE1[0], MVP[0]],
            value=MVP[0],
        ),
        gr.Slider(label='WIC Confidence Threshold', value=MVP[1]),
        gr.Slider(label='Localizer Confidence Threshold', value=MVP[2]),
        gr.Slider(label='Localizer NMS Threshold', value=MVP[3]),
    ],
    outputs=[
        gr.Image(type='numpy'),
        gr.Textbox(label='Prediction Speed', interactive=False),
        gr.Number(label='Predicted WIC Confidence', precision=5, interactive=False),
        gr.Textbox(label='Predicted Localizer Detections', interactive=False),
    ],
    examples=[
        # Phase 1
        ['examples/07a4b8db-f31c-261d-4580-e9402768fd45.true.jpg'] + PHASE1,
        ['examples/15e815d9-5aad-fa53-d1ed-33429020e15e.true.jpg'] + PHASE1,
        ['examples/1bb79811-3149-7a60-2d88-613dc3eeb261.true.jpg'] + PHASE1,
        ['examples/1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg'] + PHASE1,
        ['examples/201bc65e-d64e-80d3-2610-5865a22d04b4.false.jpg'] + PHASE1,
        ['examples/3affd8b6-9722-f2d5-9171-639615b4c38f.true.jpg'] + PHASE1,
        ['examples/4aedb818-f2f4-e462-8b75-5c8e34a01a59.false.jpg'] + PHASE1,
        ['examples/474bc2b6-dc51-c1b5-4612-efe810bbe091.true.jpg'] + PHASE1,
        ['examples/c3014107-3464-60b5-e04a-e4bfafdf8809.false.jpg'] + PHASE1,
        ['examples/f835ce33-292a-9116-794e-f8859b5956ec.true.jpg'] + PHASE1,
        # MVP
        ['examples/07a4b8db-f31c-261d-4580-e9402768fd45.true.jpg'] + MVP,
        ['examples/15e815d9-5aad-fa53-d1ed-33429020e15e.true.jpg'] + MVP,
        ['examples/1bb79811-3149-7a60-2d88-613dc3eeb261.true.jpg'] + MVP,
        ['examples/1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg'] + MVP,
        ['examples/201bc65e-d64e-80d3-2610-5865a22d04b4.false.jpg'] + MVP,
        ['examples/3affd8b6-9722-f2d5-9171-639615b4c38f.true.jpg'] + MVP,
        ['examples/4aedb818-f2f4-e462-8b75-5c8e34a01a59.false.jpg'] + MVP,
        ['examples/474bc2b6-dc51-c1b5-4612-efe810bbe091.true.jpg'] + MVP,
        ['examples/c3014107-3464-60b5-e04a-e4bfafdf8809.false.jpg'] + MVP,
        ['examples/f835ce33-292a-9116-794e-f8859b5956ec.true.jpg'] + MVP,
    ],
    cache_examples=True,
    allow_flagging='never',
)

interface.launch(server_name='0.0.0.0')
