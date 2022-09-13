# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np
import cv2

from scoutbot import wic, loc


def predict(filepath, wic_thresh, loc_thresh, nms_thresh):
    # Load data
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = [filepath]

    wic_thresh /= 100.0
    loc_thresh /= 100.0
    nms_thresh /= 100.0

    # Run WIC
    outputs = wic.post(wic.predict(wic.pre(inputs)))
    output = outputs[0]

    # Get WIC confidence
    wic_confidence = output.get('positive')

    # Run Localizer

    loc_detections = []
    if wic_confidence > wic_thresh:
        data, sizes = loc.pre(inputs)
        preds = loc.predict(data)
        outputs = loc.post(preds, sizes, loc_thresh=loc_thresh, nms_thresh=nms_thresh)
        detects = outputs[0]

        for detect in detects:
            if detect.confidence >= loc_thresh:
                point1 = (
                    int(np.around(detect.x_top_left)),
                    int(np.around(detect.y_top_left)),
                )
                point2 = (
                    int(np.around(detect.x_top_left + detect.width)),
                    int(np.around(detect.y_top_left + detect.height)),
                )
                color = (255, 0, 0)
                img = cv2.rectangle(img, point1, point2, color, 2)
                loc_detections.append(
                    f'{detect.class_label}: {detect.confidence:0.05f}'
                )
    loc_detections = '\n'.join(loc_detections)

    return img, wic_confidence, loc_detections


interface = gr.Interface(
    fn=predict,
    title='Scout Demo',
    inputs=[
        gr.Image(type='filepath'),
        gr.Slider(label='WIC Confidence Threshold', value=20),
        gr.Slider(label='Localizer Confidence Threshold', value=48),
        gr.Slider(label='Localizer NMS Threshold', value=20),
    ],
    outputs=[
        gr.Image(type='numpy'),
        gr.Number(label='Predicted WIC Confidence', precision=5, interactive=False),
        gr.Textbox(label='Predicted Localizer Detections', interactive=False),
    ],
    examples=[
        ['examples/07a4b8db-f31c-261d-4580-e9402768fd45.true.jpg', 20, 48, 20],
        ['examples/15e815d9-5aad-fa53-d1ed-33429020e15e.true.jpg', 10, 48, 20],
        ['examples/1bb79811-3149-7a60-2d88-613dc3eeb261.true.jpg', 20, 48, 20],
        ['examples/1e8372e4-357d-26e6-d7fd-0e0ae402463a.true.jpg', 20, 48, 20],
        ['examples/201bc65e-d64e-80d3-2610-5865a22d04b4.false.jpg', 20, 48, 20],
        ['examples/3affd8b6-9722-f2d5-9171-639615b4c38f.true.jpg', 20, 48, 20],
        ['examples/4aedb818-f2f4-e462-8b75-5c8e34a01a59.false.jpg', 20, 48, 20],
        ['examples/474bc2b6-dc51-c1b5-4612-efe810bbe091.true.jpg', 20, 48, 20],
        ['examples/c3014107-3464-60b5-e04a-e4bfafdf8809.false.jpg', 20, 48, 20],
        ['examples/f835ce33-292a-9116-794e-f8859b5956ec.true.jpg', 20, 48, 20],
    ],
    cache_examples=True,
    allow_flagging='never',
)

interface.launch(server_name='0.0.0.0')
