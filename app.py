# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np  # NOQA
import torch
from PIL import Image, ImageOps  # NOQA
from torchvision.transforms import Compose, Resize, ToTensor

from scoutbot import model, utils

config = 'scoutbot/configs/mnist_resnet18.yaml'

log = utils.init_logging()
cfg = utils.init_config(config, log)
device = cfg.get('device')

cfg['output'] = 'scoutbot/{}'.format(cfg['output'])

net, _, _ = model.load(cfg)
net.eval()


def predict(inp):
    inp = ImageOps.grayscale(inp)

    transforms = Compose([Resize(cfg['image_size']), ToTensor()])
    inp = transforms(inp).unsqueeze(0)
    data = inp.to(device)

    with torch.no_grad():
        prediction = net(data)

    confidences = torch.softmax(prediction[0], dim=0).cpu().numpy()
    confidences = list(enumerate(confidences))
    confidences = [
        (
            str(label),
            float(conf),
        )
        for label, conf in confidences
    ]
    confidences = dict(confidences)

    return confidences


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=3),
    examples=[f'examples/example_{index}.jpg' for index in range(1, 31)],
)

interface.launch(server_name='0.0.0.0')
