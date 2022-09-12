# -*- coding: utf-8 -*-
import torch
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Resize, ToTensor


def test_architecture_params(net):
    total_params = sum(params.numel() for params in net.parameters())
    assert total_params == 133578


def test_model_prediction(cfg, device, net):
    image = Image.open('examples/example_1.jpg')

    image = ImageOps.grayscale(image)

    transforms = Compose([Resize(cfg['image_size']), ToTensor()])
    image = transforms(image).unsqueeze(0)
    data = image.to(device)

    with torch.no_grad():
        prediction = net(data)

    prediction = torch.argmax(prediction[0], dim=0).item()
    assert prediction == 5
