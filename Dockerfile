FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

ENV GRADIO_SERVER_NAME=0.0.0.0

ENV GRADIO_SERVER_PORT=7860

# Install apt packages
RUN set -ex \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./ /code

RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install -e . \
 && pip3 uninstall -y onnxruntime \
 && pip3 install onnxruntime-gpu

CMD python3 app2.py
