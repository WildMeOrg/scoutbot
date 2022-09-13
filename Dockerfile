FROM continuumio/anaconda3:latest

ENV GRADIO_SERVER_NAME=0.0.0.0

ENV GRADIO_SERVER_PORT=7860

WORKDIR /code

COPY ./ /code

RUN conda install pip \
 && pip install --no-cache-dir -r requirements.txt

CMD python app.py
