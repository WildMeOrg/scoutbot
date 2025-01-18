FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
#  Added Rust's bin directory to PATH
ENV PATH="/root/.cargo/bin:$PATH" 

# Install apt packages
# Added curl for Rust installation
# Added build tools for compiling extensions
RUN set -ex \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        curl \ 
        build-essential \  
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain as the docker build is failing due to missing Rust Tool Chain
# Verify Rust installation
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
 && rustc --version 

WORKDIR /code

# Copy project files
COPY ./ /code

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install -e . \
 && pip3 uninstall -y onnxruntime \
 && pip3 install --no-cache-dir onnxruntime-gpu

# Default command
CMD python3 app2.py
