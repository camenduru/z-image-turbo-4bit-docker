FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libegl1 libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev cmake && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg xvfb && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

USER camenduru

RUN pip install --no-cache-dir torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 --extra-index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir xformers==0.0.32.post2 --extra-index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir git+https://github.com/huggingface/diffusers git+https://github.com/huggingface/accelerate
RUN pip install --no-cache-dir sdnq runpod boto3
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/scheduler/scheduler_config.json -d /content/z-image-4bit/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/text_encoder/config.json -d /content/z-image-4bit/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/text_encoder/generation_config.json -d /content/z-image-4bit/text_encoder -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/resolve/main/text_encoder/model.safetensors -d /content/z-image-4bit/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/text_encoder/quantization_config.json -d /content/z-image-4bit/text_encoder -o quantization_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/tokenizer/merges.txt -d /content/z-image-4bit/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/resolve/main/tokenizer/tokenizer.json -d /content/z-image-4bit/tokenizer -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/tokenizer/tokenizer_config.json -d /content/z-image-4bit/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/tokenizer/vocab.json -d /content/z-image-4bit/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/transformer/config.json -d /content/z-image-4bit/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/resolve/main/transformer/diffusion_pytorch_model.safetensors -d /content/z-image-4bit/transformer -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/transformer/quantization_config.json -d /content/z-image-4bit/transformer -o quantization_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/vae/config.json -d /content/z-image-4bit/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/z-image-4bit/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32/raw/main/model_index.json -d /content/z-image-4bit -o model_index.json

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py