FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    CUDA_HOME=/usr/local/cuda

WORKDIR $HOME/app

USER root
RUN apt-get update && apt-get install -y git curl build-essential && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user . $HOME/app

RUN pip install --upgrade pip
# Force strict synchronization of PyTorch and Torchvision directly from NVIDIA's servers
RUN pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121
# Install all required modules in one robust resolution block
RUN pip install "unsloth" "xformers==0.0.27" "trl" "peft" "accelerate" "bitsandbytes" "gymnasium" "networkx" "scipy" "datasets" "gradio" "huggingface_hub"

CMD ["python", "app.py"]
