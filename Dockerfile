FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and core build tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Fix python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install High-Performance Training Stack (Stable PyPI versions)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir unsloth
RUN pip install --no-cache-dir trl>=0.12.0 peft accelerate bitsandbytes unsloth_zoo
RUN pip install --no-cache-dir gymnasium networkx numpy datasets huggingface_hub python-dotenv

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Run the training script with -u for unbuffered logging
CMD ["python", "-u", "scripts/train_grpo_colab.py"]
