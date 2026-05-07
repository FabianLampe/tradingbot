# CUDA-enabled image for FinBERT + (optional) local LLM on the user's GPUs.
# If you don't need GPU, swap to `python:3.11-slim` and remove --extra-index-url for torch.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Berlin \
    DEBIAN_FRONTEND=noninteractive

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl tzdata ca-certificates build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# python deps first for cache friendliness
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# project code
COPY . .

# data dirs that should survive restarts -> mounted volumes via compose
RUN mkdir -p /app/data /app/data/prices /app/data/news /app/data/macro /app/data/meta /app/data/models /app/data/journal /app/data/knowledge /app/data/edgar

EXPOSE 8501

# default: run nothing — compose chooses (dashboard | telegram | scheduler | shell)
CMD ["bash"]
