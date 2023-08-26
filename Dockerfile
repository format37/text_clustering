# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
# FROM huggingface/transformers-pytorch-gpu:latest
FROM huggingface/transformers-pytorch-gpu:4.29.2

WORKDIR /app

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run app
CMD ["python3", "./app.py"]