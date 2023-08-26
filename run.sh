#!/bin/bash
mkdir -p ./data
# -e TRANSFORMERS_CACHE=/custom_cache \
# --network host \
sudo docker run -it --rm \
    --gpus device=0 \
    -v $(pwd)/app.py:/app/app.py \
    -v $(pwd)/data:/app/data \
    embeddings
