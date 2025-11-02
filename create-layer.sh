#!/bin/bash

# Create OpenCV Lambda Layer
mkdir -p layer/python
pip install opencv-python-headless==4.5.5.64 numpy==1.21.6 -t layer/python/

# Create layer zip
cd layer
zip -r opencv-layer.zip python/

# Deploy layer
aws lambda publish-layer-version \
    --layer-name opencv-python-layer \
    --zip-file fileb://opencv-layer.zip \
    --compatible-runtimes python3.9 \
    --region ap-southeast-1

# Clean up
cd ..
rm -rf layer/