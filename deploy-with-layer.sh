#!/bin/bash

echo "Creating OpenCV Lambda Layer..."

# Create layer directory
mkdir -p layer/python

# Install dependencies to layer
pip install opencv-python-headless==4.5.5.64 numpy==1.21.6 -t layer/python/

# Create layer zip
cd layer
zip -r opencv-layer.zip python/
cd ..

# Upload layer to S3
aws s3 cp layer/opencv-layer.zip s3://$S3_BUCKET_NAME/layers/opencv-layer.zip

echo "Layer uploaded to S3. Now deploying SAM application..."

# Deploy SAM application
sam build
sam deploy

# Clean up
rm -rf layer/

echo "Deployment complete!"