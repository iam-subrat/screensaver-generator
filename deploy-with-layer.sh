#!/bin/bash

set -e

if [ -z "$S3_BUCKET_NAME" ]; then
    echo "Error: S3_BUCKET_NAME environment variable not set"
    exit 1
fi

echo "Creating OpenCV Lambda Layer..."

# Create layer directory
mkdir -p layer/python

# Install dependencies to layer
pip3 install opencv-python-headless==4.5.5.64 numpy==1.21.6 -t layer/python/

# Create layer zip
cd layer
zip -r opencv-layer.zip python/
cd ..

# Upload layer to S3
aws s3 cp layer/opencv-layer.zip s3://$S3_BUCKET_NAME/layers/opencv-layer.zip

echo "Layer uploaded to S3. Now deploying SAM application..."

# Deploy SAM application
sam build
sam deploy --no-confirm-changeset --no-fail-on-empty-changeset

# Clean up
rm -rf layer/

echo "Deployment complete!"