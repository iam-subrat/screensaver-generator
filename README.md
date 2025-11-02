# Video Processing Lambda

AWS Lambda function for applying image overlays to videos using computer vision.

## Setup

### GitHub Secrets Required

Set these secrets in your GitHub repository:

- `AWS_ACCESS_KEY_ID`: AWS access key for deployment
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for deployment  
- `SAM_DEPLOYMENT_BUCKET`: S3 bucket for SAM deployment artifacts
- `S3_BUCKET_NAME`: S3 bucket for video processing files
- `WEBHOOK_BASE_URL`: Base URL for webhook notifications

### Required IAM Permissions

Add this permission to your IAM user policy:

```json
{
  "Effect": "Allow",
  "Action": [
    "lambda:GetLayerVersion"
  ],
  "Resource": "arn:aws:lambda:ap-southeast-1:770693421928:layer:Klayers-p39-opencv-python:*"
}
```

### Local Development

```bash
# Install development dependencies (includes OpenCV)
pip install -r requirements-dev.txt

# Build and test locally
sam build
sam local start-api
```

### Lambda Layers

This project uses AWS Lambda Layers for OpenCV to stay under the 250MB deployment limit:
- **OpenCV Layer**: `arn:aws:lambda:ap-southeast-1:770693421928:layer:Klayers-p39-opencv-python:1`
- **Production requirements**: Only boto3 and requests (lightweight)
- **Development requirements**: Full dependencies including OpenCV and Jupyter

### Deployment

Push to main branch to trigger automatic deployment via GitHub Actions.

## Usage

POST to `/process` endpoint with:

```json
{
  "video_s3_key": "videos/input.mp4",
  "image_s3_key": "images/overlay.png",
  "output_s3_key": "videos/output.mp4",
  "task_id": 123,
  "target_color": [255, 0, 0],
  "tolerance": 40,
  "min_area": 1000
}
```