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

Your IAM user needs these permissions for deployment:

```json
{
  "Effect": "Allow",
  "Action": [
    "cloudformation:*",
    "s3:*",
    "lambda:*",
    "iam:*",
    "apigateway:*",
    "logs:*"
  ],
  "Resource": "*"
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

### Deployment

This project uses Docker containers for Lambda deployment to include OpenCV without size limits.

#### Automated (GitHub Actions)
Push to main branch to trigger automatic deployment.

#### Manual Deployment
```bash
sam build
sam deploy --guided
```

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