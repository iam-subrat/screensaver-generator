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

### Lambda Layers

This project uses a custom AWS Lambda Layer for OpenCV to stay under the 250MB deployment limit:
- **Custom OpenCV Layer**: Created from your S3 bucket during deployment
- **Production requirements**: Only boto3 and requests (lightweight)
- **Development requirements**: Full dependencies including OpenCV and Jupyter

### Deployment

#### Option 1: Automated (GitHub Actions)
Push to main branch to trigger automatic deployment.

#### Option 2: Manual with Custom Layer
```bash
# Set your S3 bucket name
export S3_BUCKET_NAME=your-bucket-name

# Deploy with custom OpenCV layer
./deploy-with-layer.sh
```

#### Option 3: Manual Layer Creation
```bash
# Create layer manually
./create-layer.sh

# Then deploy normally
sam build
sam deploy
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