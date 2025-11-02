# Lambda Layer Setup Instructions

## Step 1: Upload the Layer to S3

The layer zip file has been created at `layer/opencv-layer.zip` (40MB).

Upload it to S3 using AWS Console or CLI:

```bash
aws s3 cp layer/opencv-layer.zip s3://{YOUR_BUCKET}/layers/opencv-layer.zip
```

Or use the AWS Console:
1. Go to S3 bucket
2. Create folder: `layers`
3. Upload file: `layer/opencv-layer.zip`

## Step 2: Deploy SAM Application

After uploading the layer, run:

```bash
sam build
sam deploy
```

The deployment will now succeed because the layer file exists in S3.

## Cleanup

After successful deployment, you can delete the local layer folder:

```bash
rm -rf layer/
```