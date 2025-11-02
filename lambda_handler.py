"""AWS Lambda handler for video processing."""

import json
import os
import tempfile
import boto3
from botocore.exceptions import ClientError
from video_processing import apply_image_to_video_per_frame
from send_event import send_webhook_event


def lambda_handler(event, context):
    """
    Lambda handler for processing video with image overlay.

    Expected event structure:
    {
        "video_s3_key": "path/to/video.mp4",
        "image_s3_key": "path/to/image.png",
        "output_s3_key": "path/to/output.mp4",
        "task_id": 123,
        "target_color": [255, 0, 0],
        "tolerance": 40,
        "min_area": 1000
    }
    """
    try:
        s3 = boto3.client("s3")

        # Parse JSON body from API Gateway
        if "body" in event:
            body = (
                json.loads(event["body"])
                if isinstance(event["body"], str)
                else event["body"]
            )
        else:
            body = event

        # Extract parameters
        s3_bucket = os.environ.get("S3_BUCKET_NAME", "default-screensaver-assets")
        video_key = body["video_s3_key"]
        image_key = body["image_s3_key"]
        output_key = body["output_s3_key"]
        task_id = body["task_id"]
        target_color = tuple(body.get("target_color", [255, 0, 0]))
        tolerance = body.get("tolerance", 40)
        min_area = body.get("min_area", 1000)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video and image from S3
            video_ext = os.path.splitext(video_key)[1] or ".mp4"
            image_ext = os.path.splitext(image_key)[1] or ".png"
            output_ext = os.path.splitext(output_key)[1] or ".mp4"

            video_path = os.path.join(temp_dir, f"input_video{video_ext}")
            image_path = os.path.join(temp_dir, f"overlay_image{image_ext}")
            output_path = os.path.join(temp_dir, f"output_video{output_ext}")

            s3.download_file(s3_bucket, video_key, video_path)
            s3.download_file(s3_bucket, image_key, image_path)

            # Process video
            apply_image_to_video_per_frame(
                video_path=video_path,
                image_path=image_path,
                target_color=target_color,
                out_path=output_path,
                tolerance=tolerance,
                min_area=min_area,
            )

            # Upload result to S3
            s3.upload_file(output_path, s3_bucket, output_key)

            # Send webhook notification
            webhook_url = os.environ.get("WEBHOOK_BASE_URL", "http://localhost:8080")
            send_webhook_event(task_id=task_id, s3_key=output_key, base_url=webhook_url)

            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": "Video processed successfully",
                        "output_s3_key": output_key,
                        "task_id": task_id,
                        "request_id": context.aws_request_id,
                    }
                ),
            }

    except ClientError as e:
        error_msg = f"S3 error: {str(e)}"
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": error_msg, "request_id": context.aws_request_id}
            ),
        }
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": error_msg, "request_id": context.aws_request_id}
            ),
        }
