"""AWS Lambda handler for video processing."""

import json
import logging
import os
import tempfile
import time
import boto3
from botocore.exceptions import ClientError
from video_processing import apply_image_to_video_per_frame
from send_event import send_webhook_event

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


async def process_video_async(
    video_key,
    image_key,
    output_key,
    target_color,
    tolerance,
    min_area,
    task_id,
    s3_bucket,
    s3,
    start_time,
):
    """Background video processing function"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video and image from S3
            video_ext = os.path.splitext(video_key)[1] or ".mp4"
            image_ext = os.path.splitext(image_key)[1] or ".png"
            output_ext = os.path.splitext(output_key)[1] or ".mp4"

            video_path = os.path.join(temp_dir, f"input_video{video_ext}")
            image_path = os.path.join(temp_dir, f"overlay_image{image_ext}")
            output_path = os.path.join(temp_dir, f"output_video{output_ext}")

            logger.info(f"Task {task_id}: Downloading files from S3 bucket {s3_bucket}")
            s3.download_file(s3_bucket, video_key, video_path)
            s3.download_file(s3_bucket, image_key, image_path)
            logger.info(f"Task {task_id}: Files downloaded successfully")

        # Process video
        logger.info(f"Task {task_id}: Starting video processing")
        process_start = time.time()
        apply_image_to_video_per_frame(
            video_path=video_path,
            image_path=image_path,
            target_color=target_color,
            out_path=output_path,
            tolerance=tolerance,
            min_area=min_area,
        )
        process_time = time.time() - process_start
        logger.info(
            f"Task {task_id}: Video processing completed in {process_time:.2f}s"
        )

        # Upload result to S3
        logger.info(f"Task {task_id}: Uploading result to {output_key}")
        s3.upload_file(output_path, s3_bucket, output_key)
        logger.info(f"Task {task_id}: Upload completed")

        # Send webhook notification
        webhook_url = os.environ.get("WEBHOOK_BASE_URL", "http://localhost:8080")
        logger.info(f"Task {task_id}: Sending webhook to {webhook_url}")
        try:
            send_webhook_event(task_id=task_id, s3_key=output_key, base_url=webhook_url)
            logger.info(f"Task {task_id}: Webhook sent successfully")
        except Exception as webhook_error:
            logger.warning(f"Task {task_id}: Webhook failed: {str(webhook_error)}")

        total_time = time.time() - start_time
        logger.info(
            f"Task {task_id}: Processing completed successfully in {total_time:.2f}s"
        )
    except Exception as e:
        logger.error(f"Task {task_id}: Background processing failed: {str(e)}")


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
    logger.info(f"Processing request {context.aws_request_id}")

    try:
        # Parse JSON body from API Gateway
        if "body" in event:
            body = (
                json.loads(event["body"])
                if isinstance(event["body"], str)
                else event["body"]
            )
        else:
            body = event

        logger.info(
            f"Parsed request body for task_id: {body.get('task_id', 'unknown')}"
        )

        # Extract parameters
        s3_bucket = os.environ.get("S3_BUCKET_NAME", "default-screensaver-assets")
        video_key = body["video_s3_key"]
        image_key = body["image_s3_key"]
        output_key = body["output_s3_key"]
        task_id = body["task_id"]
        target_color = tuple(body.get("target_color", [255, 0, 0]))
        tolerance = body.get("tolerance", 40)
        min_area = body.get("min_area", 1000)

        logger.info(
            f"Task {task_id}: Processing video={video_key}, image={image_key}, output={output_key}"
        )

        # Invoke processing Lambda asynchronously
        lambda_client = boto3.client("lambda")
        payload = {
            "video_s3_key": video_key,
            "image_s3_key": image_key,
            "output_s3_key": output_key,
            "task_id": task_id,
            "target_color": target_color,
            "tolerance": tolerance,
            "min_area": min_area,
            "s3_bucket": s3_bucket,
        }

        lambda_client.invoke(
            FunctionName=os.environ.get(
                "PROCESSOR_FUNCTION_NAME",
                "video-processing-stack-VideoProcessorFunction",
            ),
            InvocationType="Event",  # Async invocation
            Payload=json.dumps(payload),
        )

        logger.info(f"Task {task_id}: Background processing Lambda invoked")

        # Return immediate response
        return {
            "statusCode": 202,
            "body": json.dumps(
                {
                    "message": "Video processing started",
                    "task_id": task_id,
                    "request_id": context.aws_request_id,
                }
            ),
        }

    except ClientError as e:
        error_msg = f"S3 error: {str(e)}"
        logger.error(f"S3 error for request {context.aws_request_id}: {error_msg}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": error_msg, "request_id": context.aws_request_id}
            ),
        }
    except KeyError as e:
        error_msg = f"Missing required parameter: {str(e)}"
        logger.error(
            f"Parameter error for request {context.aws_request_id}: {error_msg}"
        )
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"error": error_msg, "request_id": context.aws_request_id}
            ),
        }
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(
            f"Unexpected error for request {context.aws_request_id}: {error_msg}"
        )
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": error_msg, "request_id": context.aws_request_id}
            ),
        }
