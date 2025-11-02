"""Background video processor Lambda handler."""

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


def lambda_handler(event, context):
    """Background video processing handler."""
    start_time = time.time()
    logger.info(f"Processing background task {context.aws_request_id}")
    
    try:
        s3 = boto3.client("s3")
        
        # Extract parameters from event
        video_key = event["video_s3_key"]
        image_key = event["image_s3_key"]
        output_key = event["output_s3_key"]
        task_id = event["task_id"]
        target_color = tuple(event["target_color"])
        tolerance = event["tolerance"]
        min_area = event["min_area"]
        s3_bucket = event["s3_bucket"]
        
        logger.info(f"Task {task_id}: Processing video={video_key}, image={image_key}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video and image from S3
            video_ext = os.path.splitext(video_key)[1] or ".mp4"
            image_ext = os.path.splitext(image_key)[1] or ".png"
            output_ext = os.path.splitext(output_key)[1] or ".mp4"

            video_path = os.path.join(temp_dir, f"input_video{video_ext}")
            image_path = os.path.join(temp_dir, f"overlay_image{image_ext}")
            output_path = os.path.join(temp_dir, f"output_video{output_ext}")

            logger.info(f"Task {task_id}: Downloading files from S3")
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
            logger.info(f"Task {task_id}: Video processing completed in {process_time:.2f}s")

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
        logger.info(f"Task {task_id}: Processing completed in {total_time:.2f}s")
        
        return {"statusCode": 200, "body": json.dumps({"message": "Processing completed"})}
        
    except Exception as e:
        logger.error(f"Task {task_id}: Processing failed: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}