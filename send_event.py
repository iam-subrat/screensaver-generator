import requests
from typing import Optional


def send_webhook_event(
    task_id: int,
    s3_key: str,
    event_type: str = "processed",
    base_url: str = "http://localhost:8080",
    additional_data: Optional[dict] = None,
) -> dict:
    """
    Send a webhook event to the screensaver ad backend.

    Args:
        task_id: The ID of the task
        s3_key: The S3 key for the processed file
        event_type: Type of event (default: "processed")
        base_url: Base URL of the API (default: "http://localhost:8080")
        additional_data: Optional additional data to include in payload

    Returns:
        Response JSON as dictionary

    Raises:
        requests.RequestException: If the request fails
    """
    url = f"{base_url}/api/webhook"

    # Build payload
    payload = {"task_id": task_id, "s3_key": s3_key}

    # Add any additional data
    if additional_data:
        payload.update(additional_data)

    # Build request body
    data = {"event_type": event_type, "payload": payload}

    # Make request
    response = requests.post(
        url, json=data, headers={"Content-Type": "application/json"}
    )

    # Raise exception for bad status codes
    response.raise_for_status()

    return response.json()


# Example usage:
if __name__ == "__main__":
    try:
        # Basic usage
        result = send_webhook_event(
            task_id=123, s3_key="output/processed_video_123.mp4"
        )
        print("Success:", result)

        # With additional data
        result = send_webhook_event(
            task_id=456,
            s3_key="output/processed_video_456.mp4",
            additional_data={
                "processing_time": 45.2,
                "file_size": 1024000,
                "quality": "HD",
            },
        )
        print("Success with extra data:", result)

    except requests.RequestException as e:
        print(f"Error: {e}")
