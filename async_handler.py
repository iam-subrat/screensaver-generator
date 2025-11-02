"""Handler wrapper."""

from lambda_handler import lambda_handler as main_handler


def lambda_handler(event, context):
    """Main Lambda handler."""
    return main_handler(event, context)
