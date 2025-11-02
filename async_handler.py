"""Async wrapper for Lambda handler."""

import asyncio
from lambda_handler import lambda_handler as async_lambda_handler


def lambda_handler(event, context):
    """Sync wrapper for async Lambda handler."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_lambda_handler(event, context))
    finally:
        loop.close()
