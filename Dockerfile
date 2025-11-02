FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY lambda_handler.py async_handler.py processor_handler.py video_processing.py send_event.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.lambda_handler"]