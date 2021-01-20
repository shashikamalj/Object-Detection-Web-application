FROM python:3

WORKDIR /usr/src/app

COPY client.py .
COPY server.py .
COPY __init__.py .
COPY image_manip.py .
COPY perform_operations.py .

RUN mkdir ./model/
COPY ./model/extract_coods.py ./model/

RUN mkdir ./res/
COPY ./res/coco.names ./res/
COPY ./res/yolov3.cfg ./res/
COPY ./res/yolov3.weights ./res/

RUN mkdir ./templates/
COPY ./templates/index.html ./templates/

RUN mkdir ./static/
COPY ./static/upload.js ./static/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","server.py"]