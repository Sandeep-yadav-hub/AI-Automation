FROM python:3.9.7-slim-buster
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update
RUN apt-get install -y libjpeg62-turbo-dev libtiff-dev libtesseract-dev tesseract-ocr 
RUN apt-get install libgl1 -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./index.py .