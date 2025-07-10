FROM python:3

WORKDIR /usr/src/app

COPY requirements.local.txt .

RUN pip install --no-cache-dir -r requirements.local.txt
