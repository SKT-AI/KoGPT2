FROM python:3.8-slim

WORKDIR /usr/src/KoGPT2

# gluonnlp build dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ["pip", "install", "."]
