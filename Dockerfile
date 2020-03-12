FROM python:3.8-alpine

WORKDIR /usr/src/KoGPT2

COPY requirements.txt ./

# gluonnlp build dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ["pip", "install", "."]
