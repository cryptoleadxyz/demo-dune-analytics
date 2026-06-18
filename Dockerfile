FROM python:3.13-slim
WORKDIR /code
COPY requirements.txt .
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install -y git
RUN pip install --upgrade pip
RUN pip install black
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/workspaces/code/"
COPY . .
