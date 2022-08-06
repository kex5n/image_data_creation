FROM python:3.10.6-slim-buster

COPY ./requirements.txt ./
RUN apt update && apt install -y vim
RUN pip install -U pip && pip install -r requirements.txt

WORKDIR /usr/src
