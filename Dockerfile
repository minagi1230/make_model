FROM ubuntu:16.04

WORKDIR /bin/src/app

COPY make_model.py ./
COPY use_model.py ./

RUN apt update
RUN apt install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install Image
RUN pip3 install tqdm
RUN pip3 install keras
RUN pip3 install tensorflow
RUN pip3 install sklearn
RUN pip3 install matplotlib
RUN pip3 install optuna
