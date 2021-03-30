FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get update

RUN apt-get install -y libsndfile1

RUN pip install librosa matplotlib

RUN mkdir /home/Praca-Magisterska

ADD . /home/Praca-Magisterska

WORKDIR /home/Praca-Magisterska

CMD ./main.py

