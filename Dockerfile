FROM pytorch

RUN apt-get update

RUN pip install librosa matplotlib

RUN mkdir /home/Praca-Magisterska

ADD . /home/Praca-Magisterska

WORKDIR /home/Praca-Magisterska

CMD ./main.py

