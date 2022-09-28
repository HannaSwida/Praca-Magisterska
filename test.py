import os
from os.path import exists
a = 9999999
b=9999.1
for person in os.listdir("training-data/Libri"):
    for dir in os.listdir("training-data/Libri/{}".format(person)):
        for file in os.listdir("training-data/Libri/{}/{}".format(person,dir)):
            if (file.endswith(".wav")): #or .avi, .mpeg, whatever.
                os.system("ffmpeg -v error -i training-data/Libri/{}/{}/{}.wav -f null >logs/err{}.log 2>>&1".format(person,dir,file[:-4],file[:-4]))
print(a)

