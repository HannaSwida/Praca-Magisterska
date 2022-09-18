import os
from os.path import exists

for person in os.listdir("vox2/dev/aac"):
    for dir in os.listdir("vox2/dev/aac/{}".format(person)):
        for file in os.listdir("vox2/dev/aac/{}/{}".format(person,dir)):
            if (file.endswith(".m4a") and not exists("{}.wav".format(file))): #or .avi, .mpeg, whatever.
                print("vox2/dev/aac/{}/{}/{}".format(person, dir, file))
                os.system("ffmpeg -n -i vox2/dev/aac/{}/{}/{} vox2/dev/aac/{}/{}/{}.wav -ar 16000".format(person,dir,file,person,dir,file[:-4]))
            else:
                continue

