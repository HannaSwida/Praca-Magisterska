import os
import shutil
from os.path import exists
from pathlib import Path

a = 9999999
b=9999.1
for person in os.listdir("training-data/Libri"):
    for dir in os.listdir("training-data/Libri/{}".format(person)):
        isFlac = True
        for file in os.listdir("training-data/Libri/{}/{}".format(person,dir)):
            [os.removedirs(p) for p in Path(target_path).glob('**/*') if p.is_dir() and len(list(p.iterdir())) == 0]

print(a)

a = 9999999
b=9999.1
[os.removedirs(p) for p in Path("training-data/Libri/").glob('**/*') if p.is_dir() and len(list(p.iterdir())) == 0]

print(a)

