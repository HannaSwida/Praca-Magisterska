import os
from os.path import exists
os.system("ffmpeg -i \"training-data/libsmall/1112/128138/1112-128138-0039.wav\"  -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-30dB \"training-data/libsmall/1112/128138/91118-19-009.wav\" ")
#os.system("ffmpeg -i \"training-data/libsmall/1112/128138/1112-128138-0039.wav\" -af 'silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse' \"training-data/libsmall/1112/128138/1112-128138-60039.wav\"")
