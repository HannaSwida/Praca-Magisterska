from reader import Reader
import torch


def main():
    reader = Reader("training-data/voxceleb")

    audio, speakers = reader.next()
    print(audio.shape, speakers.shape)


if __name__ == "__main__":
    main()
