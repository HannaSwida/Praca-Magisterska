import torch
from model import Model
from timit_loader import TimitLoaderDvector

path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints/checkpoint_e20.pth.tar"
#path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints_timit/checkpoint_e20.pth.tar"
with open("randomfile.txt", "a") as o:
    voices_loader = TimitLoaderDvector('training-data/timit')
    o.write("voices loader: \n ")
    o.write(str(len(voices_loader)))
    speaker_utts_arr = []
    name_arr = []
    for speaker_utts, speaker_name in voices_loader:
        speaker_utts_arr.append(speaker_utts)
        name_arr.append(speaker_name)
    x = torch.stack(speaker_utts_arr)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    checkpoint = torch.load(path)

    model = Model(len(voices_loader.speakers))
    model.to("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        dvectors = model.generateDVec(x) # chcemy tensor b/speakers
        o.write("\nDVectors: \n")
        o.write(str(dvectors))
        o.write("\nDVectors size: \n")
        o.write(str(dvectors.size()))
    s = cos(dvectors[0].unsqueeze(0), dvectors[1].unsqueeze(0))
    o.write("\ns \n")
    o.write(str(s))
