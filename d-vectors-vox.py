import torch
from model import Model
from vox_celeb_loader import VoxLoaderDvector

path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints_voxceleb/checkpoint_e0.pth.tar"
with open("voxdebug.txt", "a") as o:
    voices_loader = VoxLoaderDvector('training-data/voxceleb')
    o.write("voices loader: \n ")
    o.write(str(len(voices_loader)))
    speaker_utts_arr = []
    name_arr = []
    for speaker_utts, speaker_name in voices_loader:
        speaker_utts_arr.append(speaker_utts)
#       o.write("\n")
#       o.write(str(speaker_name))
#       o.write("speaker_utts:\n")
#       o.write(str(speaker_utts))
        name_arr.append(speaker_name)
#   x1, y1 = voices_loader[0]
#   o.write("\n y1: \n")
#   o.write(str(y1))
#   x2, y2 = voices_loader[2]
#   o.write("\n y2: \n")
#   o.write(str(y2))
#   x3, y3 = voices_loader[3]
#   x = torch.stack([x1,x2,x3])
#   o.write("\nx:\n")
#   o.write(str(x))
    x = torch.stack(speaker_utts_arr)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    checkpoint = torch.load(path)

    model = Model(len(voices_loader.speakers))
    model.to("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        dvectors = model.generateDVec(x) # chcemy tensor b/speakers
#       o.write("\nDVectors: \n")
#       o.write(str(dvectors))
#       o.write("\nDVectors size: \n")
#       o.write(str(dvectors.size()))
    s = cos(dvectors[0].unsqueeze(0), dvectors[1].unsqueeze(0))
    o.write("\ns \n")
    o.write(str(s))
#   o.write("\ndvectors[0].unsqueeze(0).size(): \n")
#   o.write(str(dvectors[0].unsqueeze(0).size()))
#   o.write("\ndvectors[0].unsqueeze(0): \n")
#   o.write(str(dvectors[0].unsqueeze(0)))
#   o.write("\ndvectors[1].unsqueeze(0).size(): \n")
#   o.write(str(dvectors[1].unsqueeze(0).size()))
#   o.write("\ndvectors[1].unsqueeze(0): \n")
#   o.write(str(dvectors[1].unsqueeze(0)))
#   o.write("\nname0: \n")
#   o.write(name_arr[0])
#   o.write("\nname1: \n")
#   o.write(name_arr[1])
#   x1, y1 = voices_loader[0]
#   o.write("\n y1: \n")
#   o.write(str(y1))
