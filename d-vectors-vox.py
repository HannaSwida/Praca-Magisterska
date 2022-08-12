import torch
from model import Model
from vox_celeb_loader import VoxLoaderDvector
import sys
print(sys.path)
path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints_voxceleb/checkpoint_test89.pth.tar"
with open("voxdebug.txt", "a") as o:
    voices_loader = VoxLoaderDvector('training-data/voxceleb')
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
    S = []
    spk_line = []
    speaker_utts_less =[]


    voices_loader = VoxLoaderDvector('training-data/voxceleb')
    o.write("voices loader: \n ")
    o.write(str(len(voices_loader)))
    speaker_utts_arr1 = []
    name_arr1 = []
    for speaker_utts, speaker_name in voices_loader:
        speaker_utts_arr1.append(speaker_utts)
        name_arr1.append(speaker_name)
    x2 = torch.stack(speaker_utts_arr1)
    cos2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    checkpoint1 = torch.load(path)

    model1 = Model(len(voices_loader.speakers))
    model1.to("cpu")
    model1.load_state_dict(checkpoint1['state_dict'])
    model1.eval()
    with torch.no_grad():
        dvectors2 = model1.generateDVec(x2) # chcemy tensor b/speakers
    S = []
    spk_line = []
    print(len(dvectors))
    speaker_utts_less =[]
    print("dvec1",dvectors[22] )
    print("dvec2",dvectors2[22].unsqueeze(0) )
    print("HEY", (cos2(dvectors[30].unsqueeze(0), dvectors[1].unsqueeze(0))))
    ba = 0
    ab = 0
    for a in range(0, (len(voices_loader))):
        spk_line = []
        for b in range(0, (len(voices_loader))):
            spk_line.append(cos(dvectors[a].unsqueeze(0), dvectors2[b].unsqueeze(0)))
            print(cos(dvectors[a].unsqueeze(0), dvectors[b].unsqueeze(0)))
        S.append(spk_line)
        speaker_utts_less.append(speaker_utts_arr[a])
    o.write("\ns \n")
    o.write(str(S))

    print(dvectors[1].unsqueeze(0), dvectors2[1].unsqueeze(0))


    import seaborn as sns
    import matplotlib.pylab as plt

    plt.title('Heatmap of S', fontsize=20)  # title with fontsize 20

    plt.ylabel('Speakers', fontsize=15)  # y-axis label with fontsize 15
    (name_arr)
    ax = sns.heatmap(S, yticklabels=name_arr, linewidth=0.4,cmap="Blues")
    plt.show()