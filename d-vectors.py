import torch
from model import Model
from timit_loader import TimitLoaderDvector

#path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints/checkpoint_e20.pth.tar"
path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints_timit_old/checkpoint_e43.pth.tar"
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
    print("len(voices_loader.speakers)", len(voices_loader.speakers))
    model.to("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        dvectors = model.generateDVec(x)
        o.write("\nDVectors: \n")
        o.write(str(dvectors))
        o.write("\nDVectors size: \n")
        o.write(str(dvectors.size()))
    S = []
    spk_line = []
    for a in range(5, 100):
        spk_line = []
        for b in range(5, 100):
            spk_line.append(cos(dvectors[a].unsqueeze(0), dvectors[b].unsqueeze(0)))
        S.append(spk_line)
    print("cos(dvectors[0].unsqueeze(0), dvectors[0].unsqueeze(0))", cos(dvectors[0].unsqueeze(0), dvectors[0].unsqueeze(0)))
    print("DVEC len:", len(dvectors[0].unsqueeze(0)[0]))
    print("DVEC len:", len(dvectors[0]))
    print("voices_loader[0].shape", len(voices_loader[0]))

    import seaborn as sns
    import matplotlib.pylab as plt

    plt.title('S dla 50 speakerów', fontsize=20)  # title with fontsize 20

    plt.ylabel('Speakers', fontsize=15)  # y-axis label with fontsize 15
    print(len(name_arr))
    ax = sns.heatmap(S, yticklabels=name_arr[50:100], xticklabels=name_arr[50:100], linewidth=0.4,cmap="Blues")
   # plt.show()
