import torch
from model import Model
from timit_loader import TimitLoaderDvector



torch.set_printoptions(profile="full")
#path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints/checkpoint_e20.pth.tar"
path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints_timit/checkpoint_e10.pth.tar"
with open("Dvectors.txt", "a") as o:
    #TRAIN

    voices_loader = TimitLoaderDvector('training-data/timit')
    o.write("\nD VECTORS FOR TRAINING")
    speaker_utts_arr = []
    name_arr = []
    for speaker_utts, speaker_name in voices_loader:
        speaker_utts_arr.append(speaker_utts)
        name_arr.append(speaker_name)
    x = torch.stack(speaker_utts_arr)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    checkpoint = torch.load(path)

    model = Model(168)
    print("len(voices_loader.speakers)", len(voices_loader.speakers))
    model.to("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        dvectors = model.generateDVec(x)
    o.write("\n")
    o.write(str(dvectors))

    #TEST

    test_voices_loader = TimitLoaderDvector('training-data/timit')
    o.write("\nD VECTORS FOR TRAINING")
    test_speaker_utts_arr = []
    test_name_arr = []
    for test_speaker_utts, test_speaker_name in test_voices_loader:
        test_speaker_utts_arr.append(speaker_utts)
        test_name_arr.append(speaker_name)
    test_x = torch.stack(test_speaker_utts_arr)

    model = Model(168)
    model.to("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        dvectors = model.generateDVec(x)
        test_dvectors = model.generateDVec(test_x)
    o.write("\n")
    o.write(str(dvectors))

    S = []
    spk_line = []
    for a in range(0, 100):
        spk_line = []
        for b in range(0, 100):
            spk_line.append(cos(test_dvectors[a].unsqueeze(0), dvectors[b].unsqueeze(0)))
        S.append(spk_line)

    for s in range(0, 10):
            print(S[s])
    #if S[s] >= 0.98: print(name_arr[s], ", ", test_name_arr[s], "\n")
    import seaborn as sns
    import matplotlib.pylab as plt

    plt.title('S dla speakers', fontsize=20)  # title with fontsize 20
    plt.ylabel('Speakers', fontsize=15)  # y-axis label with fontsize 15
    ax = sns.heatmap(S, yticklabels=name_arr[0:30], xticklabels=test_name_arr[0:30],cmap="Blues")
    #plt.show()