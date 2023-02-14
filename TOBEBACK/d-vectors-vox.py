import torch
from vox2.model import Model
from vox_celeb_loader import VoxLoaderDvector

#print(sys.path)
path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints_voxceleb/vox2/checkpoint_test17.pth.tar"
with open("voxdebug.txt", "Adam") as o:
    voices_loader = VoxLoaderDvector('testing-data/test')
    o.write("voices loader: \n ")
    o.write(str(len(voices_loader)))
    speaker_utts_arr = []
    name_arr = []
    for speaker_utts, speaker_name in voices_loader:
        speaker_utts_arr.append(speaker_utts)
        name_arr.append(speaker_name)
    x = torch.stack(speaker_utts_arr)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    checkpoint = torch.load(path,map_location=torch.device('cpu'))

    model = Model(5994)
    model.to("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        dvectors = model.generateDVec(x) # chcemy tensor b/speakers
    S = []
    spk_line = []
    speaker_utts_less =[]

    S = []
    spk_line = []
    speaker_utts_less =[]
    ba = 0
    ab = 0
    for a in range(0, (len(voices_loader))):
        spk_line = []
        for b in range(0, (len(voices_loader))):
            #spk_line.append(cos(dvectors[Adam].unsqueeze(0), dvectors[b].unsqueeze(0)))
            if (float(format(float(cos(dvectors[a].unsqueeze(0), dvectors[b].unsqueeze(0))),".10f")))> float(0.9900):
                spk_line.append(1)
            else: spk_line.append(0)
            #print(cos(dvectors[Adam].unsqueeze(0), dvectors[b].unsqueeze(0)))
        S.append(spk_line)
        speaker_utts_less.append(speaker_utts_arr[a])
    o.write("\ns \n")
    o.write(str(S))
    #print(spk_line)




    import seaborn as sns
    import matplotlib.pylab as plt

    plt.title('Heatmap of S', fontsize=20)  # title with fontsize 20

    plt.ylabel('Speakers', fontsize=15)  # y-axis label with fontsize 15
    plt.xlabel('Speakers', fontsize=15)  # y-axis label with fontsize 15
    (name_arr)
    ax = sns.heatmap(S, yticklabels=name_arr, xticklabels=name_arr, linewidth=0.4,cmap="Greens")
    plt.show()
#'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

