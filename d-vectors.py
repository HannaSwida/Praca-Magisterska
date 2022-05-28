import torch
from model import Model
from timit_loader import TimitLoaderDvector

path = "C:/Users/hanna/POLIBUDA/MAGISTERKA/Magisterka_python/checkpoints/checkpoint_e20.pth.tar"

voices_loader = TimitLoaderDvector('training-data/timit')


x1, y1 = voices_loader[0]
x2, y2 = voices_loader[2]
x3, y3 = voices_loader[3]
x = torch.stack([x1,x2,x3])
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
checkpoint = torch.load(path)
model = Model(len(voices_loader.speakers))
model.to("cpu")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
with torch.no_grad():
    dvectors = model.generateDVec(x) # chcemy tensor b/speakers
s = cos(dvectors[0].unsqueeze(0), dvectors[1].unsqueeze(0))
print(s)