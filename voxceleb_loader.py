import os
import random
from torch.utils.data import Dataset
import numpy as np


class VCset(Dataset):

    def __init__(self, input_dirs_file, rate=16000, chunk_size=0.4, n_batches=100, batch_size=128) -> None:
        super().__init__()
        self.n_labels = None

        with open(input_dirs_file, 'r') as f:
            self.input_dirs = f.read().splitlines()
        
        self.classes_dict={}
        for i, d in enumerate(self.input_dirs):
            self.classes_dict.update(self.classes_dict_from_subdirs(d, i))

        self.n_labels = len(self.classes_dict)
        
        print(self.n_labels , "c, l:", self.n_labels)
        self.rate = rate
        self.chunk_size = chunk_size
        self.n_batches = n_batches
        self.batch_size = batch_size

    def classes_dict_from_subdirs(self, input_dir, id, ext=".wav"):
        
        classes_dict={}

        for root, dirs, files in os.walk(input_dir):
            for name in files:
                path = os.path.join(root,name)
                if path.lower().endswith(ext):
                    class_name = os.path.dirname(path).split("/")[-1] + "#" + str(id) 
                    if class_name not in classes_dict:
                        classes_dict.update({class_name:{}})
                        
                    classes_dict[class_name] = path

        return classes_dict

    def file_rand_chunk(self, path, time_secs, dtype=np.int16, file_type='wav'):

        header_size = 0

        if file_type[0]=='.':
            file_type = file_type[1:]

        if (file_type.lower()=='wav'):
            header_size=44

        fsize_samples = (os.path.getsize(path)-header_size)/np.dtype(dtype).itemsize
        slice_length_samples = int(self.rate*time_secs)

        if slice_length_samples >= fsize_samples:
            sig = np.fromfile(path, dtype=dtype, offset=header_size)
        else:
            max_offset = fsize_samples - slice_length_samples
            start_samples = random.randint(0, max_offset)
            sig = np.fromfile(path, dtype=dtype, count=slice_length_samples, offset=header_size + start_samples*np.dtype(dtype).itemsize)
        
        return sig


    def __getitem__(self, index):
        
        class_name1 = random.choice(list(self.classes_dict.keys()))
        class_name2 = random.choice(list(self.classes_dict.keys()))

        file1 = self.classes_dict[class_name1]
        file2 = self.classes_dict[class_name2]

        c_1 = self.file_rand_chunk(file1, self.chunk_size)
        c_2 = self.file_rand_chunk(file1, self.chunk_size)
        c_rnd = self.file_rand_chunk(file2, self.chunk_size)
        
        return c_1, c_2, c_rnd

    def __len__(self):
        return self.n_batches*self.batch_size
    
    def get_nlabels(self):
        return self.n_labels