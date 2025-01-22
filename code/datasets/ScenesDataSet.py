from datasets import SceneData
import numpy as np
from torch.utils.data.dataset import Dataset
from datasets import Euclidean

class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.n = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.n / self.batch_size))
        self.shuffle=shuffle
        self.permutation = self.init_permutation()
        self.current_batch = 0
        self.device = 'cpu'

    def init_permutation(self):
        return np.random.permutation(self.n) if self.shuffle else np.arange(self.n)

    def __iter__(self):
        self.current_batch = 0
        self.permutation = self.init_permutation()
        return self

    def __next__(self):
        if self.current_batch == self.num_batches:    
            raise StopIteration
        start_ind = self.current_batch*self.batch_size
        end_ind = min((self.current_batch+1)*self.batch_size, self.n)
        current_indices = self.permutation[start_ind:end_ind]
        self.current_batch += 1
        return [self.dataset[i].to(self.device) for i in current_indices]

    def __len__(self):
        return self.n

    def to(self, device, **kwargs):
        self.device = device
        return self


class myDataSetds(Dataset):
    def __init__(self, conf, datalist, flag):
        self.datalist = datalist
        self.conf = conf
        self.flag = flag
        self.dilute_M = conf.get_bool('dataset.diluteM', default=False)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        file = self.datalist[idx]
        M, Ns, Rs, ts, quats, mask = Euclidean.get_raw_data(self.conf, file, self.flag)
        data = SceneData.SceneData(M, Ns, Rs, ts, quats, mask, file, self.dilute_M)
        return data


class myDataSet():
    def __init__(self, conf, flag, datalist, batch_size=1, shuffle=False):
        self.n = len(datalist)
        self.datalist = datalist
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.n / self.batch_size))
        self.shuffle=shuffle
        self.permutation = self.init_permutation()
        self.current_batch = 0
        self.device = 'cpu'
        self.conf = conf
        self.flag = flag
        self.dilute_M = conf.get_bool('dataset.diluteM', default=False)

        # print(f"num_batches {self.num_batches}")
        # print(f"permutation {self.permutation}")
        # print(f"flag {self.flag}")

    def init_permutation(self):
        return np.random.permutation(self.n) if self.shuffle else np.arange(self.n)

    def __iter__(self):
        self.current_batch = 0
        self.permutation = self.init_permutation()
        # print(f"in iter, perm: {self.permutation}")
        return self

    def __next__(self):
        if self.current_batch == self.num_batches:    
            raise StopIteration
        start_ind = self.current_batch*self.batch_size
        # print(f"start ind: {start_ind}")
        end_ind = min((self.current_batch+1)*self.batch_size, self.n)
        current_indices = self.permutation[start_ind:end_ind]
        # print(f"current_indices: {current_indices}")
        # print(f"current data: {self.datalist[current_indices[0]]}")
        self.current_batch += 1
        return [self.loaddata(self.datalist[i]).to(self.device) for i in current_indices]

    def __len__(self):
        return self.n

    def to(self, device, **kwargs):
        self.device = device
        return self
    
    def loaddata(self, file):
        M, Ns, Rs, ts, quats, mask, gpss, color, scale, use_spatial_encoder, dsc_idx, dsc_data, dsc_shape = Euclidean.get_raw_data(self.conf, file, self.flag)
        data = SceneData.SceneData(M, Ns, Rs, ts, quats, mask, file, gpss, color, scale, self.dilute_M, use_spatial_encoder, 
                                   dsc_idx, dsc_data, dsc_shape)
        return data


class ScenesDataSet:
    def __init__(self, data_list, return_all, min_sample_size=10, max_sample_size=30):
        super().__init__()
        self.data_list = data_list
        self.return_all = return_all
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size

    def __getitem__(self, item):
        current_data = self.data_list[item]
        if self.return_all:
            return current_data
        else:                          
            max_sample = min(self.max_sample_size, len(current_data.scan_name))
            if self.min_sample_size >= max_sample:
                sample_fraction = max_sample        
            else:
                sample_fraction = np.random.randint(self.min_sample_size, max_sample + 1)
            return SceneData.sample_data(current_data, sample_fraction)

    def __len__(self):
        return len(self.data_list)