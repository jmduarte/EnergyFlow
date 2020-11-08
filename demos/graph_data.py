import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import pandas as pd
import energyflow as ef
import glob

ONE_HUNDRED_GEV = 100.0

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, njets=1000):        
        self.njets = njets
        super(GraphDataset, self).__init__(root, transform, pre_transform) 


    @property
    def raw_file_names(self):
        return ['QG_jets.npz']

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        proc_list = glob.glob(osp.join(self.processed_dir, 'data_*.pt'))
        return_list = list(map(osp.basename, proc_list))
        return return_list

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def process(self):
        Js = []
        R = 0.4
        for raw_path in self.raw_paths:
            # load quark and gluon jets
            X, y = ef.qg_jets.load(self.njets, pad=False, cache_dir=self.root+'/raw')
            # the jet radius for these jets
            # process jets
            Js = []
            for i,x in enumerate(X): 
                if i >= self.njets: break
                # ignore padded particles and removed particle id information
                x = x[x[:,0] > 0,:3]
                # center jet according to pt-centroid
                yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
                x[:,1:3] -= yphi_avg
                # mask out any particles farther than R=0.4 away from center (rare)
                x = x[np.linalg.norm(x[:,1:3], axis=1) <= R]
                # add to list
                Js.append(x)
        jetpairs = [[i, j] for (i, j) in itertools.product(range(self.njets),range(self.njets))]
        datas = []
        for k, (i, j) in enumerate(jetpairs):            
            if k%100 == 0:
                datas = []
            emdval = ef.emd.emd(Js[i], Js[j], R=R)/ONE_HUNDRED_GEV
            Ei = np.sum(Js[i][:,0])
            Ej = np.sum(Js[j][:,0])
            jiNorm = Js[i].copy()
            jjNorm = Js[j].copy()
            jiNorm[:,0] = jiNorm[:,0]/Ei
            jjNorm[:,0] = jjNorm[:,0]/Ej
            jetpair = np.concatenate([jiNorm, jjNorm], axis=0)
            nparticles_i = len(Js[i])
            nparticles_j = len(Js[j])
            pairs = [[m, n] for (m, n) in itertools.product(range(0,nparticles_i),range(nparticles_i,nparticles_i+nparticles_j))]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index=edge_index.t().contiguous()
            u = torch.tensor([[Ei/ONE_HUNDRED_GEV, Ej/ONE_HUNDRED_GEV]], dtype=torch.float)
            x = torch.tensor(jetpair, dtype=torch.float)
            y = torch.tensor([[emdval]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y, u=u)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            datas.append([data])           
            if k%100 == 99:
                datas = sum(datas,[])
                torch.save(datas, osp.join(self.processed_dir, 'data_{}.pt'.format(k)))
            
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data