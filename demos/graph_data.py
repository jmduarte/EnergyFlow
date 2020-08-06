import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
import tables
import numpy as np
import pandas as pd
import energyflow as ef

ONE_HUNDRED_GEV = 100.0

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, start=0, stop=-1):
        self.start = start
        self.stop = stop
        super(GraphDataset, self).__init__(root, transform, pre_transform) 


    @property
    def raw_file_names(self):
        return ['QG_jets.npz']

    @property
    def processed_file_names(self):
        njetpairs = 10000
        if self.start!=0 and self.stop!=-1:
            njetpairs = self.stop-self.start
            return ['data_{}.pt'.format(i) for i in range(self.start,self.stop)]
        else:
            return ['data_{}.pt'.format(i) for i in range(njetpairs)]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def process(self):
        Js = []
        num = 50
        R = 0.4
        for raw_path in self.raw_paths:
            # load quark and gluon jets
            X, y = ef.qg_jets.load(200, pad=False, cache_dir=self.root+'/raw')
            # the jet radius for these jets
            # process jets
            Gs, Qs = [], []
            for arr,events in [(Gs, X[y==0]), (Qs, X[y==1])]:
                for i,x in enumerate(events): 
                    if i >= num: break
                    # ignore padded particles and removed particle id information
                    x = x[x[:,0] > 0,:3]
                    # center jet according to pt-centroid
                    yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
                    x[:,1:3] -= yphi_avg
                    # mask out any particles farther than R=0.4 away from center (rare)
                    x = x[np.linalg.norm(x[:,1:3], axis=1) <= R]
                    # add to list
                    arr.append(x)
            Js.extend(Gs)
            Js.extend(Qs)
        jetpairs = [[i, j] for (i, j) in itertools.product(range(2*num),range(2*num))]
        for k, (i, j) in enumerate(jetpairs):
            emdval = ef.emd.emd(Js[i], Js[j], R=R)/ONE_HUNDRED_GEV
            jetpair = np.concatenate([Js[i], Js[j]], axis=0)
            Ei = np.sum(Js[i][:,0])
            Ej = np.sum(Js[j][:,0])
            nparticles_i = len(Js[i])
            nparticles_j = len(Js[j])
            pairs = [[m, n] for (m, n) in itertools.product(range(0,nparticles_i),range(nparticles_i,nparticles_i+nparticles_j))]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index=edge_index.t().contiguous()
            u = torch.tensor([[Ei, Ej]], dtype=torch.float)
            x = torch.tensor(jetpair, dtype=torch.float)
            y = torch.tensor([[emdval]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y, u=u)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(k)))
            
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data