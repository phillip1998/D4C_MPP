import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import save_graphs, load_graphs
from sklearn.model_selection import train_test_split

from .utils import PATH
from .utils.scaler import Scaler

class DataManager:
    def __init__(self, config):
        self.config = config
        self.data = config["data"]
        self.target = config["target"]
        self.scaler = Scaler(config.get("scaler","identity"))
        self.molecule_columns = ['compound']
        self.molecule_smiles = {}
        self.molecule_graphs = {}
        self.random_seed = config.get("random_seed",42)

        self.import_others()
        config.update({"node_dim":self.gg.node_dim, "edge_dim":self.gg.edge_dim})

    def import_others(self):
        self.graph_type = 'mol'
        from .GraphGenerator import MolGraphGenerator
        from .utils.dataset import GraphDataset
        self.gg =MolGraphGenerator()
        self.dataset =GraphDataset
        self.unwrapper = self.dataset.unwrapper

    def init_temp_data(self,smiles,graphs= None):
        if graphs is None and smiles is None:
            raise ValueError("smiles or graphs should be given")
        self.molecule_smiles['compound'] = np.array(smiles)
        self.set = None
        self.target_value = np.zeros((len(smiles),self.config["target_dim"]))
        self.gg.verbose= False
            
        self.generate_graph('compound',graphs)
        self.prepare_dataset(temp=True)
        return self.valid_smiles, self.invalid_smiles

    def get_temp_Dataloader(self):
        self.prepare_dataset(temp=True)
        return self.init_Dataloader()

    def init_data(self):
        self.load_data()
        self.prepare_graph()

    def load_csv(self):
        try:
            self.df = pd.read_csv(os.path.join(PATH.DATA_PATH,self.data+'.csv'),encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(os.path.join(PATH.DATA_PATH,self.data+'.csv'),encoding='cp949')
            except:
                self.df = pd.read_csv(os.path.join(PATH.DATA_PATH,self.data+'.csv'),encoding='euc-kr')

    def load_data(self):
        self.load_csv()
        for col in self.molecule_columns:
            if col in self.df.columns:
                self.molecule_smiles[col] = np.array(list(self.df[col]))
            else:
                raise ValueError(f"{col} is not in the dataframe")
        if torch.tensor(self.df[self.target].values).dim() == 1:
            self.target_value = self.scaler.fit_transform(torch.tensor(self.df[self.target].values).float().unsqueeze(1))
        else:
            self.target_value = self.scaler.fit_transform(torch.tensor(self.df[self.target].values).float())
        self.config.update({"target_dim":self.target_value.shape[1]})
        self.set = self.df.get("set",None)

    def prepare_graph(self):
        for col in self.molecule_columns:
            if os.path.exists(os.path.join(PATH.GRAPH_PATH,self.data+"_"+col+"_"+self.graph_type+".bin")):
                self.load_graphs(col)
            else:
                self.generate_graph(col)
                self.save_graphs(col)

    def generate_graph(self,col,graphs=None):
        self.molecule_graphs[col] = []
        self.valid_smiles = []
        self.invalid_smiles = []
        if graphs is not None:
            for smi,g in zip(self.molecule_smiles[col],graphs):
                self.molecule_graphs[col].append(g)
                self.valid_smiles.append(smi)
            return

        for smi in self.molecule_smiles[col]:
            g=self.gg.get_graph(smi)
            if g.number_of_nodes()>0:
                self.valid_smiles.append(smi)
            else:
                self.invalid_smiles.append(smi)
            self.molecule_graphs[col].append(g)

        
    def save_graphs(self,col):
        save_graphs(os.path.join(PATH.GRAPH_PATH,self.data+"_"+col+"_"+self.graph_type+".bin"), self.molecule_graphs[col])

    def load_graphs(self,col):
        self.molecule_graphs[col],_= load_graphs(os.path.join(PATH.GRAPH_PATH,self.data+"_"+col+"_"+self.graph_type+".bin"))
        if len(self.molecule_graphs[col]) != len(self.molecule_smiles[col]):    
            raise ValueError("Graphs and smiles are not matched")


    def drop_none_graph(self):
        _masks = []
        for col in self.molecule_columns:
            _masks.append([g.number_of_nodes()>0 for g in self.molecule_graphs[col]])
        mask = np.all(_masks,axis=0)
        self.target_value = self.target_value[mask]
        if self.set is not None:
            self.set = self.set[mask]
        for col in self.molecule_columns:
            self.molecule_smiles[col] = np.array(list(self.molecule_smiles[col][mask]))
            self.molecule_graphs[col] = [g for i,g in enumerate(self.molecule_graphs[col]) if mask[i]]

    def drop_nan_value(self):
        # drow rows where all of the target values are nan
        mask = np.array(~torch.isnan(torch.tensor(self.target_value)).all(dim=1))
        self.target_value = self.target_value[mask]
        if self.set is not None:
            self.set = self.set[mask]
        for col in self.molecule_columns:
            self.molecule_smiles[col] = self.molecule_smiles[col][mask]
            self.molecule_graphs[col] = [g for i,g in enumerate(self.molecule_graphs[col]) if mask[i]]

    def prepare_dataset(self,temp=False):
        self.drop_none_graph()
        if not temp:
            self.drop_nan_value()
        self.whole_dataset = self.init_dataset()

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.target_value, self.molecule_smiles['compound'])
        
    def split_data(self):
        if self.set is None:
            train_dataset, test_dataset = train_test_split(self.whole_dataset, test_size=0.1, random_state=self.random_seed)
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=1/9, random_state=self.random_seed)
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset(),self.dataset(),self.dataset()
            self.train_dataset.reload(list(zip(*train_dataset)))
            self.val_dataset.reload(list(zip(*val_dataset)))
            self.test_dataset.reload(list(zip(*test_dataset)))
        else:
            import copy
            self.train_dataset,self.val_dataset,self.test_dataset = copy.deepcopy(self.whole_dataset),copy.deepcopy(self.whole_dataset),copy.deepcopy(self.whole_dataset)
            
            self.train_dataset.subDataset([i for i,s in enumerate(self.set) if s=="train"])
            self.val_dataset.subDataset([i for i,s in enumerate(self.set) if s=="val"])
            self.test_dataset.subDataset([i for i,s in enumerate(self.set) if s=="test"])
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def init_Dataloader(self,set=None,partial_train=None):
        if set=="train":
            if partial_train and partial_train<1.0:
                self.train_dataset.subDataset(np.random.choice(len(self.train_dataset),int(len(self.train_dataset)*partial_train),replace=False))
                print(f"Partial Training: {len(self.train_dataset)}")
            elif partial_train and partial_train>1:
                self.train_dataset.subDataset(np.random.choice(len(self.train_dataset),partial_train,replace=False))
                print(f"Partial Training: {len(self.train_dataset)}")
            dataset = self.train_dataset
            shuffle = self.config.get('shuffle',True)
        elif set=="val":
            dataset = self.val_dataset
            shuffle = False
        elif set=="test":
            dataset = self.test_dataset
            shuffle = False
        else:
            dataset = self.whole_dataset
            shuffle = False

        return DataLoader(dataset, 
                        batch_size=self.config.get('batch_size',32), 
                        shuffle=shuffle,
                        collate_fn=self.dataset.collate, 
                        pin_memory=self.config.get('pin_memory',True))

    def get_Dataloaders(self):
        self.prepare_dataset()
        self.split_data()

        self.train_loader = self.init_Dataloader("train",partial_train=self.config.get('partial_train',1.0))
        self.val_loader = self.init_Dataloader("val")
        self.test_loader = self.init_Dataloader("test")

        return self.train_loader, self.val_loader, self.test_loader
    
class DataManager_img(DataManager):
    def import_others(self):
        from .GraphGenerator import ImgGraphGenerator
        from .utils.dataset import ImgGraphDataset
        sculptor_index = self.config.get('sculptor_index',[6,1,0])
        self.graph_type = 'img'+str(sculptor_index[0])+str(sculptor_index[1])+str(sculptor_index[2])
        self.gg = ImgGraphGenerator(
            self.config.get('frag_ref',PATH.FRAG_REF),
            sculptor_index
        )
        self.dataset =ImgGraphDataset
        self.unwrapper = self.dataset.unwrapper

class DataManager_withSolv(DataManager):
    def __init__(self, config):
        super(DataManager_withSolv, self).__init__(config)
        self.molecule_columns.append('solvent')

    def import_others(self):
        self.graph_type = 'mol'
        from .GraphGenerator import MolGraphGenerator
        from .utils.dataset import GraphDataset_withSolv
        self.gg =MolGraphGenerator()
        self.dataset =GraphDataset_withSolv
        self.unwrapper = self.dataset.unwrapper
        
    def init_temp_data(self,smiles,graphs= None):
        if graphs is None and smiles is None:
            raise ValueError("smiles or graphs should be given")
        self.molecule_smiles['compound'] = np.array(smiles[:,0])
        self.molecule_smiles['solvent'] = np.array(smiles[:,1])
        self.set = None
        self.target_value = np.zeros((len(smiles),self.config["target_dim"]))
        self.gg.verbose= False
            
        self.generate_graph('compound',graphs)
        self.generate_graph('solvent',graphs)
        self.prepare_dataset(temp=True)
        return self.valid_smiles, self.invalid_smiles

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.molecule_graphs['solvent'], self.target_value, self.molecule_smiles['compound'], self.molecule_smiles['solvent'])