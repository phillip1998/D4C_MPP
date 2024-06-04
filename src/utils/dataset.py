import torch
from torch.utils.data import Dataset
import dgl
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, graphs=None, target=None, smiles=None):
        if graphs is None: return
        self.graphs = graphs
        self.node_feature = [g.ndata['f'] for g in graphs]
        self.edge_feature = [g.edata['f'] for g in graphs]
        self.target = torch.tensor(target).float()
        self.smiles = smiles

        self.DataList= [self.graphs, self.node_feature, self.edge_feature, self.target, self.smiles]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.node_feature[idx], self.edge_feature[idx], self.target[idx], self.smiles[idx]
    
    def reload(self, data):
        self.graphs, self.node_feature, self.edge_feature, self.target, self.smiles = data
        self.target = torch.stack(self.target)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)
        self.DataList= [self.graphs, self.node_feature, self.edge_feature, self.target, self.smiles]

    def subDataset(self, idx):
        self.graphs = [self.graphs[i] for i in idx]
        self.node_feature = [self.node_feature[i] for i in idx]
        self.edge_feature = [self.edge_feature[i] for i in idx]
        self.target = self.target[np.array(idx, dtype=int)]
        self.smiles = [self.smiles[i] for i in idx]

    @staticmethod
    def collate(samples):
        graphs, node_feature, edge_feature, target, smiles = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.concat(node_feature,dim=0), torch.concat(edge_feature,dim=0), torch.stack(target,dim=0), smiles
    
    @staticmethod
    def unwrapper(batch_graph, node_feature, edge_feature, target, smiles,device='cpu'):
        batch_graph = batch_graph.to(device=device)
        node_feature = node_feature.float().to(device=device)
        edge_feature = edge_feature.float().to(device=device)
        target = target.float().to(device=device)

        return {"graph":batch_graph, "node_feats":node_feature, "edge_feats":edge_feature, "target":target, "smiles":smiles}
    
class ImgGraphDataset(Dataset):
    def __init__(self, graphs=None, target=None, smiles=None):
        if graphs is None: return
        self.graphs = graphs
        self.r_node = [g.nodes['r_nd'].data['f'] for g in graphs]
        self.r2r_edge = [g.edges['r2r'].data['f'] for g in graphs]
        self.i_node = [g.nodes['i_nd'].data['f'] for g in graphs]
        self.d_node = [g.nodes['d_nd'].data['f'] for g in graphs]
        self.d2d_edge = [g.edges['d2d'].data['dist'] for g in graphs]
        self.target = torch.tensor(target).float()
        self.smiles = smiles

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.r_node[idx], self.r2r_edge[idx], self.i_node[idx], self.d_node[idx], self.d2d_edge[idx], self.target[idx], self.smiles[idx]
    
    def reload(self, data):
        self.graphs, self.r_node, self.r2r_edge, self.i_node, self.d_node, self.d2d_edge, self.target, self.smiles = data
        self.target = torch.stack(self.target)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)

    def subDataset(self, idx):
        self.graphs = [self.graphs[i] for i in idx]
        self.r_node = [self.r_node[i] for i in idx]
        self.r2r_edge = [self.r2r_edge[i] for i in idx]
        self.i_node = [self.i_node[i] for i in idx]
        self.d_node = [self.d_node[i] for i in idx]
        self.d2d_edge = [self.d2d_edge[i] for i in idx]
        self.target = self.target[np.array(idx, dtype=int)]
        self.smiles = [self.smiles[i] for i in idx]

    @staticmethod
    def collate(samples):
        graphs, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.concat(r_node,dim=0), torch.concat(r2r_edge,dim=0), torch.concat(i_node,dim=0), torch.concat(d_node,dim=0), torch.concat(d2d_edge,dim=0), torch.stack(target,dim=0), smiles
    
    @staticmethod
    def unwrapper(graph, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles, device='cpu'):
        graph = graph.to(device=device)
        r_node = r_node.float().to(device=device)
        r2r_edge = r2r_edge.float().to(device=device)
        i_node = i_node.float().to(device=device)
        d_node = d_node.float().to(device=device)
        d2d_edge = d2d_edge.float().to(device=device)
        target = target.float().to(device=device)
        return {'graph':graph, 'r_node':r_node, 'r_edge':r2r_edge, 'i_node':i_node, 'd_node':d_node, 'd_edge':d2d_edge, 'target':target, 'smiles':smiles}
    
class GraphDataset_withSolv(GraphDataset):
    def __init__(self, graphs=None, solv_graphs=None, target=None, smiles=None, solv_smiles=None):
        super().__init__(graphs, target, smiles)
        if graphs is None: return
        self.solv_graphs = solv_graphs
        self.solv_node_feature = [g.ndata['f'] for g in solv_graphs]
        self.solv_edge_feature = [g.edata['f'] for g in solv_graphs]
        self.solv_smiles = solv_smiles

    def __getitem__(self, idx):
        return self.graphs[idx], self.node_feature[idx], self.edge_feature[idx], self.solv_graphs[idx], self.solv_node_feature[idx], self.solv_edge_feature[idx], self.target[idx], self.smiles[idx], self.solv_smiles[idx]
    
    def reload(self, data):
        self.graphs, self.node_feature, self.edge_feature, self.solv_graphs, self.solv_node_feature, self.solv_edge_feature, self.target, self.smiles, self.solv_smiles = data
        self.target = torch.concat(self.target)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)
        self.DataList= [self.graphs, self.node_feature, self.edge_feature, self.solv_graphs, self.solv_node_feature, self.solv_edge_feature, self.target, self.smiles, self.solv_smiles]

    def subDataset(self, idx):
        self.graphs = [self.graphs[i] for i in idx]
        self.node_feature = [self.node_feature[i] for i in idx]
        self.edge_feature = [self.edge_feature[i] for i in idx]
        self.solv_graphs = [self.solv_graphs[i] for i in idx]
        self.solv_node_feature = [self.solv_node_feature[i] for i in idx]
        self.solv_edge_feature = [self.solv_edge_feature[i] for i in idx]
        self.target = self.target[np.array(idx, dtype=int)]
        self.smiles = [self.smiles[i] for i in idx]
        self.solv_smiles = [self.solv_smiles[i] for i in idx]


    @staticmethod
    def collate(samples):
        graphs, node_feature, edge_feature, solv_graphs, solv_node_feature, solv_edge_feature, target, smiles, solv_smiles = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_solv_graph = dgl.batch(solv_graphs)
        return batched_graph, torch.concat(node_feature,dim=0), torch.concat(edge_feature,dim=0), batched_solv_graph, torch.concat(solv_node_feature,dim=0), torch.concat(solv_edge_feature,dim=0), torch.stack(target,dim=0), smiles, solv_smiles
    
    @staticmethod
    def unwrapper(batch_graph, node_feature, edge_feature, batch_solv_graph, solv_node_feature, solv_edge_feature, target, smiles, solv_smiles, device='cpu'):
        batch_graph = batch_graph.to(device=device)
        node_feature = node_feature.float().to(device=device)
        edge_feature = edge_feature.float().to(device=device)
        batch_solv_graph = batch_solv_graph.to(device=device)
        solv_node_feature = solv_node_feature.float().to(device=device)
        solv_edge_feature = solv_edge_feature.float().to(device=device)
        target = target.float().to(device=device)

        return {"graph":batch_graph, "node_feats":node_feature, "edge_feats":edge_feature, "solv_graph":batch_solv_graph, "solv_node_feats":solv_node_feature, "solv_edge_feats":solv_edge_feature, "target":target, "smiles":smiles, "solv_smiles":solv_smiles}