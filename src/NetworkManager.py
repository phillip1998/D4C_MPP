import torch
import importlib
import os
import yaml
import pandas as pd
from .utils import PATH

class NetworkManager:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.last_lr = config.get('learning_rate',0.001)
        self.best_loss = float('inf')
        self.es_patience = config.get('early_stopping_patience',50)
        self.es_counter = 0
        self.state= "train"

        if self.config.get('load',None) is not None:
            self.load_network(self.config['load'])
        else:
            self.init_network()
        self.init_optimizer()
        self.init_scheduler()

    # Initialize the network
        
    def init_network(self):
        module = importlib.import_module('networks.' + self.config['network']).network
        self.network = module(self.config)
        self.network.to(device = self.device)
        print(f"#params: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}" )
        self.loss_fn = self.network.loss_fn


        os.makedirs(self.config['model_path'], exist_ok=True)
        os.system(f"cp {PATH.NET_PATH}/{self.config['network']}.py {self.config['model_path']}/network.py")

        with open(os.path.join(self.config['model_path'],'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
        self.set_unwrapper()

    def load_network(self, path):
        module = importlib.import_module(os.path.join(path,'network').replace('/','.')).network
        self.network = module(self.config)
        self.network.to(device = self.device)
        print(f"#params: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}" )
        self.loss_fn = self.network.loss_fn

        self.config = yaml.load(open(os.path.join(path,'config.yaml'), 'r'), Loader=yaml.FullLoader)
        self.load_params(os.path.join(path,'final.pth'))
        self.learning_curve = pd.read_csv(os.path.join(path,'learning_curve.csv'))
        self.set_unwrapper()

    def set_unwrapper(self):
        from src import DataManager
        self.unwrapper = getattr(DataManager, self.config['data_manager'])(self.config).unwrapper

    def init_optimizer(self):
        self.optimizer = getattr(torch.optim, self.config['optimizer'])(self.network.parameters(), 
                                                                        lr=self.config.get('learning_rate',0.001), 
                                                                        weight_decay=self.config.get('weight_decay',0.0005)
                                                                        )
    def init_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    patience=self.config.get('lr_patience',10), 
                                                                    min_lr=self.config.get('min_lr',1e-7),
                                                                    verbose=True)
    
    # During training

    def train(self):
        self.state = "train"
        self.network.train()

    def eval(self):
        self.state = "eval"
        self.network.eval()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
    def step(self, loader, flag= False, **kargs):
        self.optimizer.zero_grad()
        x= self.unwrapper(*loader,device=self.device)
        y= x['target']
        x.update(kargs)
        y_pred = self.network(**x)
        if kargs.get('get_score',False) or kargs.get('get_feature',False):
            return y_pred
                
        loss = self.loss_fn(y_pred, y)
        if self.state == "train":
            loss.backward()
            self.optimizer.step()
        if flag:
            return y, y_pred.detach(), loss.detach().item(), x
        return y, y_pred.detach(), loss.detach().item()
    
    def predict(self,loader):
        self.optimizer.zero_grad()
        x= self.unwrapper(*loader,device=self.device)
        y= x['target']
        y_pred = self.network(**x)
        return y_pred, y

    
    def scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss)
            self.es_counter = 0
        else:
            self.es_counter += 1

        if self.es_counter > self.es_patience:
            self.load_best_checkpoint()
            return True

        if self.get_lr()<self.last_lr:
            self.last_lr=self.get_lr()
            self.load_best_checkpoint()
            
            

    def save_checkpoint(self, val_loss):
        self.save_params(os.path.join(self.config['model_path'],"param_"+str(val_loss)+".pth"))


    def load_best_checkpoint(self):
        path = os.path.join(self.config['model_path'],"param_"+str(self.best_loss)+".pth")
        if os.path.exists(path):
            self.load_params(path)
            return path
        else:
            return None

    def load_params(self, path):
        self.network.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.network.state_dict(), path)


    