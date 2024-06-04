import yaml
import os
import traceback
from src.utils import argparser
from src.utils import PATH
from src import DataManager, NetworkManager, TrainManager, PostProcessor

config = {
    "data": None,
    "target": None,
    "network": None,

    "scaler": "standard",
    "optimizer": "Adam",

    "max_epoch": 2000,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "lr_patience": 80,
    "early_stopping_patience": 200,
    'min_lr': 1e-5,

    "device": "cuda:0",
    "pin_memory": False,

    'hidden_dim': 256,
    'conv_layers':6,
    'linear_layers': 3,
    'dropout': 0.2,
}


def main(config):
    args = argparser.parse_args()
    config.update(vars(args))
    if config.get("load",None) is not None:
        config['model_path'] = config['load']
        _config = yaml.load(open(os.path.join(config['load'],'config.yaml'), 'r'), Loader=yaml.FullLoader)
        _config.update(config)
        config = _config
    else:
        with open(PATH.NET_REFER, "r") as file:
            network_refer = yaml.safe_load(file)[args.network]
        config.update(network_refer)
        config['model_path'] = PATH.get_model_path(config)
    print(config['model_path'])
    dm, nm, tm, pp = None, None, None, None
    train_loaders, val_loaders, test_loaders = None, None, None

    try:
        dm = getattr(DataManager,config['data_manager'])(config)
        dm.init_data()
        train_loaders, val_loaders, test_loaders = dm.get_Dataloaders()

        nm = getattr(NetworkManager,config['network_manager'])(config)

        tm = getattr(TrainManager,config['train_manager'])(config)
        tm.train(nm, train_loaders, val_loaders)
    except:
        print(traceback.format_exc())

    pp = PostProcessor.PostProcessor(config)
    pp.postprocess(dm, nm, tm, train_loaders, val_loaders, test_loaders)


if __name__ == "__main__":
    main(config)