import argparse
import os
from src.utils import PATH
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", help="Database to be processed")
    parser.add_argument("-t","--target", help="Target to be predicted")
    parser.add_argument("-n","--network", help="ID of the network")
    parser.add_argument("-l","--load", help="Load the model from the path")
    parser.add_argument("-f","--fragment", help="Index for fragmentation")
    parser.add_argument("-e","--epoch", help="Number of epoch", type=int)
    parser.add_argument("-c","--cuda", help="Cuda device", type=int)
    parser.add_argument("--partial_train",help="Partial training", type=float)

    args = parser.parse_args()
    if args.data:
        if not os.path.exists(os.path.join(PATH.DATA_PATH,args.data+".csv")):
            raise FileNotFoundError(f"File {args.data}.csv not found in _Data folder")
    if args.target:
        args.target = args.target.split(",")
    if args.epoch:
        args.max_epoch = args.epoch
    if args.cuda:
        args.device = "cuda:"+str(args.cuda)
    if args.load:
        args.load = os.path.join(PATH.MODEL_PATH,args.load)
        return args
    if args.partial_train:
        if args.partial_train>1:
            args.partial_train = int(args.partial_train)
    
    if args.fragment:
        if ',' in args.fragment:
            args.sculptor_index = [int(i) for i in args.fragment.split(',')]
        else:
            args.sculptor_index = [int(i) for i in args.fragment]


    return args