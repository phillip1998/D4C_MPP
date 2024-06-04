import os
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

DATA_PATH = main_dir+"/_Data"
GRAPH_PATH = main_dir+"/_Graphs"
MODEL_PATH = main_dir+"/_Models"
NET_PATH = base_dir+"/networks"
NET_REFER = base_dir+"/network_refer.yaml"
FRAG_REF = base_dir+"/src/functional_group.csv"

def check_path():
    global DATA_PATH, GRAPH_PATH, MODEL_PATH, NET_REFER, FRAG_REF, NET_PATH
    if not os.path.exists(DATA_PATH):
        if os.path.exists("../"+DATA_PATH):
            DATA_PATH = "../"+DATA_PATH
        else:
            raise Exception("Data Path not found")
    if not os.path.exists(GRAPH_PATH):
        if os.path.exists("../"+GRAPH_PATH):
            GRAPH_PATH = "../"+GRAPH_PATH
        else:
            raise Exception("Graph Path not found")
    if not os.path.exists(MODEL_PATH):
        if os.path.exists("../"+MODEL_PATH):
            MODEL_PATH = "../"+MODEL_PATH
        else:
            raise Exception("Model Path not found")
    if not os.path.exists(NET_REFER):
        if os.path.exists("../"+NET_REFER):
            NET_REFER = "../"+NET_REFER
        else:
            raise Exception("Network Refer Path not found")
    if not os.path.exists(FRAG_REF):
        if os.path.exists("../"+FRAG_REF):
            FRAG_REF = "../"+FRAG_REF
        else:
            raise Exception("Functional Group Reference Path not found")
    if not os.path.exists(NET_PATH):
        if os.path.exists("../"+NET_PATH):
            NET_PATH = "../"+NET_PATH
        else:
            raise Exception("Network Path not found")

def get_model_path(config, make_dir=True):
    if config['version']=="1.0":
        path = os.path.join(MODEL_PATH,config['network']+"_"+config['data']+'_'+','.join(config['target']))
        if 'sculptor_index' in config:
            path += "_"+''.join([str(i) for i in config['sculptor_index']])
        path += "_"+time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(path) and make_dir:
            os.makedirs(path)
        return path
