import argparse
import torch, random, numpy as np
from configs.defaults import _C as cfg_default

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/hche/DomainGeneralization/IQADATA", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="DRAC", help="dataset name")
    parser.add_argument("--gpu", type=str,  default="0", help="gpu id")
    parser.add_argument("--output", type=str, default="test", help="output path")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--random", action='store_true')
    parser.add_argument("--override", action='store_true')    
    parser.add_argument("--model", type=str, default='VanillaNet')
    
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.MODEL.NAME = args.model
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NAME = args.dataset
    cfg.merge_from_file(f"./configs/datasets/{args.dataset}.yaml")
    return cfg
