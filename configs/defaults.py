from yacs.config import CfgNode as CN

###########################
# Config definition
###########################
_C = CN()
_C.OUT_DIR = "./output"
_C.SEED = 42
_C.USE_CUDA = True
_C.VERBOSE = True
_C.BATCH_SIZE = 16
_C.DROP_LAST = False
_C.VAL_STEP = 20
_C.MAX_EPOCHS = 200
_C.VALIDATION_DATASET = ['DEEPDR', 'EYEQ', 'IQAD_CXR', 'IQAD_CT']
_C.ALL_MODELS = ['CANet', "VanillaNet", 'DETACH', 'MKCNet', 'FirstOrder_MKCNet']
_C.ALL_DATASETS = ['DEEPDR', 'DRAC', 'EYEQ', 'IQAD_CXR', 'IQAD_CT']
_C.MKCNET_MODEL_LIST = ['MKCNet', 'FirstOrder_MKCNet']

###########################
# Model
###########################
_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.LOSS_ENTROPY_WEIGHT = 0.
_C.MODEL.META_LENGTH = 0

###########################
# Optimizer
###########################
_C.OPTIM = CN()
_C.OPTIM.LR = 0.
_C.OPTIM.META_LR = 0.
_C.OPTIM.WEIGHT_DECAY = 0.
_C.OPTIM.META_WEIGHT_DECAY = 0.
_C.OPTIM.STEP_SIZE = 50

###########################
# Dataset
###########################
_C.DATASET = CN()
_C.DATASET.NAME = ""
_C.DATASET.NORMALIZATION_MEAN =[0., 0., 0.]
_C.DATASET.NORMALIZATION_STD =[0., 0., 0.]
_C.DATASET.NUM_T = 0
_C.DATASET.NUM_IQ = 0
_C.DATASET.NUM_M = 0
_C.DATASET.CHANNEL_NUM = 0