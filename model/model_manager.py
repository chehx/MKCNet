from .models import CANet, MTMRNet, MultiTaskNet, VanillaNet, DETACH
from .MKCNet import FirstOrderTaskNet, ComputeFirstOrder, MetaLearner, TaskNet

def get_model(cfg, psi = None):

    assert cfg.MODEL.NAME in cfg.ALL_MODELS, 'model not found'
    model_name = cfg.MODEL.NAME

    if model_name in ['CANet', 'MTMRNet', 'MultiTaskNet', 'VanillaNet', 'DETACH']:
        model_class = globals()[model_name]
        model = model_class(cfg)
        return model

    elif model_name == 'FirstOrder_MKCNet':
        model = FirstOrderTaskNet(psi, cfg)
        label_gen = MetaLearner(psi, cfg)
        model_compute = ComputeFirstOrder(model, label_gen, cfg)
        return model, label_gen, model_compute
    
    elif model_name == 'MKCNet':
        assert psi is not None, 'psi is None'
        model = TaskNet(psi, cfg)
        label_gen = MetaLearner(psi, cfg)
        return model, label_gen
