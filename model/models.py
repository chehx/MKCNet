import torch
import torch.nn as nn
from .VGG import vgg16
from .CBAM import CBAM, Flatten

MODEL_LIST = ['VanillaNet', 'DETACH', 'CANet', 'MultiTaskNet', 'MTMRNet']

class VanillaNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(VanillaNet, self).__init__()
        self.cfg = cfg

        self.model = vgg16(in_channel = cfg.DATASET.CHANNEL_NUM)
        
        in_features = self.model.feature_size
                 
        self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, cfg.DATASET.NUM_T))

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)  
        return out

# # # # # # # # # # # # # Code Credits # # # # # # # # # # # 
# Paper: CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading
# Github: https://github.com/xmengli/CANet
# Arxiv: https://arxiv.org/abs/1911.01376
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class CANet(nn.Module):
    
    def __init__(self, cfg) -> None:
        super(CANet, self).__init__()
        self.cfg = cfg

        self.model = vgg16(in_channel = cfg.DATASET.CHANNEL_NUM)
        
        in_features = self.model.feature_size

        self.branch_bam1 = nn.Sequential(
            CBAM(in_features),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features, in_features//2))

        self.branch_bam2 = nn.Sequential(
            CBAM(in_features),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features, in_features//2))

        self.classifier_specific_1 = nn.Linear(in_features//2, cfg.DATASET.NUM_T)
        self.classifier_specific_2 = nn.Linear(in_features//2, cfg.DATASET.NUM_IQ)
    
        self.branch_bam3 = nn.Sequential(
            CBAM(in_features//2, no_spatial=True),
            Flatten())

        self.branch_bam4 = nn.Sequential(
            CBAM(in_features//2, no_spatial=True),
            Flatten())
        
        self.classifier1 = nn.Linear(in_features//2, cfg.DATASET.NUM_T)
        self.classifier2 = nn.Linear(in_features//2, cfg.DATASET.NUM_IQ)

    def forward(self, x):
        x = self.model(x)

        #  task specific feature
        x1 = self.branch_bam1(x)
        x2 = self.branch_bam2(x)
        out1 = self.classifier_specific_1(x1)
        out2 = self.classifier_specific_2(x2)

        # task correlation
        x1_att = self.branch_bam3(x1.view(x1.size(0), -1, 1, 1))
        x2_att = self.branch_bam4(x2.view(x2.size(0), -1, 1, 1))

        x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
        x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)

        out_t1 = self.classifier1(x1)
        out_t2 = self.classifier2(x2)

        return out_t1, out_t2, out1, out2

# # # # # # # # # # # # # Reference # # # # # # # # # # # 
# Paper: Learning Robust Representation for Joint Grading of Ophthalmic Diseases via Adaptive Curriculum and Feature Disentanglement
# Arxiv: https://arxiv.org/abs/2207.04183
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class DETACH(nn.Module):
    def __init__(self, cfg) -> None:
        super(DETACH, self).__init__()
        self.cfg = cfg

        self.model_Q = vgg16(in_channel = cfg.DATASET.CHANNEL_NUM)
        self.model_D = vgg16(in_channel = cfg.DATASET.CHANNEL_NUM)

        in_features = self.model_Q.feature_size

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier1 = nn.Linear(in_features*2, cfg.DATASET.NUM_T)
        self.classifier2 = nn.Linear(in_features*2, cfg.DATASET.NUM_IQ)

    def forward(self, x):
        feature_q = self.model_Q(x)
        feature_d = self.model_D(x)

        feature_q = (self.pool(feature_q)).squeeze()
        feature_d = (self.pool(feature_d)).squeeze()

        out_q = self.classifier1(torch.cat((feature_q, feature_d.detach()), dim=1))
        out_d = self.classifier2(torch.cat((feature_q.detach(), feature_d), dim=1))

        return out_q, out_d