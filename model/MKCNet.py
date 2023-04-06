import torch
import torch.nn as nn
import numpy as np
from .VGG import vgg16
from .CBAM import CBAM, Flatten
import torch.nn.functional as F
import copy

def model_fit(x_pred, x_output, pri=True, num_output=3, gamma=2):
    if not pri:
        x_output_onehot = x_output
    else:
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).cuda()
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

    # apply focal loss
    loss = x_output_onehot * (1 - x_pred)**gamma * torch.log(x_pred + 1e-20)
    return torch.sum(-loss, dim=1)

def entropy_loss(x_pred):
    # compute entropy loss
    x_pred = torch.mean(x_pred, dim=0)
    entro_loss = x_pred * torch.log(x_pred + 1e-20)
    return torch.sum(entro_loss)

class MetaLearner(nn.Module):
    def __init__(self, psi, cfg):
        super(MetaLearner, self).__init__()
        """
            Meta Learner:
            takes the input and generates auxiliary labels with masked softmax for an auxiliary task.
        """
        self.class_nb = psi

        self.model = vgg16(in_channel = cfg.DATASET.CHANNEL_NUM)

        in_features = self.model.feature_size

        # define fc-layers in VGG-16 (output auxiliary classes \sum_i\psi[i])
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, int(np.sum(self.class_nb))),
        )

    # define masked softmax
    def mask_softmax(self, x, mask, dim=1):
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        return logits

    def forward(self, x, y):
        feature = self.model(x)        
        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].cuda()

        predict = self.classifier(feature.view(feature.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        return label_pred


class TaskNet(nn.Module):
    def __init__(self, psi, cfg) -> None:
        super(TaskNet, self).__init__()

        self.model = vgg16(in_channel = cfg.DATASET.CHANNEL_NUM)

        in_features = self.model.feature_size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
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

        self.branch_bam3 = nn.Sequential(
            CBAM(in_features),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features, in_features//2))

        out_feature = in_features//2

        self.branch_cor = nn.Sequential(
            CBAM(out_feature, no_spatial=True),
            Flatten())
        out_feature = out_feature * 2
        out_feature_2 = out_feature // 2

        # T task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(out_feature, out_feature),
            nn.ReLU(inplace=True),
            nn.Linear(out_feature, cfg.DATASET.NUM_T))

        num_M = int(np.sum(psi))

        self.classifier2 = nn.Sequential(
            nn.Linear(out_feature_2, out_feature_2),
            nn.ReLU(inplace=True),
            nn.Linear(out_feature_2, num_M))

        # IQ task prediction
        self.classifier3 = nn.Sequential(
            nn.Linear(out_feature_2, out_feature_2),
            nn.ReLU(inplace=True),
            nn.Linear(out_feature_2, cfg.DATASET.NUM_IQ))

    def branch_bam_ff(self, input, weights, index):
        avg_pool = F.avg_pool2d( input, (input.size(2), input.size(3)), stride=(input.size(2), input.size(3)))
        channel_att_raw = avg_pool.squeeze()
        channel_att_raw = F.linear(channel_att_raw, weights['branch_bam{:d}.0.ChannelGate.mlp.1.weight'.format(index)], weights['branch_bam{:d}.0.ChannelGate.mlp.1.bias'.format(index)])
        channel_att_raw = F.relu(channel_att_raw, inplace=True)
        channel_att_raw = F.linear(channel_att_raw, weights['branch_bam{:d}.0.ChannelGate.mlp.3.weight'.format(index)], weights['branch_bam{:d}.0.ChannelGate.mlp.3.bias'.format(index)])
        scale = torch.sigmoid( channel_att_raw ).unsqueeze(2).unsqueeze(3).expand_as(input)
        channelout = input * scale
        # print(channelout.shape)
        compress = torch.cat( (torch.max(channelout,1)[0].unsqueeze(1), torch.mean(channelout,1).unsqueeze(1)), dim=1 )
        spatialout = F.conv2d(compress, weights['branch_bam{:d}.0.SpatialGate.spatial.conv.weight'.format(index)], padding=3)
        spatialout = F.batch_norm(spatialout, torch.zeros(spatialout.data.size()[1]).cuda(), torch.ones(spatialout.data.size()[1]).cuda(),
                           weights['branch_bam{:d}.0.SpatialGate.spatial.bn.weight'.format(index)], weights['branch_bam{:d}.0.SpatialGate.spatial.bn.bias'.format(index)],
                           training=True)
        # print(compress.shape, channelout.shape, spatialout.shape)
        net = channelout * torch.sigmoid(spatialout)
        net = F.adaptive_avg_pool2d(net, (1, 1))
        net = net.squeeze()
        net = F.linear(net, weights['branch_bam{:d}.3.weight'.format(index)], weights['branch_bam{:d}.3.bias'.format(index)])
        return net

    def branch_cor_ff(self, input, weights):
        net = input.squeeze()
        net = F.linear(net, weights['branch_cor.0.ChannelGate.mlp.1.weight'], weights['branch_cor.0.ChannelGate.mlp.1.bias'])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['branch_cor.0.ChannelGate.mlp.3.weight'], weights['branch_cor.0.ChannelGate.mlp.3.bias'])
        net = net.squeeze()
        return net

    # define forward conv-layer (will be used in second-derivative step)
    def conv_layer_ff(self, input, weights, index):

        if index < 3:
            net = F.conv2d(input, weights['model.block{:d}.0.weight'.format(index)], weights['model.block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).cuda(), torch.ones(net.data.size()[1]).cuda(),
                            weights['model.block{:d}.1.weight'.format(index)], weights['model.block{:d}.1.bias'.format(index)],
                            training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['model.block{:d}.3.weight'.format(index)], weights['model.block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).cuda(), torch.ones(net.data.size()[1]).cuda(),
                            weights['model.block{:d}.4.weight'.format(index)], weights['model.block{:d}.4.bias'.format(index)],
                            training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2, )
        else:
            net = F.conv2d(input, weights['model.block{:d}.0.weight'.format(index)], weights['model.block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).cuda(), torch.ones(net.data.size()[1]).cuda(),
                            weights['model.block{:d}.1.weight'.format(index)], weights['model.block{:d}.1.bias'.format(index)],
                            training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['model.block{:d}.3.weight'.format(index)], weights['model.block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).cuda(), torch.ones(net.data.size()[1]).cuda(),
                            weights['model.block{:d}.4.weight'.format(index)], weights['model.block{:d}.4.bias'.format(index)],
                            training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['model.block{:d}.6.weight'.format(index)], weights['model.block{:d}.6.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).cuda(), torch.ones(net.data.size()[1]).cuda(),
                            weights['model.block{:d}.7.weight'.format(index)], weights['model.block{:d}.7.bias'.format(index)],
                            training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2)

        return net

    def dense_layer_ff(self, input, weights, index):
        net = F.linear(input, weights['classifier{:d}.0.weight'.format(index)], weights['classifier{:d}.0.bias'.format(index)])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['classifier{:d}.2.weight'.format(index)], weights['classifier{:d}.2.bias'.format(index)])
        return net

    def forward(self, x, weights=None):
        if weights is None:
            x = self.model(x)

            # task specific feature
            x1_bam = self.branch_bam1(x)
            x2_bam = self.branch_bam2(x)
            x3_bam = self.branch_bam3(x)

            # task correlation
            UM_att = self.branch_cor(x2_bam.view(x2_bam.size(0), -1, 1, 1))                
            x1 = torch.cat((x1_bam, UM_att), dim = 1)

            out_T = self.classifier1(x1)
            out_M = self.classifier2(x2_bam)
            out_IQ = self.classifier3(x3_bam)
        
        else:

            block1 = self.conv_layer_ff(x, weights, 1)
            block2 = self.conv_layer_ff(block1, weights, 2)
            block3 = self.conv_layer_ff(block2, weights, 3)
            block4 = self.conv_layer_ff(block3, weights, 4)
            features = self.avgpool(self.conv_layer_ff(block4, weights, 5))

            x1 = self.branch_bam_ff(features, weights, 1)
            x2 = self.branch_bam_ff(features, weights, 2)
            x3 = self.branch_bam_ff(features, weights, 3)

            UM_att = self.branch_cor_ff(x2.view(x2.size(0), -1, 1, 1), weights)
            x1 = torch.cat((x1, UM_att), dim = 1)
            
            out_T = self.dense_layer_ff(x1, weights, 1)
            out_M = self.dense_layer_ff(x2, weights, 2)
            out_IQ = self.dense_layer_ff(x3, weights, 3)

        return out_T, out_M, out_IQ
    

class FirstOrderTaskNet(TaskNet):
    def __init__(self, psi, cfg) -> None:
        super(FirstOrderTaskNet, self).__init__(psi, cfg)

    def forward(self, x):
        x = self.model(x)

        #  task specific feature
        x1 = self.branch_bam1(x)
        x2 = self.branch_bam2(x)
        x3 = self.branch_bam3(x)

        UM_att = self.branch_cor(x2.view(x2.size(0), -1, 1, 1))                
        x1 = torch.cat((x1, UM_att), dim = 1)

        out_T = self.classifier1(x1)
        out_M = self.classifier2(x2)
        out_IQ = self.classifier3(x3)
        
        return out_T, out_M, out_IQ

class ComputeFirstOrder:
    def __init__(self, model, metalearner, cfg):
        self.model = model
        self.model_ = copy.deepcopy(model)
        self.model_.cuda()
        self.metalearner = metalearner
        self.softmax = torch.nn.Softmax(dim=1)
        self.cfg = cfg

    def unrolled_backward(self, image, label, M_label, IQ_label, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        #  compute unrolled multi-task network theta_1^+ (virtual step)
        meta_label = self.metalearner(image, M_label)
        output_T, output_M, output_IQ = self.model(image)

        output_T = self.softmax(output_T)
        output_M = self.softmax(output_M)
        output_IQ = self.softmax(output_IQ)

        loss1 = torch.mean(model_fit(output_T, label, True, self.cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, self.cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(model_fit(output_M, meta_label, False)) / float(self.cfg.MODEL.META_LENGTH)
        loss4 = torch.mean(entropy_loss(meta_label))
        loss = loss1 + loss2 + loss3

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step: theta_1^+ = theta_1 - alpha * (primary loss + auxiliary loss)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    if model_optim.param_groups[0]['momentum'] == 0:
                        m = 0
                    else:
                        m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

        # meta-training step: updating theta_2
        output_T, output_M, output_IQ = self.model_(image)
        output_T = self.softmax(output_T)
        output_M = self.softmax(output_M)
        output_IQ = self.softmax(output_IQ)

        loss1 = torch.mean(model_fit(output_T, label, True, self.cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, self.cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(F.mse_loss(output_M, torch.zeros_like(output_M, device=image.device)))

        loss = loss1 + loss2 + loss3 * 0 + loss4 * self.cfg.MODEL.LOSS_ENTROPY_WEIGHT 

        # compute hessian (finite difference approximation)
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_)
        hessian = self.compute_hessian(d_model, image, label, M_label, IQ_label)

        metalearner_weights_ = self.metalearner.parameters()
        d_metalearner = torch.autograd.grad(loss4 * self.cfg.MODEL.LOSS_ENTROPY_WEIGHT, metalearner_weights_)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h, d in zip(self.metalearner.parameters(), hessian, d_metalearner):
                mw.grad = - alpha * h + d

    def compute_hessian(self, d_model, image, label, M_label, IQ_label):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # theta_1^l = theta_1 + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        meta_label = self.metalearner(image, M_label)
        output_T, output_M, output_IQ = self.model(image)

        output_T = self.softmax(output_T)
        output_M = self.softmax(output_M)
        output_IQ = self.softmax(output_IQ)

        loss1 = torch.mean(model_fit(output_T, label, True, self.cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, self.cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(model_fit(output_M, meta_label, False)) / float(self.cfg.MODEL.META_LENGTH)
        loss = loss1 + loss2 + loss3

        d_weight_p = torch.autograd.grad(loss, self.metalearner.parameters())

        # theta_1^r = theta_1 - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        meta_label = self.metalearner(image, M_label)
        output_T, output_M, output_IQ = self.model(image)

        output_T = self.softmax(output_T)
        output_M = self.softmax(output_M)
        output_IQ = self.softmax(output_IQ)

        loss1 = torch.mean(model_fit(output_T, label, True, self.cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, self.cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(model_fit(output_M, meta_label, False)) / float(self.cfg.MODEL.META_LENGTH)
        loss = loss1 + loss2 + loss3

        d_weight_n = torch.autograd.grad(loss, self.metalearner.parameters())

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian