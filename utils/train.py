import torch
from model.MKCNet import model_fit, entropy_loss
from .misc import get_fast_grad_weights
from collections import OrderedDict

def train(model, train_loader, criterion, optimizer, writer, epoch, cfg):

    model_name = cfg.MODEL.NAME

    model.train()
    loss_temp = 0
    softmax = torch.nn.Softmax(dim=1)

    for image, label, IQ_label, M_label in train_loader:

        image = image.cuda()
        label = label.cuda().long()
        IQ_label = IQ_label.cuda().long()
        M_label = M_label.cuda().long()
        optimizer.zero_grad()
           
        if model_name == 'CANet':
            x1, x2, out1, out2 = model(image)
            loss = criterion(x1, label) + criterion(x2, IQ_label) + 0.5 * (criterion(out1, label) + criterion(out2, IQ_label))

        elif model_name == 'DETACH':
            x1, x2 = model(image)
            loss = criterion(x1, label) + criterion(x2, IQ_label)       
        
        elif model_name ==  'VanillaNet':
            x1 = model(image)
            out = softmax(x1)
            loss = torch.mean(model_fit(out, label, True, cfg.DATASET.NUM_T))

        loss.backward()
        optimizer.step()
        loss_temp += loss.item()

    loss_temp = loss_temp / len(train_loader)
    writer.add_scalar('info/loss', loss_temp, epoch)

    return loss_temp


def train_MKCNet(tasknet, metalearner, train_loader, lr, optimizer, optimizer_gen, writer, epoch, cfg):

    tasknet.train()
    softmax = torch.nn.Softmax(dim=1)

    # Stage 1 Train Task Net
    running_loss = subloss1 = subloss2 = subloss3 = 0.0

    for image, label, IQ_label, M_label in train_loader:
        image = image.cuda()
        label = label.cuda().long()
        IQ_label = IQ_label.cuda().long()
        M_label = M_label.cuda().long()

        meta_label = metalearner(image, M_label)
        output, output_M, output_IQ = tasknet(image.cuda())

        optimizer.zero_grad()
        optimizer_gen.zero_grad()

        output = softmax(output)
        output_M = softmax(output_M)
        output_IQ = softmax(output_IQ)

        loss1 = torch.mean(model_fit(output, label, True, cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(model_fit(output_M, meta_label, False)) / float(cfg.MODEL.META_LENGTH)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        subloss1 += loss1.item()
        subloss2 += loss2.item()
        subloss3 += loss3.item()            

    writer.add_scalar("info/round1_task_T", subloss1 / len(train_loader), epoch)
    writer.add_scalar("info/round1_task_IQ", subloss2 / len(train_loader), epoch)
    writer.add_scalar("info/round1_task_M", subloss3 / len(train_loader), epoch)

    # Stage 2 Train Meta Learner
    sumloss1 = sumloss2 = sumloss3 = sumloss4 = 0.0
    
    for image, label, IQ_label, M_label in train_loader:

        image = image.cuda()
        label = label.cuda().long()
        M_label = M_label.cuda().long()
        IQ_label = IQ_label.cuda().long()

        meta_label = metalearner(image, M_label)
        output, output_M, output_IQ = tasknet(image.cuda())

        optimizer.zero_grad()
        optimizer_gen.zero_grad()

        output = softmax(output)
        output_M = softmax(output_M)
        output_IQ= softmax(output_IQ)

        loss1 = torch.mean(model_fit(output, label, True, cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(model_fit(output_M, meta_label, False)) / float(cfg.MODEL.META_LENGTH)
        loss4 = torch.mean(entropy_loss(meta_label))
        loss = loss1 + loss2 + loss3

        # current theta_1
        fast_weights = OrderedDict((name, param) for (name, param) in tasknet.named_parameters())
    
        grads = torch.autograd.grad(loss, tasknet.parameters(), create_graph=True)

        fast_weights_updated = get_fast_grad_weights(fast_weights, optimizer, grads, lr, cfg)

        output, _, output_IQ = tasknet(image, fast_weights_updated)
        output = softmax(output)
        output_IQ = softmax(output_IQ)
        loss1 = torch.mean(model_fit(output, label, True, cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, cfg.DATASET.NUM_IQ))
        loss = loss1 + loss2 + loss4 * cfg.MODEL.LOSS_ENTROPY_WEIGHT

        loss.backward()

        optimizer_gen.step()
        running_loss += loss.item()
        sumloss1 += loss1.item()
        sumloss2 += loss2.item()
        sumloss3 += loss3.item()
        sumloss4 += loss4.item()

    writer.add_scalar("info/round2_task_T", sumloss1 / len(train_loader), epoch)
    writer.add_scalar("info/round2_task_IQ", sumloss2 / len(train_loader), epoch)
    writer.add_scalar("info/round2_task_M", sumloss3 / len(train_loader), epoch)
    writer.add_scalar("info/label_entrophy", sumloss4 / len(train_loader), epoch)

    writer.add_scalar("loss", running_loss / len(train_loader), epoch)
    
    return running_loss / len(train_loader)

def train_MKCNet_firstorder(tasknet, metalearner, model_compute, train_loader, lr, optimizer, optimizer_gen, writer, epoch, cfg):
    tasknet.train()
    softmax = torch.nn.Softmax(dim=1)

    # Stage 1 Train Task Net
    running_loss = subloss1 = subloss2 = subloss3 = 0.0

    for image, label, IQ_label, M_label in train_loader:
        image = image.cuda()
        label = label.cuda().long()
        IQ_label = IQ_label.cuda().long()
        M_label = M_label.cuda().long()

        meta_label = metalearner(image, M_label)
        output, output_M, output_IQ = tasknet(image)

        optimizer.zero_grad()
        optimizer_gen.zero_grad()

        output = softmax(output)
        output_M = softmax(output_M)
        output_IQ = softmax(output_IQ)

        loss1 = torch.mean(model_fit(output, label, True, cfg.DATASET.NUM_T))
        loss2 = torch.mean(model_fit(output_IQ, IQ_label, True, cfg.DATASET.NUM_IQ))
        loss3 = torch.mean(model_fit(output_M, meta_label, False)) / float(cfg.MODEL.META_LENGTH)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        subloss1 += loss1.item()
        subloss2 += loss2.item()
        subloss3 += loss3.item()

    writer.add_scalar("info/round1_task_T", subloss1 / len(train_loader), epoch)
    writer.add_scalar("info/round1_task_IQ", subloss2 / len(train_loader), epoch)
    writer.add_scalar("info/round1_task_M", subloss3 / len(train_loader), epoch)

    # Stage 2 Train Meta Learner
    
    for image, label, IQ_label, M_label in train_loader:

        image = image.cuda()
        label = label.cuda().long()
        M_label = M_label.cuda().long()
        IQ_label = IQ_label.cuda().long()

        optimizer.zero_grad()
        optimizer_gen.zero_grad()

        model_compute.unrolled_backward(image, label, M_label, IQ_label, lr, optimizer)

        optimizer_gen.step()

    writer.add_scalar("loss", running_loss / len(train_loader), epoch)
    
    return running_loss / len(train_loader)