import logging
import torch
from sklearn.metrics import roc_auc_score, accuracy_score,f1_score
from model.MKCNet import model_fit

def model_validate(model, data_loader, writer, criterion, epoch, val_type, cfg):
    model.eval()
    model_name = cfg.MODEL.NAME
    average_dict = {'DEEPDR': 'macro', 'DRAC': 'macro', 'EYEQ': 'macro', 'IQAD_CXR': 'binary', 'IQAD_CT': 'binary'}
    
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, _, M_label in data_loader:
            image = image.cuda()
            label = label.cuda().long()
            M_label = M_label.cuda().long()

            if model_name == 'CANet':
                output, _, _, _ = model(image)

            elif model_name == "DETACH":
                output, _ = model(image)
            
            elif model_name in cfg.MKCNET_MODEL_LIST:
                output, _, _ = model(image.cuda())  
            
            else:
                output = model(image)

            if model_name in cfg.MKCNET_MODEL_LIST:
                loss += torch.mean(model_fit(softmax(output), label, True, cfg.DATASET.NUM_T))
            else:
                loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            # label_list.append(M_label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average=average_dict[cfg.DATASET.NAME])
        if average_dict[cfg.DATASET.NAME] == 'binary':
            auc = roc_auc_score(label, pred)
        else:
            auc = roc_auc_score(label, output, average=average_dict[cfg.DATASET.NAME], multi_class='ovo')
        loss = loss / len(data_loader)

        writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
        writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
        writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc, epoch)
        writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)
        logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format(val_type, epoch, loss, acc, auc, f1))

    model.train()
    return auc, loss

