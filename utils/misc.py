import sys, os, logging
from torch.utils.tensorboard import SummaryWriter   
from collections import OrderedDict
import shutil

def init_log(args, log_path, train_loader, dataset_size):
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))
    logging.basicConfig(filename=log_path + '/log.txt', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_loader)))
    logging.info("We have {} images in train set, and have {} images in test set.".format(dataset_size[0], dataset_size[1]))
    logging.info(str(args))
        
    return writer

def check_path(log_path, args):
    if os.path.isdir(log_path):
        if args.override:
            shutil.rmtree(log_path)
        else:
            if os.path.exists(os.path.join(log_path, 'done')):
                print('Already trained, exit')
                exit()
    else:
        os.makedirs(log_path)

def get_fast_grad_weights(fast_weights, optimizer, grads, lr, cfg):

    fast_weights_updated = OrderedDict()

    if cfg.OPTIM.WEIGHT_DECAY == 0:
        fast_weights_updated = OrderedDict((name, param - lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

    elif cfg.OPTIM.WEIGHT_DECAY > 0:
        fast_weights_updated = OrderedDict((name, param - lr * (grad + cfg.OPTIM.WEIGHT_DECAY * param)) for ((name, param), grad) in zip(fast_weights.items(), grads))

    else:
        raise ValueError('Parameters are not correctly set.')

    return fast_weights_updated
