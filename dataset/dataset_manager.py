from .dataset import EYEQ, DEEPDR, DRAC, IQAD_CXR, IQAD_CT
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(cfg):
        
    root = cfg.DATASET.ROOT
    batch_size = cfg.BATCH_SIZE
    dataset_name = cfg.DATASET.NAME

    train_ts, test_ts = get_transform(cfg)
    num_worker = min (batch_size // 4, 16)
    dataset = globals()[dataset_name]

    train_dataset = dataset(root=root, split = 'train', transform=train_ts)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers= num_worker)

    if cfg.DATASET.NAME in cfg.VALIDATION_DATASET:
        val_dataset = dataset(root=root, split = 'val', transform=test_ts)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)

    test_dataset = dataset(root=root, split = 'test', transform=test_ts)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=num_worker)

    dataset_size = [len(train_dataset), len(test_dataset)]
    
    if dataset_name in cfg.VALIDATION_DATASET:
        return train_loader, test_loader, val_loader, dataset_size
    else:
        return train_loader, test_loader, dataset_size

def get_transform(cfg):

    means = cfg.DATASET.NORMALIZATION_MEAN
    std = cfg.DATASET.NORMALIZATION_STD

    transfrom_train = []
    transfrom_test = []

    # if dataset in ['DRAC', 'IQAD_CXR', 'IQAD_CT']:
    #     transfrom_train.append(transforms.Grayscale(1))
    #     transfrom_test.append(transforms.Grayscale(1))

    transfrom_train.append(transforms.Resize((256, 256)))
    transfrom_test.append(transforms.Resize((256, 256)))

    transfrom_train.append(transforms.ToTensor())
    transfrom_train.append(transforms.Normalize(means, std))

    transfrom_test.append(transforms.ToTensor())
    transfrom_test.append(transforms.Normalize(means, std))

    train_ts =  transforms.Compose(transfrom_train)
    test_ts = transforms.Compose(transfrom_test)

    return train_ts, test_ts
