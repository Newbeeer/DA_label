import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from scipy.io import loadmat


class Dataset(data.Dataset):
    def __init__(self, src, tgt, r, iseval, dataratio=1.0):
        if r[0] == 1 or r[0] == 0:
            r[0] = int(r[0])
        if r[1] == 1 or r[1] == 0:
            r[1] = int(r[1])
        self.eval = iseval

        if src == 'mnist':
            print(f"dataset : mnist, file path : {f'data/mnist32_train_{r}.mat'}")
            data = loadmat(f'data/mnist32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'svhn':
            print(f"dataset : svhn, file path : {f'data/mnist32_train_{r}.mat'}")
            data = loadmat(f'data/svhn32_train_{r}.mat')
            self.datalist_svhn = [{
                'image': data['X'][..., ij],
                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
            } for ij in range(data['y'].shape[0]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_svhn
        elif src == 'mnistm':
            print(f"dataset : mnistm, file path : {f'data/mnistm32_train_{r}.mat'}")
            data = loadmat(f'data/mnistm32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'digits':
            print(f"dataset : digits, file path : {f'data/digits32_train_{r}.mat'}")
            data = loadmat(f'data/digits32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'signs':
            print(f"dataset : signs, file path : {f'data/signs32_train_{r}.mat'}")
            data = loadmat(f'data/signs32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'gtsrb':
            print(f"dataset : gtsrb, file path : {f'data/gtsrb32_train_{r}.mat'}")
            data = loadmat(f'data/gtsrb32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'cifar':
            print(f"dataset : cifar, file path : {f'data/cifar32_train_{r}.mat'}")
            data = loadmat(f'data/cifar32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist
        elif src == 'stl':
            print(f"dataset : stl, file path : {f'data/stl32_train_{r}.mat'}")
            data = loadmat(f'data/stl32_train_{r}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_src = self.datalist_mnist

        if tgt == 'mnist':
            print(f"dataset : mnist, file path : {f'data/mnist32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/mnist32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'svhn':
            print(f"dataset : svhn, file path : {f'data/svhn32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/svhn32_train_{r[::-1]}.mat')
            self.datalist_svhn = [{
                'image': data['X'][..., ij],
                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
            } for ij in range(data['y'].shape[0]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_svhn
        elif tgt == 'mnistm':
            print(f"dataset : mnistm, file path : {f'data/mnistm32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/mnistm32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'digits':
            print(f"dataset : digits, file path : {f'data/digits32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/digits32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'signs':
            print(f"dataset : signs, file path : {f'data/signs32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/signs32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'gtsrb':
            print(f"dataset : gtsrb, file path : {f'data/gtsrb32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/gtsrb32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'cifar':
            print(f"dataset : cifar, file path : {f'data/cifar32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/cifar32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist
        elif tgt == 'stl':
            print(f"dataset : stl, file path : {f'data/stl32_train_{r[::-1]}.mat'}")
            data = loadmat(f'data/stl32_train_{r[::-1]}.mat')
            self.datalist_mnist = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1]) if np.random.rand() <= dataratio]
            self.datalist_target = self.datalist_mnist

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize_32 = transforms.Resize(32)
        self.source_larger = len(self.datalist_src) > len(self.datalist_target)
        self.n_smallerdataset = len(self.datalist_target) if self.source_larger else len(self.datalist_src)

    def __len__(self):
        return np.maximum(len(self.datalist_src), len(self.datalist_target))

    def shuffledata(self):
        self.datalist_src = [self.datalist_src[ij] for ij in torch.randperm(len(self.datalist_src))]
        self.datalist_target = [self.datalist_target[ij] for ij in torch.randperm(len(self.datalist_target))]

    def __getitem__(self, index):

        index_src = index if self.source_larger else index % self.n_smallerdataset
        index_target = index if not self.source_larger else index % self.n_smallerdataset

        image_source = self.datalist_src[index_src]['image']
        #print("raw source:", image_source)
        image_source = self.totensor(image_source)
        image_source = self.normalize(image_source).float()
        image_target = self.datalist_target[index_target]['image']
        #print("raw tgt:", image_target)
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target).float()
        # print("source:", image_source, image_source.size())
        # print("target:", image_target, image_target.size())

        return image_source, self.datalist_src[index_src]['label'], image_target, self.datalist_target[index_target]['label']


class Dataset_eval(data.Dataset):
    def __init__(self, tgt, r):

        if tgt == 'mnist':
            print(f"dataset : mnist, file path : {f'data/mnist32_test_{r}.mat'}")
            data = loadmat(f'data/mnist32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'svhn':
            print(f"dataset : svhn, file path : {f'data/svhn32_test_{r}.mat'}")
            data = loadmat(f'data/svhn32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][..., ij],
                'label': int(data['y'][ij][0]) if int(data['y'][ij][0]) < 10 else 0
            } for ij in range(data['y'].shape[0])]
        elif tgt == 'mnistm':
            print(f"dataset : mnistm, file path : {f'data/mnistm32_test_{r}.mat'}")
            data = loadmat(f'data/mnistm32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'digits':
            print(f"dataset : digits, file path : {f'data/digits32_test_{r}.mat'}")
            data = loadmat(f'data/digits32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'signs':
            print(f"dataset : signs, file path : {f'data/signs32_test_{r}.mat'}")
            data = loadmat(f'data/signs32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'gtsrb':
            print(f"dataset : gtsrb, file path : {f'data/gtsrb32_test_{r}.mat'}")
            data = loadmat(f'data/gtsrb32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'cifar':
            print(f"dataset : cifar, file path : {f'data/cifar32_test_{r}.mat'}")
            data = loadmat(f'data/cifar32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]
        elif tgt == 'stl':
            print(f"dataset : stl, file path : {f'data/stl32_test_{r}.mat'}")
            data = loadmat(f'data/stl32_test_{r}.mat')
            self.datalist_target = [{
                'image': data['X'][ij],
                'label': int(data['y'][0][ij])
            } for ij in range(data['y'].shape[1])]

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.datalist_target)

    def __getitem__(self, index):

        image_target = self.datalist_target[index]['image']
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target).float()

        return image_target, self.datalist_target[index]['label']


def GenerateIterator(args, iseval=False):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size if not iseval else args.batch_size_eval,
        'shuffle': True,
        'num_workers': args.workers,
        'drop_last': True,
    }

    return data.DataLoader(Dataset(args.src, args.tgt, args.r, iseval), **params)


def GenerateIterator_eval(args):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size_eval,
        'num_workers': args.workers,
    }

    return data.DataLoader(Dataset_eval(args.tgt, args.r[::-1]), **params)
