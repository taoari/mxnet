#!/usr/bin/env python
# -*- coding: utf-8 -*-

import find_mxnet
import logging
import argparse
import train_model

from dataiter import NDArraySimpleAugmentationIter

def load_CIFAR10(root_dir='cifar-10-batches-py', return_classes=False):
    '''Load CIFAR 10 dataset.

    Returns
    -------
    Xtr : ndarray
        50000 32x32 RGB uint8 images.
    Ytr : ndarray
        Train labels.
    Xte : ndarray
        10000 32x32 RGB uint8 images.
    Yte : ndarray
        Test labels.
    classes : list of string (if return_classes=True)
        Image labels.
    '''

    import cPickle
    import numpy as np
    import os

    def unpickle(filename):
        with open(filename, 'rb') as f:
            dict = cPickle.load(f)
        return dict

    def load_CIFAR_batch(filename):
        train = unpickle(filename)
        X = train['data'].reshape((-1,3,32,32)).transpose(0,2,3,1) # 50000 32x32 RGB unit8 images
        y = np.array(train['labels'], dtype=np.int64)
        return X, y

    xs = []
    ys = []
    for b in range(1,6):
        X, Y = load_CIFAR_batch(os.path.join(root_dir, 'data_batch_%d' % b))
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    Xte, Yte = load_CIFAR_batch(os.path.join(root_dir, 'test_batch'))

    if not return_classes:
        return Xtr, Ytr, Xte, Yte
    else:
        meta = unpickle(os.path.join(root_dir, 'batches.meta'))
        classes = meta['label_names']
        return Xtr, Ytr, Xte, Yte, classes

def load_CIFAR100(root_dir='cifar-100-python', return_classes=False):
    '''Load CIFAR 100 dataset.

    Returns
    -------
    Xtr : ndarray
        50000 32x32 RGB uint8 images.
    Ytr : ndarray
        Train labels.
    Xte : ndarray
        10000 32x32 RGB uint8 images.
    Yte : ndarray
        Test labels.
    classes : list of string (if return_classes=True)
        Image labels.
    '''

    import cPickle
    import numpy as np
    import os

    def unpickle(filename):
        with open(filename, 'rb') as f:
            dict = cPickle.load(f)
        return dict

    def load_CIFAR_batch(filename):
        train = unpickle(filename)
        X = train['data'].reshape((-1,3,32,32)).transpose(0,2,3,1) # 50000 32x32 RGB unit8 images
        y = np.array(train['fine_labels'], dtype=np.int64)
        return X, y

    Xtr, Ytr = load_CIFAR_batch(os.path.join(root_dir, 'train'))
    Xte, Yte = load_CIFAR_batch(os.path.join(root_dir, 'test'))

    if not return_classes:
        return Xtr, Ytr, Xte, Yte
    else:
        meta = unpickle(os.path.join(root_dir, 'meta'))
        classes = meta['fine_label_names']
        return Xtr, Ytr, Xte, Yte, classes

# don't use -n and -s, which are resevered for the distributed training
def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--solver', type=str,
                        help = 'solver configuration file in yaml format')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--log-file', type=str, default='auto',
                        help='the name of log file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print args
    from solver_proto import update_args
    args = update_args(args, args.solver)

    # network
    import importlib
    import sys
    sys.path.insert(0, '.') # current folder first
    net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes, args.dataset, **eval(args.network_kwargs))

    # data
    def get_iterator(args, kv):
        data_shape = tuple([int(i) for i in args.data_shape.split(',')])
        if args.dataset == 'cifar10':
            assert args.num_classes == 10
            Xtr, Ytr, Xte, Yte = load_CIFAR10(args.data_dir)
        elif args.dataset == 'cifar100':
            assert args.num_classes == 100
            Xtr, Ytr, Xte, Yte = load_CIFAR100(args.data_dir)
        else:
            raise ValueError('Invalid dataset: %s', args.dataset)
        mean_values = [float(v) for v in args.mean_values.split(',')] if args.mean_values else None
        scale = None if args.scale == 1.0 else args.scale

        train = NDArraySimpleAugmentationIter(data = Xtr.transpose(0,3,1,2),
            label = Ytr,
            batch_size = args.batch_size,
            shuffle=True, shuffle_on_reset=True,
            pad=args.pad, random_mirror=True, data_shape=data_shape, random_crop=True,
            mean_values=mean_values, scale=scale, last_batch_handle='discard')

        if args.val_dataset:
            val = NDArraySimpleAugmentationIter(data = Xte.transpose(0,3,1,2),
                label = Yte,
                batch_size = args.batch_size,
                shuffle=False, shuffle_on_reset=False,
                pad=args.pad, random_mirror=False, data_shape=data_shape, random_crop=False,
                mean_values=mean_values, scale=scale, last_batch_handle='discard')
        else:
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None

        return (train, val)

    # train
    train_model.fit(args, net, get_iterator)
