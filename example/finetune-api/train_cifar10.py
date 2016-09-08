#!/usr/bin/env python
# -*- coding: utf-8 -*-

import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model

from dataset import load_CIFAR10, NDArraySimpleAugmentationIter

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
        Xtr, Ytr, Xte, Yte = load_CIFAR10(args.data_dir)
        mean_values = [float(v) for v in args.mean_values.split(',')] if args.mean_values else None
        scale = None if args.scale == 1.0 else args.scale

        train = NDArraySimpleAugmentationIter(data = Xtr.transpose(0,3,1,2),
            label = Ytr,
            batch_size = args.batch_size,
            shuffle=True, shuffle_on_reset=True,
            pad=args.pad, random_mirror=True, data_shape=data_shape, random_crop=True, mean_values=mean_values, scale=scale)

        if args.val_dataset:
            val = NDArraySimpleAugmentationIter(data = Xte.transpose(0,3,1,2),
                label = Yte,
                batch_size = args.batch_size,
                shuffle=False, shuffle_on_reset=False,
                pad=args.pad, random_mirror=False, data_shape=data_shape, random_crop=False, mean_values=mean_values, scale=scale)
        else:
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None

        return (train, val)

    # train
    train_model.fit(args, net, get_iterator)
