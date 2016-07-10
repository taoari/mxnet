#!/usr/bin/env python
# -*- coding: utf-8 -*-

import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model
import numpy as np

def load_CIFAR10(ROOT):
    """ load all of cifar (from standford cs231n assignments) """

    import cPickle as pickle

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
        return X, Y

    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

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
    sys.path.append('.')
    net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes, args.dataset)
    
    # data
    def get_iterator(args, kv):
        data_shape = (3, args.data_shape, args.data_shape)
        Xtr, Ytr, Xte, Yte = load_CIFAR10(args.data_dir)

        # TODO: data augmentation
        train = mx.io.NDArrayIter(data = Xtr.transpose(0,3,1,2),
            label = Ytr,
            batch_size = args.batch_size,
            shuffle=True)
        # train = mx.io.ImageRecordIter(
        #     path_imgrec = os.path.abspath(os.path.join(args.data_dir, args.train_dataset)),
        #     preprocess_threads = 1,
        #     mean_r      = 123.68,
        #     mean_g      = 116.779,
        #     mean_b      = 103.939,
        #     data_shape  = data_shape,
        #     batch_size  = args.batch_size,
        #     rand_crop   = True,
        #     rand_mirror = True,
        #     num_parts   = kv.num_workers,
        #     part_index  = kv.rank)

        if args.val_dataset:
            val = mx.io.NDArrayIter(data = Xte.transpose(0,3,1,2),
                label = Yte,
                batch_size = args.batch_size,
                shuffle=False)
            # val = mx.io.ImageRecordIter(
            #     path_imgrec = os.path.abspath(os.path.join(args.data_dir, args.val_dataset)),
            #     preprocess_threads = 1,
            #     mean_r      = 123.68,
            #     mean_g      = 116.779,
            #     mean_b      = 103.939,
            #     rand_crop   = False,
            #     rand_mirror = False,
            #     data_shape  = data_shape,
            #     batch_size  = args.batch_size,
            #     num_parts   = kv.num_workers,
            #     part_index  = kv.rank)
        else:
            import logging
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None
    
        return (train, val)
    
    # train
    train_model.fit(args, net, get_iterator)
