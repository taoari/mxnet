#!/usr/bin/env python
# -*- coding: utf-8 -*-

import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model

from dataset import RandomSkipResizeIter

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
        train = mx.io.ImageRecordIter(
            path_imgrec = os.path.abspath(os.path.join(args.data_dir, args.train_dataset)),
            mean_r      = 123.68,
            mean_g      = 116.779,
            mean_b      = 103.939,
            data_shape  = data_shape,
            batch_size  = args.batch_size,
            rand_crop   = True,
            rand_mirror = True,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        if args.random_skip_ratio > 0.0:
            train = RandomSkipResizeIter(train, skip_ratio=args.random_skip_ratio,
                size=int(args.num_examples/args.batch_size))

        if args.val_dataset:
            val = mx.io.ImageRecordIter(
                path_imgrec = os.path.abspath(os.path.join(args.data_dir, args.val_dataset)),
                mean_r      = 123.68,
                mean_g      = 116.779,
                mean_b      = 103.939,
                rand_crop   = False,
                rand_mirror = False,
                data_shape  = data_shape,
                batch_size  = args.batch_size,
                num_parts   = kv.num_workers,
                part_index  = kv.rank)
        else:
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None

        return (train, val)

    # train
    train_model.fit(args, net, get_iterator)
