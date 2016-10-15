#!/usr/bin/env python
# -*- coding: utf-8 -*-

import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model

from dataiter import RecordSimpleAugmentationIter

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

    if args.aug_level > 0:
        # 0,1 for random_crop and random_mirror: always applied
        # 2: multiscale
        if args.aug_level >= 2:
            assert args.min_size > 0, 'min_size must be enabled for multiscale'
            args.max_size = 480
        # 3: aspect ratio
        if args.aug_level >= 3:
            args.random_aspect_ratio = 0.25 # aspect ratio jittering in [0.8,1.25]
        # 4: color aug
        if args.aug_level >= 4:
            args.random_hls = '0.4' # str

    # network
    import importlib
    import sys
    sys.path.insert(0, '.') # current folder first
    net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes, args.dataset, **eval(args.network_kwargs))

    # data
    def get_iterator(args, kv):
        data_shape = tuple([int(i) for i in args.data_shape.split(',')])
        mean_values = [float(v) for v in args.mean_values.split(',')] if args.mean_values else None
        scale = None if args.scale == 1.0 else args.scale
        random_hls = [float(v) for v in args.random_hls.split(',')] if args.random_hls else None
        compressed = True if args.encoding != '.raw' else False

        train = RecordSimpleAugmentationIter(os.path.abspath(os.path.join(args.data_dir, args.train_dataset)),
            data_shape, args.batch_size, compressed=compressed, offset_on_reset=True, num_thread=args.num_thread,
            epoch_size=int(args.num_examples/args.batch_size), skip_ratio=args.skip_ratio,
            random_mirror=True, random_crop=True, mean_values=mean_values, scale=scale, pad=args.pad,
            min_size=args.min_size, max_size=args.max_size, random_aspect_ratio=args.random_aspect_ratio,
            random_hls=random_hls, lighting_pca_noise=args.lighting_pca_noise)

        if args.val_dataset:
            val = RecordSimpleAugmentationIter(os.path.abspath(os.path.join(args.data_dir, args.val_dataset)),
                data_shape, args.batch_size, compressed=compressed, offset_on_reset=False, num_thread=args.num_thread,
                epoch_size=int(args.num_eval_examples/args.batch_size),
                random_mirror=False, random_crop=False, mean_values=mean_values, scale=scale, pad=0,
                min_size=args.min_size) # no pad, skip_ratio, epoch_size, max_size (multi-scale) for test
        else:
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None

        return (train, val)

    # train
    train_model.fit(args, net, get_iterator)
