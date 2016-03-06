import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model

# don't use -n and -s, which are resevered for the distributed training
def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--solver', type=str,
                        help = 'solver configuration file in yaml format')
    parser.add_argument('--network', type=str, default='inception-bn',
                        choices = ['alexnet', 'vgg', 'googlenet', 'inception-bn', 'inception-bn-full', 'inception-v3', 'vgg16'],
                        help = 'the cnn to use')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--finetune-from', type=str,
                        help="finetune from model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print args
    from solver_proto import update_args
    args = update_args(args, args.solver)

    # network
    import importlib
    net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes, args.dataset)
    
    # data
    def get_iterator(args, kv):
        data_shape = (3, args.data_shape, args.data_shape)
        train = mx.io.ImageRecordIter(
            path_imgrec = os.path.join(args.data_dir, args.train_dataset),
            mean_r      = 123.68,
            mean_g      = 116.779,
            mean_b      = 103.939,
            data_shape  = data_shape,
            batch_size  = args.batch_size,
            rand_crop   = True,
            rand_mirror = True,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)
    
        val = mx.io.ImageRecordIter(
            path_imgrec = os.path.join(args.data_dir, args.val_dataset),
            mean_r      = 123.68,
            mean_g      = 116.779,
            mean_b      = 103.939,
            rand_crop   = False,
            rand_mirror = False,
            data_shape  = data_shape,
            batch_size  = args.batch_size,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)
    
        return (train, val)
    
    # train
    train_model.fit(args, net, get_iterator)
