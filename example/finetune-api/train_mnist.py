import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model

def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train-images-idx3-ubyte')) or \
       (not os.path.exists('train-labels-idx1-ubyte')) or \
       (not os.path.exists('t10k-images-idx3-ubyte')) or \
       (not os.path.exists('t10k-labels-idx1-ubyte')):
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip")
        os.system("unzip -u mnist.zip; rm mnist.zip")
    os.chdir("..")

def get_mlp(dataset='mnist', num_classes=10):
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet(dataset='mnist', num_classes=10):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(name='fc2' if dataset == 'mnist' else 'fc2_%s' % dataset, data=tanh3, num_hidden=num_classes)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir
        if '://' not in args.data_dir:
            _download(args.data_dir)
        flat = False if len(data_shape) == 3 else True

        train           = mx.io.MNISTIter(
            image       = data_dir + "train-images-idx3-ubyte",
            label       = data_dir + "train-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            shuffle     = True,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

        if args.val_dataset:
            val = mx.io.MNISTIter(
                image       = data_dir + "t10k-images-idx3-ubyte",
                label       = data_dir + "t10k-labels-idx1-ubyte",
                input_shape = data_shape,
                batch_size  = args.batch_size,
                flat        = flat,
                num_parts   = kv.num_workers,
                part_index  = kv.rank)
        else:
            import logging
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None

        return (train, val)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--solver', type=str,
                        help = 'solver configuration file in yaml format')
#    parser.add_argument('--network', type=str, default='mlp',
#                        choices = ['mlp', 'lenet'],
#                        help = 'the cnn to use')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--finetune-from', type=str,
                        help="finetune from model")
    parser.add_argument('--log-file', type=str, default='auto',
                        help='the name of log file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from solver_proto import update_args
    args = update_args(args, args.solver)


    if args.network == 'mlp':
        data_shape = (784, )
        net = get_mlp(args.dataset, args.num_classes)
    else:
        data_shape = (1, 28, 28)
        net = get_lenet(args.dataset, args.num_classes)

    # train
    train_model.fit(args, net, get_iterator(data_shape))
