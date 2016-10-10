import argparse
import yaml

def proto_parser():
    parser = argparse.ArgumentParser(description='train an image classifer on imagenet')

    # dataset
    group_dataset = parser.add_argument_group('options for dataset')
    group_dataset.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset')
    group_dataset.add_argument('--num-examples', type=int, default=1281167,
                        help='the number of training examples')
    group_dataset.add_argument('--num-classes', type=int, default=1000,
                        help='the number of classes')
    group_dataset.add_argument('--batch-size', type=int, default=32,
                        help='the batch size')
    group_dataset.add_argument('--epoch-size', type=int,
                        help='the epoch size, if not set, default to num_examples/batch_size')
    # dataset location
    group_dataset.add_argument('--data-dir', type=str,
                        help='the input data directory')
    group_dataset.add_argument('--train-dataset', type=str, default="train.rec",
                        help='train dataset name')
    group_dataset.add_argument('--val-dataset', type=str, default="val.rec",
                        help="validation dataset name")
    group_dataset.add_argument('--data-shape', type=str, default='3,224,224',
                        help='set image\'s shape')

    # dataset preprocessing
    group_dataset.add_argument('--aug-level', type=int, default=0,
                        help='augmetation level, 0,1: random_crop and random_mirror, 2: multiscale, 3: aspect ratio, 4: color.')
    group_dataset.add_argument('--mean-values', type=str,
                        help='RGB mean values to substract e.g. [123,117,104] or [123.68,116.779,103.939] (for cifar10)')
    group_dataset.add_argument('--scale', type=float, default=1.0,
                        help='multiply scale for mean substracted images (for cifar10)')
    group_dataset.add_argument('--pad', type=int, default=0,
                        help='pad extra pixels for data augmentation (for cifar10)')
    group_dataset.add_argument('--skip-ratio', type=float, default=0,
                        help='random skip ratio in [0,1), if 0 no random skip (for mnist, imagenet)')
    group_dataset.add_argument('--encoding', type=str, default='.jpg', choices=['.raw', '.jpg', '.png'],
                        help='encoding for the record file')
    group_dataset.add_argument('--min-size', type=int, default=0,
                        help='minimum image size for the shorter edge')
    group_dataset.add_argument('--max-size', type=int, default=0,
                        help='maximum image size for the shorter edge, for multi-scale augmentation')
    group_dataset.add_argument('--random-aspect-ratio', type=float, default=0.0,
                        help='random aspect ratio for augmentation, in [0.0,1.0].')
    group_dataset.add_argument('--random-hls', type=str,
                        help='random HLS for color jittering.')
    group_dataset.add_argument('--lighting-pca-noise', type=float, default=0.0,
                        help='lighting PCA color augmentation.')

    # optimizer and lr scheduler
    group_opt = parser.add_argument_group('options for optimizer and lr scheduler')
    group_opt.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    group_opt.add_argument('--lr', type=float, default=.01,
                        help='the initial learning rate')
    group_opt.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    group_opt.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay')
    group_opt.add_argument('--clip-gradient', type=float, default=5.,
                        help='clip min/max gradient to prevent extreme value')
    group_opt.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    group_opt.add_argument('--lr-factor-epoch', type=str, default='1.',
                        help='the numbers of epoch to factor the lr, could be a float or comma seperated floats')
    group_opt.add_argument('--lr-slow-epoch', type=float, default=0,
                        help='the number of epoch for slow start')

    # display, evaluation, checkpoint frequency
    group_freq = parser.add_argument_group('options for display, evaluation, checkpoint frequency')
    group_freq.add_argument('--display', type=int, default=50,
                        help='display speedometer per display iterations')
    group_freq.add_argument('--eval-metric', type=str, default='ce,acc',
                        help='evaluation metrics, comma separated list')
    group_freq.add_argument('--eval-epoch', type=int, default=1,
                        help='do evaluation every <eval-epoch> epochs')
    group_freq.add_argument('--eval-initialization', type=bool, default=True,
                        help='do evaluation at initial')
    group_freq.add_argument('--checkpoint-epoch', type=int, default=1,
                        help='do checkpoint every <checkpoint-epoch> epochs')
    group_freq.add_argument('--num-epochs', type=int, default=20,
                        help='the number of training epochs')

    # checkpoint
    group_cp = parser.add_argument_group('options for checkpoint')
    group_cp.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
#    group_cp.add_argument('--load-epoch', type=int,
#                        help="load the model on an epoch using the model-prefix")
    group_cp.add_argument('--finetune-from', type=str,
                        help="finetune from model")

    # misc
    parser.add_argument('--network', type=str, default='inception-bn',
                        help = 'the cnn to use')
    parser.add_argument('--network-kwargs', type=str, default='{}',
                        help = 'network symbol kwargs')
    parser.add_argument('--initializer', type=str, default='default',
                        help = 'the initializer to use, can be one of "xavier", "msra", "default" or {"pattern": value, ...} for mixed')
#    parser.add_argument('--gpus', type=str,
#                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--log-file', type=str, default='auto',
                        help='the name of log file')
    parser.add_argument('--monitor', type=str,
                        help='the monitor to install')
    group_freq.add_argument('--clip-gamma', type=bool, default=False,
                        help='clip gamma of prelu into [0,1]')

    return parser

def dict_to_arg_list(params):
    arg_list = []
    for k, v in params.items():
        k = k.replace('_', '-')
        if v is None:
            pass
        elif type(v) is bool:
            # pitfall for bool: --<key>='' is False, any non-empty string is True
            # e.g. --<key>=False and --<key>=None are still True
            if v == True:
                arg_list.extend(['--' + k, 'True'])
            else:
                arg_list.extend(['--' + k, ''])
        elif type(v) in [str, float, int]:
            arg_list.extend(['--' + k, str(v)])
        else:
            raise ValueError("Invalid parameter: %s=%s" % (k, v))

    return arg_list

def parse_args_from_file(solver_yml):
    parser = proto_parser()
    if solver_yml is not None:
        with open(solver_yml) as f:
            solver_params = yaml.load(f)
        args = parser.parse_args(dict_to_arg_list(solver_params),
            namespace=argparse.Namespace(**{k:v for k,v in solver_params.items() if v is None}))
    else:
        args = parser.parse_args([])
    return args

def namespace_update(ns1, ns2, exceptions=None):
    # make a copy
    dict1 = vars(ns1)
    dict1.update({k:v for k, v in vars(ns2).items() if k not in exceptions})
    ns = argparse.Namespace(**dict1)
    return ns

def update_args(args, solver_file):
    return namespace_update(parse_args_from_file(args.solver), args, exceptions=['solver'])

if __name__ == '__main__':
    # args = proto_parser().parse_args()
    # print (args)

    def parse_args():
        parser = argparse.ArgumentParser(description='train an image classifer on mnist')
        parser.add_argument('--solver', type=str, default='solver.yml',
                            help = 'solver configuration file in yaml format')
        parser.add_argument('--gpus', type=str,
                            help='the gpus will be used, e.g "0,1,2,3"')
        parser.add_argument('--load-epoch', type=int,
                            help="load the model on an epoch using the model-prefix")
        return parser.parse_args()

    args = parse_args()
    # print args

    args = update_args(args, args.solver)
    print(args)
