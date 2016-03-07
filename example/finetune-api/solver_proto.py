import argparse
import yaml

def proto_parser():
    parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
    parser.add_argument('--network', type=str, default='inception-bn',
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str,
                        help='the input data directory')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--lr', type=float, default=.01,
                        help='the initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=str, default='1.',
                        help='the numbers of epoch to factor the lr, could be a float or comma seperated floats')
    parser.add_argument('--clip-gradient', type=float, default=5.,
                        help='clip min/max gradient to prevent extreme value')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='the number of training epochs')
#    parser.add_argument('--load-epoch', type=int,
#                        help="load the model on an epoch using the model-prefix")
#    parser.add_argument('--finetune-from', type=str,
#                        help="finetune from model")
    parser.add_argument('--finetune-lr-scale', type=float, default=10,
                        help="finetune layer lr_scale")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='the batch size')
#    parser.add_argument('--gpus', type=str,
#                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--num-examples', type=int, default=1281167,
                        help='the number of training examples')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='the number of classes')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset')
    parser.add_argument('--log-file', type=str, default='auto',
    		    help='the name of log file')
#    parser.add_argument('--log-dir', type=str, default="/tmp/",
#                        help='directory of the log file')
    parser.add_argument('--train-dataset', type=str, default="train.rec",
                        help='train dataset name')
    parser.add_argument('--val-dataset', type=str, default="val.rec",
                        help="validation dataset name")
    parser.add_argument('--data-shape', type=int, default=224,
                        help='set image\'s shape')
    parser.add_argument('--display', type=int, default=50,
                        help='display speedometer per display iterations')
    parser.add_argument('--eval-metric', type=str, default='acc,ce',
                        help='evaluation metrics, comma separated list')
    parser.add_argument('--checkpoint-epoch', type=int, default=1,
                        help='do checkpoint every <checkpoint-epoch> epochs')
    parser.add_argument('--eval-epoch', type=int, default=1,
                        help='do evaluation every <checkpoint-epoch> epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay')
    return parser
    
def dict_to_arg_list(params):
    arg_list = []
    for k, v in params.items():
        k = k.replace('_', '-')
        if v is None:
            pass
        elif type(v) is bool:
            if v == True:
                arg_list.append('--' + k)
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
    def parse_args():
        parser = argparse.ArgumentParser(description='train an image classifer on mnist')
        parser.add_argument('--solver', type=str, default='solver.yml',
                            help = 'solver configuration file in yaml format')
        parser.add_argument('--network', type=str, default='mlp',
                            choices = ['mlp', 'lenet'],
                            help = 'the cnn to use')
        parser.add_argument('--gpus', type=str,
                            help='the gpus will be used, e.g "0,1,2,3"')
        parser.add_argument('--load-epoch', type=int,
                            help="load the model on an epoch using the model-prefix")
        parser.add_argument('--finetune-from', type=str,
                            help="finetune from model")
        return parser.parse_args()
    
    args = parse_args()
    # print args
    
    args = update_args(args, args.solver)
    print(args)