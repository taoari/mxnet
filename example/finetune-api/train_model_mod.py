import find_mxnet
import mxnet as mx
import logging
import os
import numpy as np
from collections import OrderedDict

def monitor_stats(d):
    dd = d.asnumpy()
    stats = mx.nd.zeros((4,), d.context)
    stats[:] = np.array([dd.mean(), dd.std(), dd.min(), dd.max()])
    return stats # [mx.nd.min(d), mx.nd.max(d)]

class ConstantInitializer(mx.initializer.Initializer):
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.nd.NDArray):
            raise TypeError('arr must be NDArray')

        logging.info('Init (Constant) %s with value %s', name, self.value)
        arr[:] = self.value

def init_logger(log_file, head='%(asctime)-15s] %(message)s'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # auto time stamp the log file name
    if log_file == 'auto':
        from datetime import datetime
        log_file = str(datetime.now()).replace(' ', 'T').replace(':', '-') + '.log.txt'

    # create debug file handler and set level to debug
    handler = logging.FileHandler(log_file, "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def fit(args, network, data_loader):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging (Tao: auto logging and machine info)
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        init_logger(args.log_file, head)
        logging.info('Start with arguments:')
        for k, v in args._get_kwargs():
            logging.info('    %s: %s', k, v)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('Start with arguments:')
        for k, v in args._get_kwargs():
            logging.info('    %s: %s', k, v)

    try:
        import socket
        logging.info('Running on machine: %s', socket.gethostname())
    except Exception:
        pass

    # load model (Tao: resume or finetune)
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}

    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        logging.info('loading from %s-%04d.params', model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}

        # TODO: check epoch_size for 'dist_sync'
        epoch_size = args.num_examples / args.batch_size
        model_args['begin_num_update'] = epoch_size * args.load_epoch

    elif args.finetune_from is not None:
        # load_epoch has higher priority than finetune_from
        assert args.finetune_from.endswith('.params')
        finetune_from_prefix, finetune_from_epoch = args.finetune_from[:-len('.params')].rsplit('-', 1)
        finetune_from_epoch = int(finetune_from_epoch)
        logging.info('finetuning from %s', args.finetune_from)
        tmp = mx.model.FeedForward.load(finetune_from_prefix, finetune_from_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params}

    # save model (Tao: checkpoint with checkpoint_epoch)
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix, args.checkpoint_epoch)

    # data
    (train, val) = data_loader(args, kv)

    logging.info('\n'+mx.viz.print_summary(network, shape=dict(train.provide_data), return_str=True))

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    # lr_scheduler
    if 'lr_factor' in args and args.lr_factor < 1:
        lr_factor_epoch = [float(_fe) for _fe in args.lr_factor_epoch.split(',')]
        if len(lr_factor_epoch) == 1:
            model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                step = max(int(epoch_size * lr_factor_epoch[0]), 1),
                factor = args.lr_factor)
        else:
            model_args['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
                step = [max(int(epoch_size * _fe), 1) for _fe in lr_factor_epoch],
                factor = args.lr_factor,
                slow_step = int(args.lr_slow_epoch * epoch_size)) # TODO: lr_slow_epoch only support for MultiFactorScheduler now

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    # initialization
    def get_initializer(key):
        # 'xavier', 'msra', 'default'
        if key == 'xavier':
            return mx.init.Xavier(factor_type="in", magnitude=3.0)
        elif key == 'xavier-gaussian':
            return mx.init.Xavier(factor_type="in", rnd_type="gaussian", magnitude=1.0)
        elif key == 'msra':
            return mx.init.Xavier(factor_type="in", rnd_type="gaussian", magnitude=2.0)
        elif key == 'default':
            return mx.init.Xavier(factor_type="in", magnitude=2.34)
        # 'normal', 'uniform', 'const'
        elif key.startswith('normal'):
            return mx.init.Normal(sigma=float(key[len('normal'):]))
        elif key.startswith('uniform'):
            return mx.init.uniform(scale=float(key[len('uniform'):]))
        elif key.startswith('const'):
            return ConstantInitializer(value=float(key[len('const'):]))
        else:
            raise ValueError('Invalid initializer: %s' % args.initializer)

    # initializer
    initializer = None
    if args.initializer in ['xavier', 'xavier-gaussian', 'msra', 'default']:
        initializer = get_initializer(args.initializer)
    else:
        args.initializer = OrderedDict(eval(args.initializer))
        keys = args.initializer.keys()
        patterns = keys + ['.*']
        initializers = [get_initializer(args.initializer[k]) for k in keys] + [mx.initializer.Initializer()]
        initializer = mx.initializer.Mixed(patterns, initializers)

    # monitor
    if args.monitor:
        mon = mx.mon.Monitor(args.display, monitor_stats, pattern=args.monitor, sort=True)
    else:
        mon = None

    mod = mx.module.Module(
        symbol              = network,
        label_names         = ['softmax_label'],
        context             = devs)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=initializer, arg_params=model_args['arg_params'] if 'arg_params' in model_args else None,
                    aux_params=model_args['aux_params'] if 'aux_params' in model_args else None,
                    allow_missing=True)
    mod.init_optimizer(kvstore=kv,
        optimizer=args.optimizer,
        optimizer_params={
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'wd': args.wd,
        'lr_scheduler': model_args['lr_scheduler'],
        'begin_num_update': model_args['begin_num_update'] if 'begin_num_update' in model_args else 0,
        'clip_gradient': model_args['clip_gradient'],
        })

    mod.fit(
        train_data         = train,
        eval_data          = val,
        monitor = mon,
#        eval_epoch         = args.eval_epoch,
#        eval_initialization = args.eval_initialization,
        eval_metric        = args.eval_metric.split(','),
#        clip_gamma         = args.clip_gamma,
        batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.display)],
        epoch_end_callback = [checkpoint],
        num_epoch          = args.num_epochs,
        begin_epoch        = args.load_epoch if args.load_epoch else 0)

    logging.info('Optimization done.')
