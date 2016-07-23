import find_mxnet
import mxnet as mx
import logging
import os

def monitor_stats(d):
    dd = d.asnumpy()
    stats = mx.nd.zeros((4,), d.context)
    stats[:] = np.array([dd.mean(), dd.std(), dd.min(), dd.max()])
    return stats # [mx.nd.min(d), mx.nd.max(d)]

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
        logging.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

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

    if args.finetune_from is not None:
        assert args.load_epoch is None
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

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    # (Tao: MultiFactorScheduler support)
    if 'lr_factor' in args and args.lr_factor < 1:
        lr_factor_epoch = [float(_fe) for _fe in args.lr_factor_epoch.split(',')]
        if len(lr_factor_epoch) == 1:
            model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                step = max(int(epoch_size * lr_factor_epoch[0]), 1),
                factor = args.lr_factor)
        else:
            model_args['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
                step = [max(int(epoch_size * _f), 1) for _f in lr_factor_epoch],
                factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    initializer = None
    if args.initializer == 'xavier':
        initializer = mx.init.Xavier(factor_type="in", magnitude=3.0)
    elif args.initializer == 'msra':
        initializer = mx.init.Xavier(factor_type="in", rnd_type="gaussian", magnitude=2)
    elif args.initializer == 'old':
        initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    else:
        raise ValueError('Invalid initializer: %s' % args.initializer)

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        initializer        = initializer,
        learning_rate      = args.lr,
        momentum           = args.momentum,
        wd                 = args.wd,
        **model_args)
    
    # (Tao: eval_epoch support, eval_metric and display)
    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        monitor            = mon,
        eval_epoch         = args.eval_epoch,
        eval_metric        = ['accuracy', 'ce'],
        batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.display)],
        epoch_end_callback = [checkpoint])