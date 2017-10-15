import find_mxnet
import mxnet as mx
import logging
import numpy as np

def monitor_stats(d):
    dd = d.asnumpy()
    stats = mx.nd.zeros((4,), d.context)
    stats[:] = np.array([dd.mean(), dd.std(), dd.min(), dd.max()])
    return stats # [mx.nd.min(d), mx.nd.max(d)]

def load_params_from_file(fname):
    save_dict = mx.ndarray.load(fname)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            arg_params[name] = value
        elif arg_type == 'aux':
            aux_params[name] = value
        else:
            raise ValueError("Invalid param file " + fname)
    return arg_params, aux_params

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

    # logging arguments and machine info
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

    # data
    (train, val) = data_loader(args, kv)

    # logging network summary
    logging.info('\n'+mx.viz.print_summary(network, shape=dict(train.provide_data), return_str=True))

    # epoch_size (for begin_num_update and lr_scheduler step)
    model_args = {}
    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    # load model (resume, finetune, or init)
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)

    if args.load_epoch is not None:
        assert model_prefix is not None
        param_file = '%s-%04d.params' % (model_prefix, args.load_epoch)
        arg_params, aux_params = load_params_from_file(param_file)
        logging.info('loading from %s', param_file)
        model_args = {'arg_params' : arg_params,
                      'aux_params' : aux_params,
                      'begin_epoch' : args.load_epoch,
                      'begin_num_update' : epoch_size * args.load_epoch}
        # TODO: check epoch_size for 'dist_sync'
    elif args.finetune_from is not None:
        # load_epoch has higher priority than finetune_from
        assert args.finetune_from.endswith('.params')
        arg_params, aux_params = load_params_from_file(args.finetune_from)
        logging.info('finetuning from %s', args.finetune_from)
        model_args = {'arg_params' : arg_params,
                      'aux_params' : aux_params}
    else:
        model_args = {'arg_params' : None,
                      'aux_params' : None}

    # save model
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix, args.checkpoint_epoch)

    # devices
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    # lr_scheduler
    if 'lr_factor' in args and args.lr_factor < 1:
        lr_factor_epoch = [float(_fe) for _fe in args.lr_factor_epoch.split(',')]
        if len(lr_factor_epoch) == 1:
            model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                step = max(int(epoch_size * lr_factor_epoch[0]), 1),
                factor = args.lr_factor,
                slow_step = int(args.lr_slow_epoch * epoch_size))
        else:
            model_args['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
                step = [max(int(epoch_size * _fe), 1) for _fe in lr_factor_epoch],
                factor = args.lr_factor,
                slow_step = int(args.lr_slow_epoch * epoch_size))

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # initializer
    if not args.initializer in ['xavier', 'xavier-gaussian', 'msra', 'default']:
        args.initializer = eval(args.initializer)
    initializer = mx.initializer.get_initializer(args.initializer)

    # monitor
    if args.monitor:
        mon = mx.mon.Monitor(args.display, monitor_stats, pattern=args.monitor, sort=True)
    else:
        mon = None

    # train
    mod = mx.module.Module(
        symbol              = network,
        label_names         = ['softmax_label'],
        context             = devs)

    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)

    mod.init_params(
        initializer         = initializer,
        arg_params          = model_args['arg_params'],
        aux_params          = model_args['aux_params'],
        allow_missing       = True)

    if args.optimizer in ['sgd', 'nag']:
        mod.init_optimizer(
            kvstore             = kv,
            optimizer           = args.optimizer,
            optimizer_params    = {
                'learning_rate':    args.lr,
                'momentum':         args.momentum,
                'wd':               args.wd,
                'lr_scheduler':     model_args['lr_scheduler'],
                'clip_gradient':    model_args['clip_gradient'],
                'begin_num_update': model_args['begin_num_update'] if 'begin_num_update' in model_args else 0,
            })
    else:
        # no momentum
        mod.init_optimizer(
            kvstore             = kv,
            optimizer           = args.optimizer,
            optimizer_params    = {
                'learning_rate':    args.lr,
                'wd':               args.wd,
                'lr_scheduler':     model_args['lr_scheduler'],
                'clip_gradient':    model_args['clip_gradient'],
                'begin_num_update': model_args['begin_num_update'] if 'begin_num_update' in model_args else 0,
            })

    mod.fit(
        train_data          = train,
        eval_data           = val,
        monitor             = mon,
        eval_epoch          = args.eval_epoch,
        eval_initialization = args.eval_initialization,
        eval_metric         = args.eval_metric.split(','),
#        clip_gamma          = args.clip_gamma,
        batch_end_callback  = [mx.callback.Speedometer(args.batch_size, args.display)],
        epoch_end_callback  = [checkpoint],
        num_epoch           = args.num_epochs,
        begin_epoch         = args.load_epoch if args.load_epoch else 0)

    logging.info('Optimization done.')
