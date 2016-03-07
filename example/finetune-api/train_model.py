import find_mxnet
import mxnet as mx
import logging
import os

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

    # logging
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

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}

    if args.finetune_from is not None:
        assert args.load_epoch is None
        finetune_from_prefix, finetune_from_epoch = args.finetune_from.rsplit('-', 1)
        finetune_from_epoch = int(finetune_from_epoch)
        logger.info('finetune from %s at epoch %d', finetune_from_prefix, finetune_from_epoch)
        tmp = mx.model.FeedForward.load(finetune_from_prefix, finetune_from_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params} 

    # save model
    from mxnet.model import save_checkpoint
    
    def do_checkpoint(prefix, frequent=1):
        """Callback to checkpoint the model to prefix every epoch.
    
        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
    
        Returns
        -------
        callback : function
            The callback function that can be passed as iter_end_callback to fit.
        """
        def _callback(iter_no, sym, arg, aux):
            """The checkpoint function."""
            if (iter_no + 1) % frequent == 0:
                save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        return _callback
        
    checkpoint = None if model_prefix is None else do_checkpoint(model_prefix, args.checkpoint_epoch if 'checkpoint_epoch' in args else 1)
    
    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

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

    # optimizer
    batch_size = args.batch_size
    # reference: model.FeedForward.fit()
    if kv and kv.type == 'dist_sync':
        batch_size *= kv.num_workers
    optimizer = mx.optimizer.create('sgd',
        rescale_grad=(1.0/batch_size),
        learning_rate      = args.lr,
        momentum           = args.momentum if 'momentum' in args else 0.0,
        wd                 = args.wd if 'wd' in args else 0.00001,
        clip_gradient = model_args['clip_gradient'] if 'clip_gradient' in model_args else None,
        lr_scheduler = model_args['lr_scheduler'] if 'lr_scheduler' in model_args else None,
        arg_names = network.list_arguments())

    # lr_scale
    if args.finetune_from is not None:
        # convention: for argument param_name, if args.dataset in param_name, then it is
        # to be fine-tuned
        lr_scale = {}
        net_args = network.list_arguments()
        for i, name in enumerate(net_args):
            if args.dataset in name:
                lr_scale[i] = args.finetune_lr_scale
        optimizer.set_lr_scale(lr_scale)
        logger.info('lr_scale: %s', {net_args[i]: s for i,s in lr_scale.items()})

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        # learning_rate      = args.lr,
        # momentum           = 0.9,
        # wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer          = optimizer,
        **model_args)

    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        eval_epoch         = args.eval_epoch if 'eval_epoch' in args else 1,
        eval_metric        = args.eval_metric.split(',') if 'eval_metric' in args else 'acc',
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.display if 'display' in args else 50),
        epoch_end_callback = checkpoint)
