#!/usr/bin/env python
# -*- coding: utf-8 -*-

import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model
import numpy as np

def load_CIFAR10(ROOT):
    """ load all of cifar (from standford cs231n assignments) """

    import cPickle as pickle

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
        return X, Y

    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

class NDArraySimpleAugmentationIter(mx.io.NDArrayIter):
    """NDArrayIter object in mxnet. """

    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad',
        pad=0, random_mirror=False, data_shape=None, random_crop=False, mean_values=None):
        # pylint: disable=W0201

        super(NDArraySimpleAugmentationIter, self).__init__(data, label, batch_size, shuffle, last_batch_handle)
        self.pad = pad
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.data_shape = data_shape
        self.mean_values = mean_values

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(self.data_shape))) for k, v in self.data]
        
#    def next(self):
#        """Get next data batch from iterator. Equivalent to
#        self.iter_next()
#        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)
#
#        Returns
#        -------
#        data : DataBatch
#            The data of next batch.
#        """
#        if self.iter_next():
#            return DataBatch(data=self.getdata(), label=self.getlabel(), \
#                    pad=self.getpad(), index=self.getindex())
#        else:
#            raise StopIteration


    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """
        data = super(NDArraySimpleAugmentationIter, self).getdata() # (N,C,H,W)
        imgs = data[0].asnumpy().transpose(0,2,3,1) # (N,H,W,C)
        if self.mean_values:
            imgs = imgs - np.array(self.mean_values) # broading casting in channels
        processed_imgs = []
        
        if self.random_mirror:
            _m = np.random.randint(0,2,len(imgs))
        if self.random_crop:
            _c_y = np.random.randint(0,imgs.shape[1]+2*self.pad-self.data_shape[1]+1,len(imgs))
            _c_x = np.random.randint(0,imgs.shape[2]+2*self.pad-self.data_shape[2]+1,len(imgs))
        else:
            _c_y = (imgs.shape[1]+2*self.pad-self.data_shape[1])/2
            _c_x = (imgs.shape[2]+2*self.pad-self.data_shape[2])/2
        for i, img in enumerate(imgs):
            if self.pad > 0:
                import cv2
                img = cv2.copyMakeBorder(img,self.pad,self.pad,self.pad,self.pad,cv2.BORDER_REFLECT_101)
            if self.random_mirror and _m[i]:
                img = img[:,::-1,:] # flip on x axis
            if self.random_crop:
                img = img[_c_y[i]:_c_y[i]+self.data_shape[1], _c_x[i]:_c_x[i]+self.data_shape[2],:]
            else:
                img = img[_c_y:_c_y+self.data_shape[1], _c_x:_c_x+self.data_shape[2],:]
            processed_imgs.append(img)
            
        processed_imgs = np.asarray(processed_imgs).transpose(0,3,1,2) # (N,C,H,W)
        assert processed_imgs.shape[1:] == self.data_shape
        
        data = [mx.nd.empty(processed_imgs.shape, data[0].context)]
        data[0][:] = processed_imgs
            
        return data

    
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
        data_shape = (3, args.data_shape, args.data_shape)
        Xtr, Ytr, Xte, Yte = load_CIFAR10(args.data_dir)

        # TODO: data augmentation
        train = NDArraySimpleAugmentationIter(data = Xtr.transpose(0,3,1,2),
            label = Ytr,
            batch_size = args.batch_size,
            shuffle=True,
            pad=args.pad, random_mirror=True, data_shape=data_shape, random_crop=True, mean_values=[123.,117.,104.])
        # train = mx.io.ImageRecordIter(
        #     path_imgrec = os.path.abspath(os.path.join(args.data_dir, args.train_dataset)),
        #     preprocess_threads = 1,
        #     mean_r      = 123.68,
        #     mean_g      = 116.779,
        #     mean_b      = 103.939,
        #     data_shape  = data_shape,
        #     batch_size  = args.batch_size,
        #     rand_crop   = True,
        #     rand_mirror = True,
        #     num_parts   = kv.num_workers,
        #     part_index  = kv.rank)

        if args.val_dataset:
            val = NDArraySimpleAugmentationIter(data = Xte.transpose(0,3,1,2),
                label = Yte,
                batch_size = args.batch_size,
                shuffle=False,
                pad=args.pad, random_mirror=False, data_shape=data_shape, random_crop=False, mean_values=[123.,117.,104.])
            # val = mx.io.ImageRecordIter(
            #     path_imgrec = os.path.abspath(os.path.join(args.data_dir, args.val_dataset)),
            #     preprocess_threads = 1,
            #     mean_r      = 123.68,
            #     mean_g      = 116.779,
            #     mean_b      = 103.939,
            #     rand_crop   = False,
            #     rand_mirror = False,
            #     data_shape  = data_shape,
            #     batch_size  = args.batch_size,
            #     num_parts   = kv.num_workers,
            #     part_index  = kv.rank)
        else:
            import logging
            logging.info('Valication dataset is not provided, hence evaluation is disabled.')
            val = None
    
        return (train, val)
    
    # train
    train_model.fit(args, net, get_iterator)
