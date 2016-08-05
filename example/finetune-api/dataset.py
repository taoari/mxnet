import find_mxnet
import mxnet as mx
import numpy as np
import os

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
        pad=0, random_mirror=False, data_shape=None, random_crop=False, mean_values=None, scale=None):
        # pylint: disable=W0201

        super(NDArraySimpleAugmentationIter, self).__init__(data, label, batch_size, shuffle, last_batch_handle)
        self.pad = pad
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.data_shape = data_shape
        self.mean_values = mean_values
        self.scale = scale

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
        if self.scale:
            imgs = imgs * self.scale
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

class RandomSkipResizeIter(mx.io.DataIter):
    """Resize a DataIter to given number of batches per epoch.
    May produce incomplete batch in the middle of an epoch due
    to padding from internal iterator.

    Parameters
    ----------
    data_iter : DataIter
        Internal data iterator.
    max_random_skip : maximum random skip number
        If max_random_skip is 1, no random skip.
    size : number of batches per epoch to resize to.
    reset_internal : whether to reset internal iterator on ResizeIter.reset
    """

    def __init__(self, data_iter, size, max_random_skip=5, reset_internal=False):
        super(RandomSkipResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None
        self.max_random_skip = max_random_skip

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def __iter_next(self):
        try:
            self.current_batch = self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            self.current_batch = self.data_iter.next()

    def iter_next(self):
        if self.cur == self.size:
            return False

        random_skip = np.random.randint(0, self.max_random_skip, 1)[0] + 1
        for i in range(random_skip):
            self.__iter_next()

        self.cur += 1
        return True

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad