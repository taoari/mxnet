import find_mxnet
import mxnet as mx
import numpy as np
import os
import warnings
import cv2
import random

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

    def __init__(self, data, label=None, batch_size=1, shuffle=False, shuffle_on_reset=False, last_batch_handle='pad',
        pad=0, random_mirror=False, data_shape=None, random_crop=False, mean_values=None, scale=None):
        # pylint: disable=W0201

        super(NDArraySimpleAugmentationIter, self).__init__(data, label, batch_size, shuffle, shuffle_on_reset, last_batch_handle)
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

    def __init__(self, data_iter, size, skip_ratio=0.5, reset_internal=False):
        super(RandomSkipResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None
        self.prev_batch = None
        self.skip_ratio = skip_ratio

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def __get_next(self):
        try:
            return self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            return self.data_iter.next()

    def iter_next(self):
        if self.cur == self.size:
            return False

        data, label = [], []
        if self.current_batch is None:
            # very first
            batch = self.__get_next()
            self.current_batch = mx.io.DataBatch(data=[mx.nd.empty(batch.data[0].shape)], label=[mx.nd.empty(batch.label[0].shape)])
            keep = np.random.rand(self.batch_size) > self.skip_ratio
            batch_data = batch.data[0].asnumpy()
            batch_label = batch.label[0].asnumpy()
            data.extend(batch_data[keep])
            label.extend(batch_label[keep])
        elif self.prev_batch is not None:
            # prev_batch
            batch_data, batch_label = self.prev_batch
            data.extend(batch_data)
            label.extend(batch_label)

        while len(data) < self.batch_size:
            batch = self.__get_next()
            keep = np.random.rand(self.batch_size) > self.skip_ratio
            batch_data = batch.data[0].asnumpy()
            batch_label = batch.label[0].asnumpy()
            data.extend(batch_data[keep])
            label.extend(batch_label[keep])

        if len(data) > self.batch_size:
            self.prev_batch = data[self.batch_size:], label[self.batch_size:]
        else:
            self.prev_batch = None
        self.current_batch.data[0][:] = np.asarray(data[:self.batch_size])
        self.current_batch.label[0][:] = np.asarray(label[:self.batch_size])

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

class RecordIter(mx.io.DataIter):
    def __init__(self, path_imgrec, data_shape, batch_size, compressed=True, offset_on_reset=False):
        super(RecordIter, self).__init__()
        self.path_imagerec = path_imgrec
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.compressed = compressed
        self.offset_on_reset = offset_on_reset

        self.record = mx.recordio.MXRecordIO(os.path.abspath(path_imgrec), 'r')
        self._data = None

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [('data', (self.batch_size,) + self.data_shape)]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [('softmax_label', (self.batch_size,))]

    def reset(self):
        self.record.reset()
        if self.offset_on_reset:
            # avoid data batch always starting from 0, (True if training, False if testing)
            for i in range(random.randint(0, self.batch_size-1)):
                self.record.read()

    def iter_next(self):
        # ensure that there are <batch_size> data left, otherwise return False
        self._data = []
        for i in range(self.batch_size):
            self._data.append(self.record.read())
            if self._data[-1] is None:
                return False
        return True

    def _aug_img(self, img):
        return img

    def _parse_data_label(self, _data):
        data = []
        label = []
        for d in _data:
            if self.compressed:
                header, img = mx.recordio.unpack_img(d) # img: BGR uint8 (H,W,C)
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    if len(img.shape) == 2:
                        warnings.warn('gray image encountered')
                        img = np.dstack([img,img,img])
                    elif len(img.shape) == 4:
                        warnings.warn('RGBA image encountered')
                        img = img[:,:,:3]
                img = img[:,:,::-1] # RGB
            else:
                header, img = mx.recordio.unpack(d)
                shape = np.fromstring(img, dtype=np.float32, count=3) # H,W,C
                shape = tuple([int(i) for i in shape])
                img = np.fromstring(img[12:], dtype=np.uint8).reshape(shape) # img: RGB uint8 (H,W,C)
            assert img.shape[2] == 3
            img = self._aug_img(np.float32(img))
            data.append(img.transpose(2,0,1)) # RGB uint8 (C,H,W)
            label.append(header.label)
        return data, label

    def next(self):
        if self.iter_next():
            data, label = self._parse_data_label(self._data)
            return mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)])
        else:
            raise StopIteration

class RecordSkipIter(RecordIter):
    def __init__(self, path_imgrec, data_shape, batch_size, compressed=True, offset_on_reset=False,
                 skip_ratio=0.0, epoch_size=None):
        super(RecordSkipIter, self).__init__(path_imgrec, data_shape, batch_size, compressed, offset_on_reset)
        self.skip_ratio = skip_ratio
        self.epoch_size = epoch_size
        self.cur = 0

        assert skip_ratio >= 0.0 and skip_ratio < 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if skip_ratio > 0.0:
                assert self.epoch_size is not None and self.epoch_size > 0
                if self.offset_on_reset:
                    warnings.warn('when skip_ratio > 0.0, offset_on_reset is automatically disabled')
                    self.offset_on_reset = False
            else:
                # assert self.size is None # size is not support by RecordIter
                if self.epoch_size is not None:
                    warnings.warn('when skip_ratio == 0.0, size is not used')
                    self.epoch_size = None


    def reset(self):
        if self.skip_ratio == 0.0:
            super(RecordSkipIter, self).reset()
        else:
            self.cur = 0

    def iter_next(self):
        if self.skip_ratio == 0.0:
            # do not forget return
            return super(RecordSkipIter, self).iter_next()
        else:
            if self.cur < self.epoch_size:
                self._data = []
                while len(self._data) < self.batch_size:
                    s = self.record.read()
                    if s is None:
                        self.record.reset()
                        s = self.record.read()
                    # logic: if skip_ratio == 0, no drop
                    if random.random() >= self.skip_ratio:
                        self._data.append(s)
                self.cur += 1
                return True
            else:
                return False

def get_min_size(height, width, size):
    if height >= width:
        return int(height*size/width), size
    else:
        return size, int(width*size/height)

class RecordSimpleAugmentationIter(RecordSkipIter):
    def __init__(self, path_imgrec, data_shape, batch_size, compressed=True, offset_on_reset=False,
                 skip_ratio=0.0, epoch_size=None,
                 random_mirror=False, random_crop=False, mean_values=None, scale=None, pad=0,
                 min_size=0, max_size=0):
        super(RecordSimpleAugmentationIter, self).__init__(path_imgrec, data_shape, batch_size, compressed, offset_on_reset, skip_ratio, epoch_size)
        self.random_mirror=random_mirror
        self.random_crop=random_crop
        self.mean_values=mean_values
        self.scale = scale
        self.pad = pad
        self.min_size = min_size
        self.max_size = max_size
        if max_size > 0:
            assert max_size >= min_size

    def _aug_img(self, img):
        # assume img RGB float32 (H,W,C)
        # pad
        if self.pad > 0:
            img = cv2.copyMakeBorder(img,self.pad,self.pad,self.pad,self.pad,cv2.BORDER_REFLECT_101)
        # resize (multi-scale): min_size and max_size
        size = None
        if self.min_size > 0 and self.max_size > 0:
            size = random.randint(self.min_size, self.max_size)
        elif self.min_size > 0:
            size = self.min_size
        if size:
            # NOTE: OpenCV use (width, height), while Numpy use (height, width)
            _h, _w = get_min_size(img.shape[0], img.shape[1], size)
            img = cv2.resize(img, (_w,_h))
        # random_crop
        if self.random_crop:
            _c_y = random.randint(0, img.shape[0]-self.data_shape[1])
            _c_x = random.randint(0, img.shape[1]-self.data_shape[2])
        else:
            _c_y = (img.shape[0]-self.data_shape[1])/2
            _c_x = (img.shape[1]-self.data_shape[2])/2
        assert img.shape[0] >= self.data_shape[1] and img.shape[1] >= self.data_shape[2]
        img = img[_c_y:_c_y+self.data_shape[1], _c_x:_c_x+self.data_shape[2],:]
        # random_mirror
        if self.random_mirror and random.randint(0,1):
            img = img[:,::-1,:] # flip on x axis
        # mean_values
        if self.mean_values:
            img -= np.array(self.mean_values, dtype=np.float32)
        # scale
        if self.scale and self.scale != 1.0:
            img *= self.scale
        assert img.shape == (self.data_shape[1], self.data_shape[2], self.data_shape[0])
        return img
