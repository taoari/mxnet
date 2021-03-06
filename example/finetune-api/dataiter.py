import find_mxnet
import mxnet as mx
import numpy as np
import os
import warnings
import cv2
import random

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

def _decode_data(buf, compressed):
    if compressed:
        header, img = mx.recordio.unpack_img(buf) # img: BGR uint8 (H,W,C)
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            if len(img.shape) == 2:
                warnings.warn('gray image encountered')
                img = np.dstack([img,img,img])
            elif img.shape[2] == 4:
                warnings.warn('BGRA image encountered')
                img = img[:,:,:3]
        img = img[:,:,::-1] # RGB
    else:
        header, img = mx.recordio.unpack(buf)
        shape = np.fromstring(img, dtype=np.float32, count=3) # H,W,C
        shape = tuple([int(i) for i in shape])
        img = np.fromstring(img[12:], dtype=np.uint8).reshape(shape) # img: RGB uint8 (H,W,C)
    assert img.shape[2] == 3
    return img, header.label

class BaseRecordIter(mx.io.DataIter):
    def __init__(self, path_imgrec, data_shape, batch_size, compressed=True):
        super(BaseRecordIter, self).__init__()

        self.path_imagerec = path_imgrec
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.compressed = compressed

        if path_imgrec:
            self.record = mx.recordio.MXRecordIO(os.path.abspath(path_imgrec), 'r')
        else:
            self.record = None
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

    def iter_next(self):
        self._data = []
        for i in range(self.batch_size):
            self._data.append(self.record.read())
            if self._data[-1] is None:
                return False
        return True

    def _aug_img(self, img):
        '''Called inside _parse_data_label.'''
        return img

    def _parse_data_label(self, _data):
        data = []
        label = []
        for buf in _data:
            img, lab = _decode_data(buf, self.compressed)
            img = self._aug_img(np.float32(img))
            img = img.transpose(2,0,1) # RGB uint8 (C,H,W)
            data.append(img)
            label.append(lab)
        return data, label

    def next(self):
        if self.iter_next():
            data, label = self._parse_data_label(self._data)
            return mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)])
        else:
            raise StopIteration

class RecordIter(BaseRecordIter):
    def __init__(self, path_imgrec, data_shape, batch_size, compressed=True,
            offset_on_reset=False, epoch_size=0, skip_ratio=0.0):
        super(RecordIter, self).__init__(path_imgrec, data_shape, batch_size, compressed)

        self.offset_on_reset = offset_on_reset
        self.epoch_size = epoch_size
        self.skip_ratio = skip_ratio
        if skip_ratio > 0.0:
            assert epoch_size > 0, 'epoch size must be postive when skip_ratio is enabled'

        self.cur = 0

    def _offset(self):
        if self.offset_on_reset:
            # avoid data batch always starting from 0, (True if training, False if testing)
            for i in range(random.randint(0, self.batch_size-1)):
                self.record.read()

    def reset(self):
        if not self.epoch_size:
            self.record.reset()
            self._offset()
        else:
            self.cur = 0

    def iter_next(self):
        if not self.epoch_size:
            # ensure that there are <batch_size> data left, otherwise return False
            self._data = []
            for i in range(self.batch_size):
                self._data.append(self.record.read())
                if self._data[-1] is None:
                    return False
            return True
        else:
            if self.cur < self.epoch_size:
                self._data = []
                while len(self._data) < self.batch_size:
                    s = self.record.read()
                    if s is None:
                        self.record.reset()
                        self._offset()
                        s = self.record.read()
                    # logic: if skip_ratio == 0, no drop
                    if self.skip_ratio == 0.0 or random.random() >= self.skip_ratio:
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

def _aug_hls(img, random_hls):
    assert img.dtype == np.uint8, '_aug_hls is only valid for uint8 images'
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    delta = np.random.uniform(-1.0,1.0,3) * np.array([90,127,127]) * np.array(random_hls)
    # uint8 array + float64 scalar gives float64 array
    hls[:,:,0] = (hls[:,:,0] + delta[0]) % 180 # circular hue
    hls[:,:,1] = np.clip(hls[:,:,1] + delta[1], 0.0, 255)
    hls[:,:,2] = np.clip(hls[:,:,2] + delta[2], 0.0, 255)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def _aug_lighting(src, alphastd):
    # TODO: need to check
    eigval = np.array([55.46, 4.794, 1.148])
    eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948, 0.4203]])
    alpha = np.random.normal(0, alphastd, size=(3,))
    rgb = np.dot(eigvec*alpha, eigval)
    src = src + np.array(rgb)
    return src

def _aug_img(img, data_shape, random_mirror=False, random_crop=False, mean_values=None, scale=None, pad=0,
    min_size=0, max_size=0, random_aspect_ratio=0.0,
    random_hls=None, lighting_pca_noise=0.0):
    # input RGB uint8 (H,W,C) image, output RGB float32 (H,W,C) image
    # data_shape is in (C,H,W)
    ### valid only for uint8 images
    # color aug (should before mean_values and scale)
    if random_hls is not None:
        img = _aug_hls(img, random_hls)
    if lighting_pca_noise > 0.0:
        img = _aug_lighting(img, lighting_pca_noise)
    ### valid for both uint8 and float32 images
    # pad
    if pad > 0:
        img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    # resize: random_aspect_ratio
    _h, _w = img.shape[:2]
    if random_aspect_ratio > 0.0:
        _ar = 1.0+random_aspect_ratio
        aspect_ratio = random.uniform(1.0/_ar, _ar)
        # image is always larger in size
        if aspect_ratio > 1.0:
            _h = int(np.round(_h*aspect_ratio))
        else:
            _w = int(np.round(_w/aspect_ratio))
    # resize (multi-scale): min_size and max_size
    size = None
    if min_size > 0 and max_size > 0:
        size = random.randint(min_size, max_size)
    elif min_size > 0:
        size = min_size
    if size:
        # NOTE: OpenCV use (width, height), while Numpy use (height, width)
        _h, _w = get_min_size(_h, _w, size)
    if _h != img.shape[0] or _w != img.shape[1]:
        img = cv2.resize(img, (_w,_h))
    # random_crop
    if random_crop:
        _c_y = random.randint(0, img.shape[0]-data_shape[1])
        _c_x = random.randint(0, img.shape[1]-data_shape[2])
    else:
        _c_y = int((img.shape[0]-data_shape[1])/2)
        _c_x = int((img.shape[1]-data_shape[2])/2)
    assert img.shape[0] >= data_shape[1] and img.shape[1] >= data_shape[2]
    img = img[_c_y:_c_y+data_shape[1], _c_x:_c_x+data_shape[2],:]
    # random_mirror
    if random_mirror and random.randint(0,1):
        img = img[:,::-1,:] # flip on x axis
    ### valid only for float32 images
    img = np.float32(img)
    # mean_values
    if mean_values is not None:
        img -= np.array(mean_values, dtype=np.float32)
    # scale
    if scale and scale != 1.0:
        img *= scale
    assert img.shape == (data_shape[1], data_shape[2], data_shape[0])
    return img

def _proc_fun(buf, compressed, data_shape,
    random_mirror=False, random_crop=False, mean_values=None, scale=None, pad=0,
    min_size=0, max_size=0, random_aspect_ratio=0.0,
    random_hls=None, lighting_pca_noise=0.0):

    img, lab = _decode_data(buf, compressed)
    img = _aug_img(img, data_shape,
        random_mirror, random_crop, mean_values, scale, pad,
        min_size, max_size, random_aspect_ratio,
        random_hls, lighting_pca_noise)
    img = img.transpose(2,0,1) # RGB uint8 (C,H,W)
    return img, lab

def _proc_fun_batch(buf_batch, compressed, data_shape,
    random_mirror=False, random_crop=False, mean_values=None, scale=None, pad=0,
    min_size=0, max_size=0, random_aspect_ratio=0.0,
    random_hls=None, lighting_pca_noise=0.0):
    res = []
    for buf in buf_batch:
        img, lab = _decode_data(buf, compressed)
        img = _aug_img(img, data_shape,
            random_mirror, random_crop, mean_values, scale, pad,
            min_size, max_size, random_aspect_ratio,
            random_hls, lighting_pca_noise)
        img = img.transpose(2,0,1) # RGB uint8 (C,H,W)
        res.append((img, lab))
    return res

class RecordSimpleAugmentationIter(RecordIter):
    """Simple agumentation for RecordIter.

    random_mirror : bool
        Random horizontal filp.
    random_crop : bool
        Random crop to specified data_shape.
    mean_values : None, float, or ndarray (length 3)
        Substract mean_values.
    scale : float
        Multiple with scale, after substract mean_values
    pad : int
        Pad extra pixels by reflection.
    min_size : int
        Minimum size of the shorter edge.
    max_size : int
        Minimum size of the shorter edge at [min_size,max_size], for scale augmentation.
    random_aspect_ratio : float
        Aspect ratio augmentation, jittering aspect ratio in [1/(1+random_aspect_ratio), 1+random_aspect_ratio].
    random_hls : None, float, or ndarray (length 3)
        HLS color augmentation, float or ndarray in range [0,1].
    lighting_pca_noise : float
        Lighting color augmentation, scalar in range [0,1].
    """
    def __init__(self, path_imgrec, data_shape, batch_size, compressed=True,
                 offset_on_reset=False, num_thread=0, epoch_size=0, skip_ratio=0.0,
                 random_mirror=False, random_crop=False, mean_values=None, scale=None, pad=0,
                 min_size=0, max_size=0, random_aspect_ratio=0.0,
                 random_hls=None, lighting_pca_noise=0.0):
        super(RecordSimpleAugmentationIter, self).__init__(path_imgrec, data_shape, batch_size, compressed,
            offset_on_reset, epoch_size, skip_ratio)

        self.num_thread = num_thread
        self.random_mirror=random_mirror
        self.random_crop=random_crop
        self.mean_values=mean_values
        self.scale = scale
        self.pad = pad
        self.min_size = min_size
        self.max_size = max_size
        if max_size > 0:
            assert max_size >= min_size
        self.random_aspect_ratio = random_aspect_ratio
        self.random_hls = random_hls
        self.lighting_pca_noise = lighting_pca_noise

    def _parse_data_label(self, _data):
        n_jobs = self.num_thread

        if n_jobs == 0:
            data = []
            label = []
            for buf in _data:
                img, lab = _proc_fun(buf, self.compressed, self.data_shape,
                    self.random_mirror, self.random_crop, self.mean_values, self.scale, self.pad,
                    self.min_size, self.max_size, self.random_aspect_ratio,
                    self.random_hls, self.lighting_pca_noise)
                data.append(img)
                label.append(lab)
        else:
            def split(l, n):
                return [l[i:i + n] for i in range(0, len(l), n)]

            def join(ll):
                return [item for sublist in ll for item in sublist]

            from joblib import Parallel, delayed
            # thread is created per batch, rather than per item
            out = Parallel(n_jobs=n_jobs, backend="threading")(delayed(_proc_fun_batch)(buf, self.compressed, self.data_shape,
                    self.random_mirror, self.random_crop, self.mean_values, self.scale, self.pad,
                    self.min_size, self.max_size, self.random_aspect_ratio,
                    self.random_hls, self.lighting_pca_noise) for buf in split(_data, n_jobs))
            out = join(out)
            # end multi-threading
            data = [o[0] for o in out]
            label = [o[1] for o in out]
        return data, label

    def _aug_img(self, img):
        '''Not being called inside _parse_data_label, only for debugging.'''
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn('_aug_img should only be called for debugging')
        return _aug_img(img, self.data_shape, self.random_mirror, self.random_crop,
                        self.mean_values, self.scale, self.pad,
                        self.min_size, self.max_size, self.random_aspect_ratio,
                        self.random_hls, self.lighting_pca_noise)

def test_aug_img():
    # mean_values
    rec = RecordSimpleAugmentationIter('',(3,32,32),10, mean_values=[1,1,1])
    assert (rec._aug_img(np.ones((32,32,3))) == 0.0).all()
    # scale
    rec = RecordSimpleAugmentationIter('',(3,32,32),10, scale=2.0)
    assert (rec._aug_img(np.ones((32,32,3))) == 2.0).all()
    # pad (note that image is finally cropped)
    rec = RecordSimpleAugmentationIter('',(3,40,40),10, pad=4)
    assert rec._aug_img(np.ones((32,32,3))).shape == (40,40,3)

    # aug lighting
    from skimage.data import astronaut
    import matplotlib.pyplot as plt
    img = astronaut()[:480] # (480,512,3)
    # img_aug_hls = _aug_hls(img, 0.4)
    # plt.imshow(img_aug_hls)
    rec = RecordSimpleAugmentationIter('',(3,480,480),10, random_hls=0.4)
    plt.imshow(rec._aug_img(img))

    def _aug_img(self, img):
        # same code, but remove random crop
        # assume img RGB float32 (H,W,C)
        # pad
        if self.pad > 0:
            img = cv2.copyMakeBorder(img,self.pad,self.pad,self.pad,self.pad,cv2.BORDER_REFLECT_101)
        # resize: random_aspect_ratio
        _h, _w = img.shape[:2]
        if self.random_aspect_ratio > 0.0:
            _ar = 1.0+self.random_aspect_ratio
            aspect_ratio = random.uniform(1.0/_ar, _ar)
            # image is always larger in size
            if aspect_ratio > 1.0:
                _h = int(np.round(_h*aspect_ratio))
            else:
                _w = int(np.round(_w/aspect_ratio))
        # resize (multi-scale): min_size and max_size
        size = None
        if self.min_size > 0 and self.max_size > 0:
            size = random.randint(self.min_size, self.max_size)
        elif self.min_size > 0:
            size = self.min_size
        if size:
            # NOTE: OpenCV use (width, height), while Numpy use (height, width)
            _h, _w = get_min_size(_h, _w, size)
        if _h != img.shape[0] or _w != img.shape[1]:
            img = cv2.resize(img, (_w,_h))
        return img

    # min_size
    rec = RecordSimpleAugmentationIter('',(3,32,32),10, min_size=64)
    assert _aug_img(rec, np.ones((32,40,3))).shape == (64,80,3)
    assert _aug_img(rec, np.ones((40,32,3))).shape == (80,64,3)
    # max_size
    rec = RecordSimpleAugmentationIter('',(3,32,32),10, min_size=32, max_size=64)
    assert 32 <= min(_aug_img(rec, np.ones((128,256,3))).shape[:2]) <= 64
    # random_aspect_ratio
    rec = RecordSimpleAugmentationIter('',(3,32,32),10, random_aspect_ratio=0.25)
    img = _aug_img(rec, np.ones((32,40,3)))
    assert img.shape[0] >= 32 and img.shape[1] >= 40
    aspect_ratio = img.shape[0]/float(img.shape[1])
    assert aspect_ratio != 1.0 # may fail with prob zero
    assert 0.8 * (32.0/40) <= aspect_ratio <= 1.25 * (32.0/40)


