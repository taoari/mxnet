"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
# import find_mxnet
import mxnet as mx

def get_symbol(num_classes = 1000, dataset='imagenet'):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(name='conv1',
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(name='relu1', data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(name='pool1',
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(name='lrn1', data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2 (group=2)
    sliced1 = mx.symbol.SliceChannel(lrn1, num_outputs=2)
    conv2_1 = mx.symbol.Convolution(name='conv2_1',
        data=sliced1[0], kernel=(5, 5), pad=(2, 2), num_filter=128)
    conv2_2 = mx.symbol.Convolution(name='conv2_2',
        data=sliced1[1], kernel=(5, 5), pad=(2, 2), num_filter=128)
    concat2 = mx.symbol.Concat(name='concat2', *[conv2_1, conv2_2])

    relu2 = mx.symbol.Activation(name='relu2', data=concat2, act_type="relu")
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(name='lrn2', data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3 (group=2)
    conv3 = mx.symbol.Convolution(name='conv3',
        data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(name='relu3', data=conv3, act_type="relu")

    sliced3 = mx.symbol.SliceChannel(relu3, num_outputs=2)
    conv4_1 = mx.symbol.Convolution(name='conv4_1',
        data=sliced3[0], kernel=(3, 3), pad=(1, 1), num_filter=192)
    conv4_2 = mx.symbol.Convolution(name='conv4_2',
        data=sliced3[1], kernel=(3, 3), pad=(1, 1), num_filter=192)
    concat4 = mx.symbol.Concat(name='concat4', *[conv4_1, conv4_2])

    relu4 = mx.symbol.Activation(name='relu4', data=concat4, act_type="relu")

    sliced4 = mx.symbol.SliceChannel(relu4, num_outputs=2)
    conv5_1 = mx.symbol.Convolution(name='conv5_1',
        data=sliced4[0], kernel=(3, 3), pad=(1, 1), num_filter=128)
    conv5_2 = mx.symbol.Convolution(name='conv5_2',
        data=sliced4[1], kernel=(3, 3), pad=(1, 1), num_filter=128)
    concat5 = mx.symbol.Concat(name='concat5', *[conv5_1, conv5_2])

    relu5 = mx.symbol.Activation(name='relu5', data=concat5, act_type="relu")
    pool3 = mx.symbol.Pooling(name='pool3', data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(name='flatten', data=pool3)
    fc1 = mx.symbol.FullyConnected(name='fc6', data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(name='relu6', data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(name='dropout1', data=relu6, p=0.5)
    # stage 5
    fc2 = mx.symbol.FullyConnected(name='fc7', data=dropout1, num_hidden=4096)
    relu7 = mx.symbol.Activation(name='relu7', data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(name='dropout2', data=relu7, p=0.5)
    # stage 6
    if dataset == 'imagenet':
        fc3 = mx.symbol.FullyConnected(name='fc8', data=dropout2, num_hidden=num_classes)
    else:
        fc3 = mx.symbol.FullyConnected(name='fc8_%s' % dataset, data=dropout2, num_hidden=num_classes, attr={'lr_mult': '10'})
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
