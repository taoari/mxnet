import find_mxnet
import mxnet as mx

def get_symbol(num_classes=1000, dataset='imagenet'):
	data = mx.symbol.Variable(name="data")
	conv1_7x7_s2 = mx.symbol.Convolution(name='conv1_7x7_s2', data=data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=False)
	conv1_relu_7x7 = mx.symbol.Activation(name='conv1_relu_7x7', data=conv1_7x7_s2 , act_type='relu')
	pool1_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv1_relu_7x7 , pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
	pool1_norm1 = mx.symbol.LRN(name='pool1_norm1', data=pool1_3x3_s2 , alpha=0.000100, beta=0.750000, knorm=1.000000, nsize=5)
	conv2_3x3_reduce = mx.symbol.Convolution(name='conv2_3x3_reduce', data=pool1_norm1 , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	conv2_relu_3x3_reduce = mx.symbol.Activation(name='conv2_relu_3x3_reduce', data=conv2_3x3_reduce , act_type='relu')
	conv2_3x3 = mx.symbol.Convolution(name='conv2_3x3', data=conv2_relu_3x3_reduce , num_filter=192, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	conv2_relu_3x3 = mx.symbol.Activation(name='conv2_relu_3x3', data=conv2_3x3 , act_type='relu')
	conv2_norm2 = mx.symbol.LRN(name='conv2_norm2', data=conv2_relu_3x3 , alpha=0.000100, beta=0.750000, knorm=1.000000, nsize=5)
	pool2_3x3_s2 = mx.symbol.Pooling(name='pool2_3x3_s2', data=conv2_norm2 , pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
	inception_3a_1x1 = mx.symbol.Convolution(name='inception_3a_1x1', data=pool2_3x3_s2 , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3a_relu_1x1 = mx.symbol.Activation(name='inception_3a_relu_1x1', data=inception_3a_1x1 , act_type='relu')
	inception_3a_3x3_reduce = mx.symbol.Convolution(name='inception_3a_3x3_reduce', data=pool2_3x3_s2 , num_filter=96, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3a_relu_3x3_reduce = mx.symbol.Activation(name='inception_3a_relu_3x3_reduce', data=inception_3a_3x3_reduce , act_type='relu')
	inception_3a_3x3 = mx.symbol.Convolution(name='inception_3a_3x3', data=inception_3a_relu_3x3_reduce , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_3a_relu_3x3 = mx.symbol.Activation(name='inception_3a_relu_3x3', data=inception_3a_3x3 , act_type='relu')
	inception_3a_5x5_reduce = mx.symbol.Convolution(name='inception_3a_5x5_reduce', data=pool2_3x3_s2 , num_filter=16, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3a_relu_5x5_reduce = mx.symbol.Activation(name='inception_3a_relu_5x5_reduce', data=inception_3a_5x5_reduce , act_type='relu')
	inception_3a_5x5 = mx.symbol.Convolution(name='inception_3a_5x5', data=inception_3a_relu_5x5_reduce , num_filter=32, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_3a_relu_5x5 = mx.symbol.Activation(name='inception_3a_relu_5x5', data=inception_3a_5x5 , act_type='relu')
	inception_3a_pool = mx.symbol.Pooling(name='inception_3a_pool', data=pool2_3x3_s2 , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_3a_pool_proj = mx.symbol.Convolution(name='inception_3a_pool_proj', data=inception_3a_pool , num_filter=32, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3a_relu_pool_proj = mx.symbol.Activation(name='inception_3a_relu_pool_proj', data=inception_3a_pool_proj , act_type='relu')
	inception_3a_output = mx.symbol.Concat(name='inception_3a_output', *[inception_3a_relu_1x1,inception_3a_relu_3x3,inception_3a_relu_5x5,inception_3a_relu_pool_proj] )
	inception_3b_1x1 = mx.symbol.Convolution(name='inception_3b_1x1', data=inception_3a_output , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3b_relu_1x1 = mx.symbol.Activation(name='inception_3b_relu_1x1', data=inception_3b_1x1 , act_type='relu')
	inception_3b_3x3_reduce = mx.symbol.Convolution(name='inception_3b_3x3_reduce', data=inception_3a_output , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3b_relu_3x3_reduce = mx.symbol.Activation(name='inception_3b_relu_3x3_reduce', data=inception_3b_3x3_reduce , act_type='relu')
	inception_3b_3x3 = mx.symbol.Convolution(name='inception_3b_3x3', data=inception_3b_relu_3x3_reduce , num_filter=192, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_3b_relu_3x3 = mx.symbol.Activation(name='inception_3b_relu_3x3', data=inception_3b_3x3 , act_type='relu')
	inception_3b_5x5_reduce = mx.symbol.Convolution(name='inception_3b_5x5_reduce', data=inception_3a_output , num_filter=32, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3b_relu_5x5_reduce = mx.symbol.Activation(name='inception_3b_relu_5x5_reduce', data=inception_3b_5x5_reduce , act_type='relu')
	inception_3b_5x5 = mx.symbol.Convolution(name='inception_3b_5x5', data=inception_3b_relu_5x5_reduce , num_filter=96, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_3b_relu_5x5 = mx.symbol.Activation(name='inception_3b_relu_5x5', data=inception_3b_5x5 , act_type='relu')
	inception_3b_pool = mx.symbol.Pooling(name='inception_3b_pool', data=inception_3a_output , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_3b_pool_proj = mx.symbol.Convolution(name='inception_3b_pool_proj', data=inception_3b_pool , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_3b_relu_pool_proj = mx.symbol.Activation(name='inception_3b_relu_pool_proj', data=inception_3b_pool_proj , act_type='relu')
	inception_3b_output = mx.symbol.Concat(name='inception_3b_output', *[inception_3b_relu_1x1,inception_3b_relu_3x3,inception_3b_relu_5x5,inception_3b_relu_pool_proj] )
	pool3_3x3_s2 = mx.symbol.Pooling(name='pool3_3x3_s2', data=inception_3b_output , pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
	inception_4a_1x1 = mx.symbol.Convolution(name='inception_4a_1x1', data=pool3_3x3_s2 , num_filter=192, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4a_relu_1x1 = mx.symbol.Activation(name='inception_4a_relu_1x1', data=inception_4a_1x1 , act_type='relu')
	inception_4a_3x3_reduce = mx.symbol.Convolution(name='inception_4a_3x3_reduce', data=pool3_3x3_s2 , num_filter=96, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4a_relu_3x3_reduce = mx.symbol.Activation(name='inception_4a_relu_3x3_reduce', data=inception_4a_3x3_reduce , act_type='relu')
	inception_4a_3x3 = mx.symbol.Convolution(name='inception_4a_3x3', data=inception_4a_relu_3x3_reduce , num_filter=208, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_4a_relu_3x3 = mx.symbol.Activation(name='inception_4a_relu_3x3', data=inception_4a_3x3 , act_type='relu')
	inception_4a_5x5_reduce = mx.symbol.Convolution(name='inception_4a_5x5_reduce', data=pool3_3x3_s2 , num_filter=16, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4a_relu_5x5_reduce = mx.symbol.Activation(name='inception_4a_relu_5x5_reduce', data=inception_4a_5x5_reduce , act_type='relu')
	inception_4a_5x5 = mx.symbol.Convolution(name='inception_4a_5x5', data=inception_4a_relu_5x5_reduce , num_filter=48, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_4a_relu_5x5 = mx.symbol.Activation(name='inception_4a_relu_5x5', data=inception_4a_5x5 , act_type='relu')
	inception_4a_pool = mx.symbol.Pooling(name='inception_4a_pool', data=pool3_3x3_s2 , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_4a_pool_proj = mx.symbol.Convolution(name='inception_4a_pool_proj', data=inception_4a_pool , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4a_relu_pool_proj = mx.symbol.Activation(name='inception_4a_relu_pool_proj', data=inception_4a_pool_proj , act_type='relu')
	inception_4a_output = mx.symbol.Concat(name='inception_4a_output', *[inception_4a_relu_1x1,inception_4a_relu_3x3,inception_4a_relu_5x5,inception_4a_relu_pool_proj] )
	inception_4b_1x1 = mx.symbol.Convolution(name='inception_4b_1x1', data=inception_4a_output , num_filter=160, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4b_relu_1x1 = mx.symbol.Activation(name='inception_4b_relu_1x1', data=inception_4b_1x1 , act_type='relu')
	inception_4b_3x3_reduce = mx.symbol.Convolution(name='inception_4b_3x3_reduce', data=inception_4a_output , num_filter=112, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4b_relu_3x3_reduce = mx.symbol.Activation(name='inception_4b_relu_3x3_reduce', data=inception_4b_3x3_reduce , act_type='relu')
	inception_4b_3x3 = mx.symbol.Convolution(name='inception_4b_3x3', data=inception_4b_relu_3x3_reduce , num_filter=224, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_4b_relu_3x3 = mx.symbol.Activation(name='inception_4b_relu_3x3', data=inception_4b_3x3 , act_type='relu')
	inception_4b_5x5_reduce = mx.symbol.Convolution(name='inception_4b_5x5_reduce', data=inception_4a_output , num_filter=24, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4b_relu_5x5_reduce = mx.symbol.Activation(name='inception_4b_relu_5x5_reduce', data=inception_4b_5x5_reduce , act_type='relu')
	inception_4b_5x5 = mx.symbol.Convolution(name='inception_4b_5x5', data=inception_4b_relu_5x5_reduce , num_filter=64, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_4b_relu_5x5 = mx.symbol.Activation(name='inception_4b_relu_5x5', data=inception_4b_5x5 , act_type='relu')
	inception_4b_pool = mx.symbol.Pooling(name='inception_4b_pool', data=inception_4a_output , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_4b_pool_proj = mx.symbol.Convolution(name='inception_4b_pool_proj', data=inception_4b_pool , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4b_relu_pool_proj = mx.symbol.Activation(name='inception_4b_relu_pool_proj', data=inception_4b_pool_proj , act_type='relu')
	inception_4b_output = mx.symbol.Concat(name='inception_4b_output', *[inception_4b_relu_1x1,inception_4b_relu_3x3,inception_4b_relu_5x5,inception_4b_relu_pool_proj] )
	inception_4c_1x1 = mx.symbol.Convolution(name='inception_4c_1x1', data=inception_4b_output , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4c_relu_1x1 = mx.symbol.Activation(name='inception_4c_relu_1x1', data=inception_4c_1x1 , act_type='relu')
	inception_4c_3x3_reduce = mx.symbol.Convolution(name='inception_4c_3x3_reduce', data=inception_4b_output , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4c_relu_3x3_reduce = mx.symbol.Activation(name='inception_4c_relu_3x3_reduce', data=inception_4c_3x3_reduce , act_type='relu')
	inception_4c_3x3 = mx.symbol.Convolution(name='inception_4c_3x3', data=inception_4c_relu_3x3_reduce , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_4c_relu_3x3 = mx.symbol.Activation(name='inception_4c_relu_3x3', data=inception_4c_3x3 , act_type='relu')
	inception_4c_5x5_reduce = mx.symbol.Convolution(name='inception_4c_5x5_reduce', data=inception_4b_output , num_filter=24, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4c_relu_5x5_reduce = mx.symbol.Activation(name='inception_4c_relu_5x5_reduce', data=inception_4c_5x5_reduce , act_type='relu')
	inception_4c_5x5 = mx.symbol.Convolution(name='inception_4c_5x5', data=inception_4c_relu_5x5_reduce , num_filter=64, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_4c_relu_5x5 = mx.symbol.Activation(name='inception_4c_relu_5x5', data=inception_4c_5x5 , act_type='relu')
	inception_4c_pool = mx.symbol.Pooling(name='inception_4c_pool', data=inception_4b_output , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_4c_pool_proj = mx.symbol.Convolution(name='inception_4c_pool_proj', data=inception_4c_pool , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4c_relu_pool_proj = mx.symbol.Activation(name='inception_4c_relu_pool_proj', data=inception_4c_pool_proj , act_type='relu')
	inception_4c_output = mx.symbol.Concat(name='inception_4c_output', *[inception_4c_relu_1x1,inception_4c_relu_3x3,inception_4c_relu_5x5,inception_4c_relu_pool_proj] )
	inception_4d_1x1 = mx.symbol.Convolution(name='inception_4d_1x1', data=inception_4c_output , num_filter=112, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4d_relu_1x1 = mx.symbol.Activation(name='inception_4d_relu_1x1', data=inception_4d_1x1 , act_type='relu')
	inception_4d_3x3_reduce = mx.symbol.Convolution(name='inception_4d_3x3_reduce', data=inception_4c_output , num_filter=144, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4d_relu_3x3_reduce = mx.symbol.Activation(name='inception_4d_relu_3x3_reduce', data=inception_4d_3x3_reduce , act_type='relu')
	inception_4d_3x3 = mx.symbol.Convolution(name='inception_4d_3x3', data=inception_4d_relu_3x3_reduce , num_filter=288, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_4d_relu_3x3 = mx.symbol.Activation(name='inception_4d_relu_3x3', data=inception_4d_3x3 , act_type='relu')
	inception_4d_5x5_reduce = mx.symbol.Convolution(name='inception_4d_5x5_reduce', data=inception_4c_output , num_filter=32, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4d_relu_5x5_reduce = mx.symbol.Activation(name='inception_4d_relu_5x5_reduce', data=inception_4d_5x5_reduce , act_type='relu')
	inception_4d_5x5 = mx.symbol.Convolution(name='inception_4d_5x5', data=inception_4d_relu_5x5_reduce , num_filter=64, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_4d_relu_5x5 = mx.symbol.Activation(name='inception_4d_relu_5x5', data=inception_4d_5x5 , act_type='relu')
	inception_4d_pool = mx.symbol.Pooling(name='inception_4d_pool', data=inception_4c_output , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_4d_pool_proj = mx.symbol.Convolution(name='inception_4d_pool_proj', data=inception_4d_pool , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4d_relu_pool_proj = mx.symbol.Activation(name='inception_4d_relu_pool_proj', data=inception_4d_pool_proj , act_type='relu')
	inception_4d_output = mx.symbol.Concat(name='inception_4d_output', *[inception_4d_relu_1x1,inception_4d_relu_3x3,inception_4d_relu_5x5,inception_4d_relu_pool_proj] )
	inception_4e_1x1 = mx.symbol.Convolution(name='inception_4e_1x1', data=inception_4d_output , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4e_relu_1x1 = mx.symbol.Activation(name='inception_4e_relu_1x1', data=inception_4e_1x1 , act_type='relu')
	inception_4e_3x3_reduce = mx.symbol.Convolution(name='inception_4e_3x3_reduce', data=inception_4d_output , num_filter=160, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4e_relu_3x3_reduce = mx.symbol.Activation(name='inception_4e_relu_3x3_reduce', data=inception_4e_3x3_reduce , act_type='relu')
	inception_4e_3x3 = mx.symbol.Convolution(name='inception_4e_3x3', data=inception_4e_relu_3x3_reduce , num_filter=320, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_4e_relu_3x3 = mx.symbol.Activation(name='inception_4e_relu_3x3', data=inception_4e_3x3 , act_type='relu')
	inception_4e_5x5_reduce = mx.symbol.Convolution(name='inception_4e_5x5_reduce', data=inception_4d_output , num_filter=32, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4e_relu_5x5_reduce = mx.symbol.Activation(name='inception_4e_relu_5x5_reduce', data=inception_4e_5x5_reduce , act_type='relu')
	inception_4e_5x5 = mx.symbol.Convolution(name='inception_4e_5x5', data=inception_4e_relu_5x5_reduce , num_filter=128, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_4e_relu_5x5 = mx.symbol.Activation(name='inception_4e_relu_5x5', data=inception_4e_5x5 , act_type='relu')
	inception_4e_pool = mx.symbol.Pooling(name='inception_4e_pool', data=inception_4d_output , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_4e_pool_proj = mx.symbol.Convolution(name='inception_4e_pool_proj', data=inception_4e_pool , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_4e_relu_pool_proj = mx.symbol.Activation(name='inception_4e_relu_pool_proj', data=inception_4e_pool_proj , act_type='relu')
	inception_4e_output = mx.symbol.Concat(name='inception_4e_output', *[inception_4e_relu_1x1,inception_4e_relu_3x3,inception_4e_relu_5x5,inception_4e_relu_pool_proj] )
	pool4_3x3_s2 = mx.symbol.Pooling(name='pool4_3x3_s2', data=inception_4e_output , pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
	inception_5a_1x1 = mx.symbol.Convolution(name='inception_5a_1x1', data=pool4_3x3_s2 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5a_relu_1x1 = mx.symbol.Activation(name='inception_5a_relu_1x1', data=inception_5a_1x1 , act_type='relu')
	inception_5a_3x3_reduce = mx.symbol.Convolution(name='inception_5a_3x3_reduce', data=pool4_3x3_s2 , num_filter=160, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5a_relu_3x3_reduce = mx.symbol.Activation(name='inception_5a_relu_3x3_reduce', data=inception_5a_3x3_reduce , act_type='relu')
	inception_5a_3x3 = mx.symbol.Convolution(name='inception_5a_3x3', data=inception_5a_relu_3x3_reduce , num_filter=320, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_5a_relu_3x3 = mx.symbol.Activation(name='inception_5a_relu_3x3', data=inception_5a_3x3 , act_type='relu')
	inception_5a_5x5_reduce = mx.symbol.Convolution(name='inception_5a_5x5_reduce', data=pool4_3x3_s2 , num_filter=32, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5a_relu_5x5_reduce = mx.symbol.Activation(name='inception_5a_relu_5x5_reduce', data=inception_5a_5x5_reduce , act_type='relu')
	inception_5a_5x5 = mx.symbol.Convolution(name='inception_5a_5x5', data=inception_5a_relu_5x5_reduce , num_filter=128, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_5a_relu_5x5 = mx.symbol.Activation(name='inception_5a_relu_5x5', data=inception_5a_5x5 , act_type='relu')
	inception_5a_pool = mx.symbol.Pooling(name='inception_5a_pool', data=pool4_3x3_s2 , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_5a_pool_proj = mx.symbol.Convolution(name='inception_5a_pool_proj', data=inception_5a_pool , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5a_relu_pool_proj = mx.symbol.Activation(name='inception_5a_relu_pool_proj', data=inception_5a_pool_proj , act_type='relu')
	inception_5a_output = mx.symbol.Concat(name='inception_5a_output', *[inception_5a_relu_1x1,inception_5a_relu_3x3,inception_5a_relu_5x5,inception_5a_relu_pool_proj] )
	inception_5b_1x1 = mx.symbol.Convolution(name='inception_5b_1x1', data=inception_5a_output , num_filter=384, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5b_relu_1x1 = mx.symbol.Activation(name='inception_5b_relu_1x1', data=inception_5b_1x1 , act_type='relu')
	inception_5b_3x3_reduce = mx.symbol.Convolution(name='inception_5b_3x3_reduce', data=inception_5a_output , num_filter=192, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5b_relu_3x3_reduce = mx.symbol.Activation(name='inception_5b_relu_3x3_reduce', data=inception_5b_3x3_reduce , act_type='relu')
	inception_5b_3x3 = mx.symbol.Convolution(name='inception_5b_3x3', data=inception_5b_relu_3x3_reduce , num_filter=384, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
	inception_5b_relu_3x3 = mx.symbol.Activation(name='inception_5b_relu_3x3', data=inception_5b_3x3 , act_type='relu')
	inception_5b_5x5_reduce = mx.symbol.Convolution(name='inception_5b_5x5_reduce', data=inception_5a_output , num_filter=48, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5b_relu_5x5_reduce = mx.symbol.Activation(name='inception_5b_relu_5x5_reduce', data=inception_5b_5x5_reduce , act_type='relu')
	inception_5b_5x5 = mx.symbol.Convolution(name='inception_5b_5x5', data=inception_5b_relu_5x5_reduce , num_filter=128, pad=(2,2), kernel=(5,5), stride=(1,1), no_bias=False)
	inception_5b_relu_5x5 = mx.symbol.Activation(name='inception_5b_relu_5x5', data=inception_5b_5x5 , act_type='relu')
	inception_5b_pool = mx.symbol.Pooling(name='inception_5b_pool', data=inception_5a_output , pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='max')
	inception_5b_pool_proj = mx.symbol.Convolution(name='inception_5b_pool_proj', data=inception_5b_pool , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
	inception_5b_relu_pool_proj = mx.symbol.Activation(name='inception_5b_relu_pool_proj', data=inception_5b_pool_proj , act_type='relu')
	inception_5b_output = mx.symbol.Concat(name='inception_5b_output', *[inception_5b_relu_1x1,inception_5b_relu_3x3,inception_5b_relu_5x5,inception_5b_relu_pool_proj] )
	pool5_7x7_s1 = mx.symbol.Pooling(name='pool5_7x7_s1', data=inception_5b_output , pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
	pool5_drop_7x7_s1 = mx.symbol.Dropout(name='pool5_drop_7x7_s1', data=pool5_7x7_s1 , p=0.400000)
	flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool5_drop_7x7_s1)
	if dataset == 'imagenet':
		loss3_classifier = mx.symbol.FullyConnected(name='loss3_classifier', data=flatten_0 , num_hidden=num_classes, no_bias=False)
	else:
		loss3_classifier = mx.symbol.FullyConnected(name='loss3_classifier_%s' % dataset, data=flatten_0 , num_hidden=num_classes, no_bias=False, attr={'lr_mult': '10'})
	prob = mx.symbol.SoftmaxOutput(name='softmax', data=loss3_classifier )
	return prob

