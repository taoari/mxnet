Unified Training and Transfer Learning API (Caffe-flavored)
======================================

### Setup

Simply add `<mxnet_root>/example/finetune-api` into the `PATH` variable.

Training Examples
-----------------

### Train on the MNIST dataset (MLP)

```
# reference validation accuracy: 0.973658
train_mnist.py --solver mnist_mlp.yml --gpus 0
```

* mnist_mlp.yml

```
network: mlp

data_dir: ../data/mnist/
dataset: mnist
num_classes: 10
num_examples: 60000
batch_size: 128
data_shape: 784 # flatten the data_shape: 784 = 28x28

lr: 0.01
lr_factor: 0.9
lr_factor_epoch: 2,5
momentum: 0.9
wd: 0.00005

display: 50
eval_epoch: 1
checkpoint_epoch: 5
num_epochs: 10
model_prefix: mnist_mlp

kv_store: local
initializer: msra
# monitor: ".*weight"
```

### Train on the MNIST dataset (LeNet)

```
# reference validation accuracy: 0.989784
train_mnist.py --solver mnist_lenet.yml --gpus 0
```

* mnist_lenet.yml

```
network: lenet

data_dir: ../data/mnist/
dataset: mnist
num_classes: 10
num_examples: 60000
batch_size: 128
data_shape: 1,28,28 # setup the correct data_shape: 28x28 gray images

lr: 0.01
lr_factor: 0.9
lr_factor_epoch: 2,5
momentum: 0.9
wd: 0.00005

display: 50
eval_epoch: 1
checkpoint_epoch: 5
num_epochs: 10
model_prefix: mnist_lenet

kv_store: local
initializer: msra
# monitor: ".*weight"
```

### Train on the Cifar10 dataset

```
# reference validation accuracy: 0.881210
train_cifar10.py --solver solver_8.yml --gpus 0
```

* solver_8.yml

```
network: resnet-cifar
network_kwargs: "{'num_block': 1, 'bottleneck': False}"

data_dir: ../data/cifar10/cifar-10-batches-py
dataset: cifar10
num_classes: 10
num_examples: 50000
batch_size: 256
data_shape: 3,32,32 # 32x32 RGB images

pad: 4
# mean_values: 123,117,104
# scale: 0.01667 # 1/60.

lr: 0.1
lr_factor: 0.1
lr_factor_epoch: 80,120
momentum: 0.9
wd: 0.0001

display: 50
eval_epoch: 8
checkpoint_epoch: 40
num_epochs: 320 # 64000 iters * 256 batch_size / 50000 samples = 327.68 epochs
model_prefix: cifar_resnet8 # 6n+2 layers, n=num_block, or 9n+2 layers with bottleneck

kv_store: local
initializer: msra
```

### Train on the ImageNet dataset

```
train_rec.py --solver solver_50.yml --gpus 0,1,2,3
```

* solver_50.yml

```
network: resnet-imagenet
network_kwargs: "{'depth': 50}"

dataset: imagenet
data_dir: ../data/imagenet
train_dataset: train_ori.rec
val_dataset: val_ori.rec

data_shape: 3,224,224
num_classes: 1000
num_examples: 1281167
batch_size: 256

# mean_values: 123,117,104 # BatchNorm to accumulate mean and var
min_size: 256 # min_size and max_size for scale augmentation
max_size: 480
# random_aspect_ratio: 0.25 # aspect ratio augmentation
# random_hls: 0.4 # color augmentation

lr: 0.1
lr_factor: 0.1
lr_factor_epoch: 30
momentum: 0.9
wd: 0.0001

display: 50
eval_epoch: 1
eval_metric: ce,acc,top5 # logging top-5 accuracy
checkpoint_epoch: 2 # 4h/epoch on 4GPU
num_epochs: 100
model_prefix: imagenet_resnet50

kv_store: local
initializer: msra
num_thread: 4 # multi-thread for decoding and resizing images
```

Transfer Learning Examples
--------------------------

### Finetune Caltech101 with the pre-trained Inception-BN network

```
train_rec.py --solver finetune_caltech101_inception.yml --gpus 0
```

* finetune_caltech101_inception.yml

```
network: inception-bn

dataset: caltech101
data_dir: ../data/caltech101
train_dataset: train_ori.rec
val_dataset: val_ori.rec

data_shape: 3,224,224
num_classes: 102
num_examples: 3060
batch_size: 25

mean_values: 123,117,104
min_size: 256

lr: 0.0005
lr_factor: 0.1
lr_factor_epoch: 160
momentum: 0.9
wd: 0.0005

display: 50
eval_epoch: 8
checkpoint_epoch: 80
num_epochs: 400 # 2.5x lr_factor_epoch

model_prefix: finetune_caltech101_inception
finetune_from: ../share/inception-bn/Inception_BN-0039.params

kv_store: local
initializer: msra
```

### Finetune Caltech101 with the pre-trained VGG16 network

```
train_rec.py --solver finetune_caltech101_vgg16.yml --gpus 0
```

* finetune_caltech101_vgg16.yml

```
network: vgg16

dataset: caltech101
data_dir: ../data/caltech101
train_dataset: train_ori.rec
val_dataset: val_ori.rec

data_shape: 3,224,224
num_classes: 102
num_examples: 3060
batch_size: 25

mean_values: 123,117,104
min_size: 256

lr: 0.0005
lr_factor: 0.1
lr_factor_epoch: 160
momentum: 0.9
wd: 0.0005

display: 50
eval_epoch: 8
checkpoint_epoch: 80
num_epochs: 400 # 2.5x lr_factor_epoch

model_prefix: finetune_caltech101_vgg16
finetune_from: ../share/vgg/vgg16-0001.params

kv_store: local
initializer: msra
```

Appendix
--------

### Data Preparation

Download Caltech101 dataset from <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>, and organize them as the following structure:

```
<root>/images/<cls>/*.jpg
```

Convert images to `.rec` format:

```
python make_list.py images caltech101 --train_ratio=0.8 --recursive=True
python im2rec.py caltech101_train images --resize=256
python im2rec.py caltech101_val images --resize=256
```

Rename and organize `caltech101_train.rec` and `caltech101_val.rec` to:

```
caltech101/train.rec
          /val.rec
```

For this split, there are 7315 training images, and 1829 validation images.

### Pre-trained VGG16 Model Preparation

Download VGG16 pre-trained model:

```
wget -c https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt
wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
```

Convert caffe model to mxnet model:

```
python convert_model.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel vgg16
```

Organize then generates as:

```
vgg16/vgg16-0001.params
     /vgg16-symbol.json
```
