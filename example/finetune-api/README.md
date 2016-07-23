Transfer Learning API (Caffe-flavored)
======================================

Setup
-----

Simply add `<mxnet_root>/example/finetune-api` into the `PATH` variable.

Examples
--------

**Train on the MNIST dataset**

```
train_mnist.py --solver mnist_lenet.yml --gpus 0
```

* mnist_lenet.yml

```
data_dir: ../data/mnist/
dataset: mnist
num_classes: 10
num_examples: 60000
batch_size: 128

lr: 0.01
lr_factor: 0.9
lr_factor_epoch: 2,5
momentum: 0.8
wd: 0.00005

display: 50
eval_epoch: 1
checkpoint_epoch: 5
num_epochs: 10
model_prefix: mnist_lenet

kv_store: local
network: lenet
initializer: msra
# monitor: ".*weight"
```

**Finetune Caltech101 with the pre-trained Inception-BN network**

```
train_imagenet.py --solver finetune_caltech101_inception.yml --gpus 0
```

* finetune_caltech101_inception.yml

```
network: inception-bn

data_dir: ../data/caltech101
dataset: caltech101
num_classes: 102
num_examples: 3060
batch_size: 25

lr: 0.0005
lr_factor: 0.1
lr_factor_epoch: 320
momentum: 0.9
wd: 0.0005

num_epochs: 800
kv_store: local

model_prefix: finetune_caltech101_inception
finetune_from: ../share/inception-bn/Inception_BN-0039.params
checkpoint_epoch: 80
eval_epoch: 16
```

**Finetune Caltech101 with the pre-trained VGG16 network**

```
train_imagenet.py --solver finetune_caltech101_vgg16.yml --gpus 0
```

* finetune_caltech101_vgg16.yml

```
network: vgg16

data_dir: ../data/caltech101
dataset: caltech101
num_classes: 102
num_examples: 3060
batch_size: 25

lr: 0.0005
lr_factor: 0.1
lr_factor_epoch: 320
momentum: 0.9
wd: 0.0005

num_epochs: 800
kv_store: local

model_prefix: finetune_caltech101_vgg16
finetune_from: ../share/vgg16/vgg16-0001.params
checkpoint_epoch: 80
eval_epoch: 16
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