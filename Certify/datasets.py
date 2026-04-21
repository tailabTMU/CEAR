from tensorflow.keras import layers, models, datasets, utils, optimizers, losses
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.image import random_crop, random_flip_left_right
from typing import *
import os
import ssl

# from setup_mnist import MNIST
from sklearn.model_selection import train_test_split
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request
from tensorflow.keras.datasets import mnist

#IMAGENET_LOC_ENV = "../cifar10img"  # Update this path as needed
#IMAGENET_LOC_ENV = "C://Users//danie//Documents//Smoothing//smoothing-master//cifar10img"
IMAGENET_LOC_ENV = "cifar0.25"
#C://Users//danie//Documents//MASC-Git//Daniel//code//cifar-10-binary//cifar-10-batches-bin
DATASETS = ["imagenet", "cifar10", "mnist"]

def get_dataset(dataset, split):
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
def get_num_classes(dataset):
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10

def get_normalize_layer(dataset):
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == 'mnist':
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)

# _IMAGENET_MEAN = [0.485, 0.456, 0.406]
# _IMAGENET_STDDEV = [0.229, 0.224, 0.225]

# ImageNet mean and std for normalization
_IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 1, 3], dtype=tf.float32)
_IMAGENET_STDDEV  = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 1, 3], dtype=tf.float32)

# _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
# _CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR10_MEAN = [-3.3746648e-05, -6.9930626e-05, -2.0962233e-04]
_CIFAR10_STDDEV = [0.00090255, 0.0008892,  0.00094748]

# _CIFAR10_MEAN = [0, 0, 0]
# _CIFAR10_STDDEV = [1, 1, 1]

#MNIST uses a grey scale therfore only have one mean and stdv value 
_MNIST_MEAN = [0.13066062, 0, 0]
_MNIST_STDDEV = [0.30810776, 0, 0]
import numpy as np

#-------------- Calculating the mean and std for Mnist -----------------
# Load MNIST dataset
# def mnist_mean_std():
#     # Load MNIST dataset
#     (x_train, _), (_, _) = mnist.load_data()

#     # Normalize pixel values to be between 0 and 1
#     x_train = x_train.astype('float32') / 255.0

#     # Calculate mean and standard deviation for the single channel
#     mean_value = np.mean(x_train)
#     stddev_value = np.std(x_train)

#     print("Mean value for MNIST:", mean_value)
#     print("Standard deviation for MNIST:", stddev_value)
#     # Calculate mean values for each channel
#     mean_values = np.mean(x_train, axis=(0, 1, 2))

#     print("Mean values for each channel (RGB):", mean_values)

#--------------------------------------------------
class NormalizeLayer(Layer):
    def __init__(self, means, sds):
        super(NormalizeLayer, self).__init__()
        self.means = means
        self.sds = sds

    def call(self, inputs):
        return (inputs - self.means) / self.sds

def _cifar10(split):
    if split == "train":
        (x_train, y_train), (_, _) = datasets.cifar10.load_data()
        x_train = data_augmentation(x_train)
        return x_train, y_train
    
    elif split == "test":
        print("Testing")
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        
        
        # y_train = tf.keras.utils.to_categorical(y_train, 10)
        # y_test = tf.keras.utils.to_categorical(y_test, 10)

        # Convert class vectors to binary class matrices
        y_train = utils.to_categorical(y_train, 10)
        y_test = utils.to_categorical(y_test, 10)
        
        return x_test, y_test

ssl._create_default_https_context = ssl._create_unverified_context

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    


# def _cifar10(split):
#     if split == "train":
#         pass
#     elif split == "test":
#         train_data = []
#         train_labels = []
#         #cifar_path = 'C://Users//danie//Documents//MASC-Git//Daniel//code//cifar-10-binary//cifar-10-batches-bin'
#         cifar_path = '//home//grad//dsadig//Documents//code//cifar-10-binary//cifar-10-batches-bin' #For server use
#         #Create OS path with binary data
#         # if not os.path.exists("cifar-10-batches-bin"):
#         if not os.path.exists(cifar_path):
#             print("Downloding")
#             urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
#                                        "cifar-data.tar.gz")
#             os.popen("tar -xzf cifar-data.tar.gz").read()
            
#         print("\n Grabing Cifar Data \n")
#         for i in range(5):
#             r,s = load_batch(f"{cifar_path}/data_batch_"+str(i+1)+".bin")
#             train_data.extend(r)
#             train_labels.extend(s)
            
#         train_data = np.array(train_data,dtype=np.float32)
#         train_labels = np.array(train_labels)
        
#         test_data, test_labels = load_batch(f"{cifar_path}/test_batch.bin")
       
#         return test_data, test_labels

def _mnist(split):
    if split == "train":
        pass
    elif split == "test":
        print("testing Mnist")
        # Extract features and labels
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        print("using x_test")
        
        return x_test, y_test
    
# def _imagenet(split):
    if IMAGENET_LOC_ENV not in os.environ:
        raise RuntimeError("Environment variable for ImageNet directory not set")
    
    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = tf.keras.Sequential([
            Lambda(lambda x: random_crop(x, [224, 224, 3])),
            Lambda(lambda x: random_flip_left_right(x)),
            Lambda(preprocess_input)
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = tf.keras.Sequential([
            Lambda(lambda x: random_crop(x, [224, 224, 3])),
            Lambda(preprocess_input)
        ])
    return datasets.image_dataset_from_directory(subdir, label_mode="categorical", batch_size=64, image_size=(224, 224))
def _imagenet(split):
    train_ds = tf.keras.utils.image_dataset_from_directory(
    "tiny-imagenet-200/train",
    image_size=(64, 64),
    batch_size=128,
    label_mode="categorical",
    validation_split=0.2,   # drop 40%
    subset="training",
    seed=42
)

    print(len(train_ds.class_names))

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "tiny-imagenet-200/val",
        image_size=(64, 64),
        batch_size=256,
        label_mode="categorical",
        seed=42
    )


    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label


    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # train_ds = train_ds.map(normalize).prefetch(AUTOTUNE)
    # val_ds = val_ds.map(normalize).prefetch(AUTOTUNE)
    def augment(images, labels):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, 0.2)
        images = tf.image.random_contrast(images, 0.8, 1.2)
        return images, labels


    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# test_ds  = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def dataset_to_numpy(dataset):
        """Convert a tf.data.Dataset -> (X, y) NumPy arrays."""
        
        images, labels = [], []
        for batch_imgs, batch_labels in dataset:
            images.append(batch_imgs.numpy())
            labels.append(batch_labels.numpy())
        X = np.concatenate(images, axis=0)
        y = np.concatenate(labels, axis=0)
        return X, y

    # Convert your datasets
    x_train, y_train = dataset_to_numpy(train_ds)
    x_val,   y_val   = dataset_to_numpy(val_ds)
    
    return x_val, y_val



# def _imagenet(split):
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#     'tiny-imagenet-200/train',
#     labels='inferred', label_mode='categorical',
#     image_size=(64, 64), batch_size=256, shuffle=True,
#     validation_split=0.2, subset='training', seed=42
#     )
    
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#     'tiny-imagenet-200/train',
#     labels='inferred', label_mode='categorical',
#     image_size=(64, 64), batch_size=256, shuffle=True,
#     validation_split=0.2, subset='validation', seed=42
# )
    
#     def normalize(images, labels):
#         images = tf.cast(images, tf.float32) / 255.0  # scale to [0,1]
#         images = (images - _IMAGENET_MEAN) / _IMAGENET_STDDEV  # normalize
#         return images, labels

#     train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.map(normalize, num_parallel_calls = tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
#     def dataset_to_numpy(dataset):
#         """Convert a tf.data.Dataset -> (X, y) NumPy arrays."""
#         images, labels = [], []
#         for batch_imgs, batch_labels in dataset:
#             images.append(batch_imgs.numpy())
#             labels.append(batch_labels.numpy())
#         X = np.concatenate(images, axis=0)
#         y = np.concatenate(labels, axis=0)
#         return X, y

#     x_train, y_train = dataset_to_numpy(train_ds)
#     x_val, y_val = dataset_to_numpy(val_ds)
#     return x_train, y_train
    


def data_augmentation(x_train):
    x_train = random_crop(x_train, [32, 32, 3])
    x_train = random_flip_left_right(x_train)
    return x_train
