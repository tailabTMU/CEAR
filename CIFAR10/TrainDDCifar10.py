import gc

import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils, losses
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.callbacks import LearningRateScheduler

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("sigma", type=float, help="noise")
parser.add_argument("WWE_or_base", type=int, help="Select 1: Teacher, 2: to train DD single networks, 3: To train Noisy sginle network, 4: Compile WE")
parser.add_argument("modelNo", type=int, default=1, help="number of the model you want to train")
parser.add_argument("temp", type=int, default=1, help="temperature used to distill the soft logits")
args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train[0])
# Normalize the data


x_train, x_test = x_train / 255.0, x_test / 255.0



input_shape = x_train.shape[1:]
num_classes = 10

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
    horizontal_flip=True)
        
datagen.fit(x_train)
        
def resnet_block(input_layer, filters, kernel_size=3, strides=1, activation='relu'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def build_resnet_110(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = resnet_block(inputs, 16)
    
    for stack in range(3):
        for res_block in range(18):  # 54 total residual blocks (3 stacks * 18 blocks each)
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # Downsample
            y = resnet_block(x, 16 * (2 ** stack), strides=strides)
            y = resnet_block(y, 16 * (2 ** stack), activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_block(x, 16 * (2 ** stack), kernel_size=1, strides=strides, activation=None)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    #outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    #outputs = layers.Dense(num_classes, activation='None')(x)
    # outputs = layers.Dense(num_classes, kernel_initializer='he_normal')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def add_gaussian_noise(images, mean=0, std=0.5):
    noise = np.random.normal(loc=mean, scale=std, size=images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1) 

def train_teacher(sigma):
    '''
    Desc: Train Techer Network, trained without Noise. This is the model we will use to generate
    our soft labels for the coming student networds

    Param: None

    rtype: A trained Resnet110 model
    '''
    batch_size = 256
    
    def lr_schedule(epoch, lr):
        if epoch < 40:
            return 0.1
        elif epoch < 70:
            return 0.01
        else:
            return 0.001  
        
        
    lr_scheduler = LearningRateScheduler(lr_schedule)

    teacher = build_resnet_110(input_shape, num_classes)
    #Loss is using logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    teacher.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    teacher.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=110, validation_data=(x_test, y_test),  callbacks=[lr_scheduler])
    teacher.save(f'Cifar_updated_model/WithNoise/Cifar10Teacher.h5')

def train_WE(sigma, modelNO):
    '''
    Desc: Train a weighted ensemble without a distilled network of varying noise paramater beta

    Param: 
        sigma: the amount of Gaus noise (standard deviation)
        modelNo: the single networks model number in the network

    rtype: A trained Resnet110 model
    '''
    batch_size = 256

    noisy_x_train = add_gaussian_noise(x_train, std=sigma)
    noisy_x_test = add_gaussian_noise(x_test, std=sigma)
    
    def lr_schedule(epoch, lr):
        if epoch < 90:
            return 0.1
        elif epoch < 120:
            return 0.01
        else:
            return 0.001  
        
    # def lr_schedule(epoch, lr):
    #     if epoch > 0 and epoch % 30 == 0:
    #         return lr * 0.1  # Reduce the learning rate by a factor of 10 every 30 epochs
        
    #     return lr
    lr_scheduler = LearningRateScheduler(lr_schedule)

    teacher = build_resnet_110(input_shape, num_classes)

    teacher.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    #teacher.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    teacher.fit(datagen.flow(noisy_x_train, y_train, batch_size=batch_size), epochs=130, validation_data=(noisy_x_test, y_test),  callbacks=[lr_scheduler])
    teacher.save(f'Cifar_updated_model/WithNoise/SOTA/RS_WE_sigma{sigma}_modelno{modelNO}.h5')


import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks

# def _make_logits_model_from_softmax_model(softmax_model, name="logits_model"):
#     """
#     Create a logits-output model from a model whose last layer is Dense(units, activation='softmax').
#     We rebuild a final Dense(units, activation=None) on the penultimate tensor and copy weights.
#     """
#     last = softmax_model.layers[-1]
#     if not isinstance(last, tf.keras.layers.Dense):
#         raise ValueError("Expected last layer to be Dense(..., activation='softmax').")
#     if last.activation != tf.keras.activations.softmax:
#         raise ValueError("Expected last Dense activation to be softmax.")

#     # Penultimate tensor (input to the final Dense)
#     penultimate = last.input  # shape [?, 512] in your ResNet18

#     # New Dense with same units but linear activation => logits
#     logits = layers.Dense(
#         last.units,
#         activation=None,
#         use_bias=last.use_bias,
#         name=last.name + "_logits",
#         dtype="float32",  # keep logits stable
#     )(penultimate)

#     logits_model = tf.keras.Model(softmax_model.input, logits, name=name)

#     # Copy weights from original softmax Dense into logits Dense
#     logits_layer = logits_model.get_layer(last.name + "_logits")
#     logits_layer.set_weights(last.get_weights())

#     return logits_model


def eval_teach():
    print("Evaluating teacher")
    #teacher = models.load_model("Cifar_updated_model/WithNoise/Cifar10Teacher.h5", compile=False)
    teacher = models.load_model(f'Cifar_updated_model/WithNoise/Cifar10Teacher.h5', compile=False)
    # Compile for evaluation
    teacher.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # print(f"Loaded teacher from: {"Cifar_updated_model/WithNoise/Cifar10Teacher.h5"}")

    # Clean evaluation
    clean_loss, clean_acc = teacher.evaluate(x_test, y_test, batch_size=256, verbose=1)
    print(f"\nClean test loss: {clean_loss:.4f}")
    print(f"Clean test accuracy: {clean_acc:.4f}")


def train_model(sigma, modelno, temp):
    '''
    Desc: Train the individual student networks,

    param: 
        Sigma (float), the standard deviation for the Gaus. 
        modelno (int), identifies wich student in the ensemble you want to train
        temp (int), a small value from 1-4, that softenes the logits generated by the teacher network, 
            used to smooth the gradient
    
    rType: A trained Resnet110 model

    ''' 
    print("testing")
    epochs = 90
    batch_size = 256
    
    print(f"Training proposed Without Variable Gaus {sigma} with temp {modelno}")
    
    def train_defensive_distillation(model, noisy_x_train, y_train, noisy_x_val, y_val, T, soft_labels):
        kld = tf.keras.losses.KLDivergence()
        def lr_schedule(epoch, lr):
            if epoch < 30:
                return 0.1
            elif epoch < 60:
                return 0.01
            else:
                return 0.001
    
        
        lr_scheduler_prop = LearningRateScheduler(lr_schedule)

        # student = build_resnet18(input_shape, num_classes)
        # def kd_loss(y_true, y_pred):
        #     Ttf = tf.cast(T, tf.float32)

        #     y_true = tf.cast(y_true, tf.float32)
        #     y_pred = tf.cast(y_pred, tf.float32)

        #     s_probs = tf.nn.softmax(y_pred / Ttf, axis=-1)
        #     kl = tf.keras.losses.KLDivergence()(y_true, s_probs)

        #     return kl * (Ttf * Ttf)
        def kd_loss(y_true, y_pred):
            eps = 1e-7
            Ttf = tf.cast(T, tf.float32)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # clip
            y_true = tf.clip_by_value(y_true, eps, 1.0)
            y_pred = tf.clip_by_value(y_pred, eps, 1.0)

            # soften student probs
            y_pred_T = tf.pow(y_pred, 1.0 / Ttf)
            y_pred_T = y_pred_T / tf.reduce_sum(y_pred_T, axis=-1, keepdims=True)

            # (y_true already softened offline)
            kl = kld(y_true, y_pred_T)

            return kl * (Ttf * Ttf)

        model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True), loss=kd_loss, metrics=['accuracy'])
        model.fit(noisy_x_train, soft_labels, batch_size=batch_size, epochs=epochs, validation_data=(noisy_x_val, y_val),  callbacks=[lr_scheduler_prop])


        return model
    print("test2")
    teacher_model = models.load_model(f'Cifar_updated_model/WithNoise/Cifar10Teacher.h5', )

    var = (1/3) * sigma
    if modelno == 1:
        beta = float(sigma)
    elif modelno % 3 == 0 or modelno % 4 == 0:
        beta = sigma - round(random.uniform(0, var), 3)
    else:
        beta = sigma + round(random.uniform(0, var), 3)
        
    noisy_x_train = add_gaussian_noise(x_train, std=beta)
    noisy_x_test = add_gaussian_noise(x_test, std=sigma)
    print(f"Beta is {beta}")
    
    T = float(temp)
    eps = 1e-7

    soft_labels = teacher_model.predict(x_train)
    soft_labels = np.clip(soft_labels, eps, 1.0)
    soft_labels = soft_labels ** (1.0 / T)
    soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)


    student = build_resnet_110(input_shape, num_classes)
    trained_model = train_defensive_distillation(student,noisy_x_train, y_train, noisy_x_test, y_test, T, soft_labels)
    trained_model.save(f'Cifar_updated_model/WithNoise/proposed_WVGA/Student_network{sigma}_temp{modelno}_test.h5')
    # trained_model.save(f'ImagenetModels/Proposed/WVGA/Noise_{sigma}/Student_network{sigma}_modelNo{i}_temp{T}_test.h5')
    
    
    val_loss, val_acc = trained_model.evaluate(noisy_x_test, y_test, batch_size=batch_size, verbose=1)
    print(f"Noisy validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

    clean_loss, clean_acc = trained_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print(f"Clean validation: loss={clean_loss:.4f}, acc={clean_acc:.4f}")
    
    del trained_model
    tf.keras.backend.clear_session()
    gc.collect()



def train_proposed_WOVGA(sigma, modelno, temp):
    '''
    Desc: Train the individual student networks,

    param: 
        Sigma (float), the standard deviation for the Gaus. 
        modelno (int), identifies wich student in the ensemble you want to train
        temp (int), a small value from 1-4, that softenes the logits generated by the teacher network, 
            used to smooth the gradient
    
    rType: A trained Resnet110 model

    ''' 
    print("testing")
    epochs = 20
    batch_size = 256
    
    print(f"Training proposed Without Variable Gaus {sigma} with temp {modelno}")
    
    def train_defensive_distillation(model, noisy_x_train, y_train, noisy_x_val, y_val, T, soft_labels):
        kld = tf.keras.losses.KLDivergence()
        def lr_schedule(epoch, lr):
            if epoch < 30:
                return 0.1
            elif epoch < 60:
                return 0.01
            else:
                return 0.001
    
        
        lr_scheduler_prop = LearningRateScheduler(lr_schedule)

        # student = build_resnet18(input_shape, num_classes)
        # def kd_loss(y_true, y_pred):
        #     Ttf = tf.cast(T, tf.float32)

        #     y_true = tf.cast(y_true, tf.float32)
        #     y_pred = tf.cast(y_pred, tf.float32)

        #     s_probs = tf.nn.softmax(y_pred / Ttf, axis=-1)
        #     kl = tf.keras.losses.KLDivergence()(y_true, s_probs)

        #     return kl * (Ttf * Ttf)
        def kd_loss(y_true, y_pred):
            eps = 1e-7
            Ttf = tf.cast(T, tf.float32)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # clip
            y_true = tf.clip_by_value(y_true, eps, 1.0)
            y_pred = tf.clip_by_value(y_pred, eps, 1.0)

            # soften student probs
            y_pred_T = tf.pow(y_pred, 1.0 / Ttf)
            y_pred_T = y_pred_T / tf.reduce_sum(y_pred_T, axis=-1, keepdims=True)

            # (y_true already softened offline)
            kl = kld(y_true, y_pred_T)

            return kl * (Ttf * Ttf)

        model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True), loss=kd_loss, metrics=['accuracy'])
        model.fit(noisy_x_train, soft_labels, batch_size=batch_size, epochs=epochs, validation_data=(noisy_x_val, y_val),  callbacks=[lr_scheduler_prop])


        return model
    print("test2")
    teacher_model = models.load_model(f'Cifar_updated_model/WithNoise/Cifar10Teacher.h5', )
        
    noisy_x_train = add_gaussian_noise(x_train, std=sigma)
    noisy_x_test = add_gaussian_noise(x_test, std=sigma)
    print(f"sigma is {sigma}")
    
    T = float(temp)
    eps = 1e-7

    soft_labels = teacher_model.predict(x_train)
    soft_labels = np.clip(soft_labels, eps, 1.0)
    soft_labels = soft_labels ** (1.0 / T)
    soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)


    student = build_resnet_110(input_shape, num_classes)
    trained_model = train_defensive_distillation(student,noisy_x_train, y_train, noisy_x_test, y_test, T, soft_labels)
    trained_model.save(f'Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp{modelno}_test.h5')
    # trained_model.save(f'ImagenetModels/Proposed/WVGA/Noise_{sigma}/Student_network{sigma}_modelNo{i}_temp{T}_test.h5')
    
    
    val_loss, val_acc = trained_model.evaluate(noisy_x_test, y_test, batch_size=batch_size, verbose=1)
    print(f"Noisy validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

    clean_loss, clean_acc = trained_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print(f"Clean validation: loss={clean_loss:.4f}, acc={clean_acc:.4f}")
    
    del trained_model
    tf.keras.backend.clear_session()
    gc.collect()



def load_model_with_unique_name(model, temp):
    model._name = f"Model_{temp}_DD_T_{temp}"
    return model

def WeightedEnsemble(sigma):
    '''
    Desc: Takes the trained individual networks and forms a weighted ensemble

    param:
        sigma (float), depeicts the amount of noise we will add to the instances/ images
    
    rType: 
        A weighted ensemble formed from all the single networks created above 

    '''
    print(f"------- Creating weighted Ensemble for Sigma = {sigma}-------------")
    arr_models = []
    
    for i in range(1,6):
        #Change 0.7 to sigma
        model = tf.keras.models.load_model(f"Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp{i}.h5", compile=False) #SOTA
        #model = tf.keras.models.load_model(f"Cifar_updated_model/WithNoise/Noise_{sigma}/Student_network{sigma}_temp{i}.h5", compile=False)
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        model = load_model_with_unique_name(model, str(i))
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
 
    
    noisy_x_train = add_gaussian_noise(x_train, 0, args.sigma)
    noisy_x_test = add_gaussian_noise(x_test, 0, args.sigma)
            
    base_model_accuracies = [base_model.evaluate(noisy_x_test, y_test)[1] for base_model in arr_models]
    # Calculate weights based on accuracies
    print(base_model_accuracies)
    weights = np.array(base_model_accuracies) / sum(base_model_accuracies)

    inputs = layers.Input(shape=(input_shape))
    outputs = layers.average([model(inputs) * weight for model, weight in zip(arr_models, weights)])
    weighted_ensemble = models.Model(inputs, outputs)
    weighted_ensemble.compile(optimizer=optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    #weighted_ensemble.save(f'Cifar_updated_model/cifar_WWE_{str(args.sigma)}.h5')
    weighted_ensemble.save(f'Cifar_updated_model/Proposed_WOVGA_WWE_{str(args.sigma)}.h5')
    # def lr_schedule(epoch, lr):
    #     if epoch > 0 and epoch % 50 == 0:
    #         return lr * 0.1  # Reduce the learning rate by a factor of 10 every 30 epochs
        
    #     return lr
    
    # lr_scheduler = LearningRateScheduler(lr_schedule)

    # weighted_ensemble.compile(optimizer=optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    # weighted_ensemble.fit(datagen.flow(noisy_x_train, y_train, batch_size=256), epochs=60, validation_data=(noisy_x_test, y_test),  callbacks=[lr_scheduler])
    # weighted_ensemble.save(f'Cifar_updated_model/trained_cifar_WWE_{str(args.sigma)}.h5')

def MajorityVoting(sigma):
    '''
    Desc: Takes the trained individual networks and forms a weighted ensemble

    param:
        sigma (float), depeicts the amount of noise we will add to the instances/ images
    
    rType: 
        A weighted ensemble formed from all the single networks created above 

    '''
    print(f"------- Creating Majority Voting Ensemble for Sigma = {sigma}-------------")
    arr_models = []
    
    for i in range(1,6):
        #Change 0.7 to sigma
        model = tf.keras.models.load_model(f"Cifar_updated_model/WithNoise/proposed_WOVGA/Student_network{sigma}_temp{i}.h5", compile=False) #SOTA
        #model = tf.keras.models.load_model(f"Cifar_updated_model/WithNoise/Noise_{sigma}/Student_network{sigma}_temp{i}.h5", compile=False)
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        model = load_model_with_unique_name(model, str(i))
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
 
    
    # noisy_x_train = add_gaussian_noise(x_train, 0, args.sigma)
    # noisy_x_test = add_gaussian_noise(x_test, 0, args.sigma)
            
    # base_model_accuracies = [base_model.evaluate(noisy_x_test, y_test)[1] for base_model in arr_models]
    # # Calculate weights based on accuracies
    # print(base_model_accuracies)
    # weights = np.array(base_model_accuracies) / sum(base_model_accuracies)
    weights = np.array([0.2] * 5)
    
    inputs = layers.Input(shape=(input_shape))
    outputs = layers.average([model(inputs) * weight for model, weight in zip(arr_models, weights)])
    weighted_ensemble = models.Model(inputs, outputs)
    weighted_ensemble.compile(optimizer=optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    #weighted_ensemble.save(f'Cifar_updated_model/cifar_WWE_{str(args.sigma)}.h5')
    weighted_ensemble.save(f'Cifar_updated_model/WithNoise/Proposed_MVE_WOVGA/Proposed_MVE_WOVGA{str(args.sigma)}.h5')
    # def lr_schedule(epoch, lr):
    #     if epoch > 0 and epoch % 50 == 0:
    #         return lr * 0.1  # Reduce the learning rate by a factor of 10 every 30 epochs
        
    #     return lr
    
    # lr_scheduler = LearningRateScheduler(lr_schedule)

    # weighted_ensemble.compile(optimizer=optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    # weighted_ensemble.fit(datagen.flow(noisy_x_train, y_train, batch_size=256), epochs=60, validation_data=(noisy_x_test, y_test),  callbacks=[lr_scheduler])
    # weighted_ensemble.save(f'Cifar_updated_model/trained_cifar_WWE_{str(args.sigma)}.h5')



if __name__ == "__main__":
    
    print(args.WWE_or_base)
    if(args.WWE_or_base == 1):
        train_teacher(args.sigma)
    elif(args.WWE_or_base == 2):
        print(f"Training model Number {args.modelNo} with temp {args.temp}")
        train_model(args.sigma, args.modelNo, args.temp)
    elif(args.WWE_or_base == 3):
        print(f"Training model number {args.modelNo}")
        train_WE(args.sigma, args.modelNo)
    elif(args.WWE_or_base == 4):
        print(f"Train Proposed without VGA: modelNo{args.modelNo}")
        train_proposed_WOVGA(args.sigma, args.modelNo, args.temp)
    elif(args.WWE_or_base == 5):
        WeightedEnsemble(args.sigma)
    elif(args.WWE_or_base == 6):
        MajorityVoting(args.sigma)
    elif(args.WWE_or_base == 7):
        eval_teach()
        
    
