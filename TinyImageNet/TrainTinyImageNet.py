# import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils, losses
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np, math, random
from tensorflow.keras import mixed_precision
import gc
import math



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("sigma", type=float, help="noise")
parser.add_argument("WWE_or_base", type=int, help="Select 1: Teacher, 2: to train DD single networks, 3: To train Noisy sginle network, 4: Compile WE")
parser.add_argument("modelNo", type=int, default=1, help="number of the model you want to train")
parser.add_argument("temp", type=int, default=1, help="temperature used to distill the soft logits")
args = parser.parse_args()


# gpus = tf.config.list_physical_devices('GPU')
# print("Available GPUs:", gpus)

# if gpus:
#     try:
#         tf.config.set_visible_devices(gpus[1], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[1], True)  # Prevents OOM on startup
#         print("Now using GPU:1")
#     except RuntimeError as e:
#         print(e)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# mixed_precision.set_global_policy("mixed_float16")


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
    batch_size=128,
    label_mode="categorical",
    seed=42
)


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

def augment(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, 0.2)
    images = tf.image.random_contrast(images, 0.8, 1.2)
    return images, labels


train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# test_ds  = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

input_shape = (64, 64, 3)
num_classes = 200

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
# x_test,  y_test  = dataset_to_numpy(test_ds)



data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomCrop(64, 64),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ],
    name="data_augmentation"
)

def basic_block(x, filters, strides=1, dropout_rate=0.025):
    shortcut = x

    y = layers.Conv2D(filters, 3, strides=strides, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(filters, 3, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # Adjust shortcut if needed
    if x.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same',
                                 kernel_initializer='he_normal', use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([shortcut, y])
    out = layers.ReLU()(out)
    out = layers.Dropout(dropout_rate)(out)
    return out


def build_resnet18(input_shape=(64, 64, 3), num_classes=200, dropout_rate=0.025):
    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)
    
    # x = layers.Rescaling(1.0 / 255)(x)
    # Smaller initial conv (no stride 2)
    
    x = layers.Conv2D(64, 3, strides=1, padding='same',
                      kernel_initializer='he_normal', use_bias=False)(inputs)
    
    # x = layers.Conv2D(64, 3, strides=1, padding='same',
    #                   kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet18 config: [2, 2, 2, 2]
    filters = [64, 128, 256, 512]
    blocks = [2, 2, 2, 2]
    for f, n in zip(filters, blocks):
        for i in range(n):
            stride = 2 if (i == 0 and f != 64) else 1
            x = basic_block(x, f, strides=stride, dropout_rate=dropout_rate)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name="ResNet18_TinyImageNet")




def add_gaussian_noise(images, mean=0, std=0.5):
    noise = np.random.normal(loc=mean, scale=std, size=images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 1) 



def lr_schedule(epoch, lr=0.1):
    if(epoch) > 15:
        lr = 0.05
    return lr
        



lr_scheduler = LearningRateScheduler(lr_schedule)

batch_size = 128


  # for Tiny-ImageNet

def train_teacher():
    print("We made a change")
    cat_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    )
    
    def lr_schedule(epoch, lr=0.1):
        if(epoch) > 15:
            lr = 0.05
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    model = build_resnet18(input_shape=(64,64,3), num_classes=200)
    model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=5e-4, momentum=0.9, nesterov=True), loss=cat_loss, metrics=['accuracy']) #'categorical_crossentropy'
    model.fit(x_train, y_train, batch_size=batch_size, epochs=30, validation_data=(x_val, y_val), callbacks=[lr_scheduler]) #callbacks=[lr_scheduler]

    # Evaluate on validation and test sets explicitly
    val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)
    #test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    model.save(f'ImagenetModels/ImageNetTeacher.h5')

    print(f"Validation accuracy: {val_acc:.4f}")
    # print(f"Test accuracy: {test_acc:.4f}")



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

    epochs = 25
    batch_size = 128
    
    print(f"Training proposed Without Variable Gaus {sigma} with temp {modelno}")
    #To get better accuracy for tinyimgNet we need to use the KDloss, using the standard deviation way does not
    #Yield good accuracy 
    def train_defensive_distillation(model, noisy_x_train, y_train, noisy_x_val, y_val, T, soft_labels):
        kld = tf.keras.losses.KLDivergence()
        def lr_schedule(epoch, lr):
            if epoch > 15:
                return 0.01
            else:
                return 0.1 
    
        
        lr_scheduler_prop = LearningRateScheduler(lr_schedule)

        # student = build_resnet18(input_shape, num_classes)
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

        model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-5, momentum=0.9, nesterov=True), loss=kd_loss, metrics=['accuracy'])
        model.fit(noisy_x_train, soft_labels, batch_size=128, epochs=25, validation_data=(noisy_x_val, y_val),  callbacks=[lr_scheduler_prop])


        return model
    print("test2")
    teacher_model = models.load_model(f"ImagenetModels/ImageNetTeacher.h5")
    #Remove for loop and set T = float(temp)
    for i in range(1,6):
        var = (1/3) * sigma
        if modelno == 1:
            beta = float(sigma)
        elif modelno % 3 == 0 or modelno % 4 == 0:
            beta = sigma - round(random.uniform(0, var), 3)
        else:
            beta = sigma + round(random.uniform(0, var), 3)
            
        noisy_x_train = add_gaussian_noise(x_train, std=beta)
        noisy_x_val = add_gaussian_noise(x_val, std=sigma)
        print(f"Beta is {beta}")
        
        T = float(i + 1)
        eps = 1e-7

        soft_labels = teacher_model.predict(x_train).astype(np.float32)
        soft_labels = np.clip(soft_labels, eps, 1.0)
        soft_labels = soft_labels ** (1.0 / T)
        soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)

        student = build_resnet18(input_shape, num_classes)
        trained_model = train_defensive_distillation(student,noisy_x_train, y_train, noisy_x_val, y_val, T, soft_labels)
        trained_model.save(f'ImagenetModels/Proposed/WVGA/Noise_{sigma}/Student_network{sigma}_modelNo{i}_temp{T}.h5')
        
        
        val_loss, val_acc = trained_model.evaluate(noisy_x_val, y_val, batch_size=batch_size, verbose=1)
        print(f"Noisy validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

        clean_loss, clean_acc = trained_model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        print(f"Clean validation: loss={clean_loss:.4f}, acc={clean_acc:.4f}")
        
        del trained_model
        tf.keras.backend.clear_session()
        gc.collect()




def evaluate_teacher_on_test():
    
    x_train_f = x_train.astype(np.float32)
    x_val_f   = x_val.astype(np.float32)

    if x_train_f.max() > 1.5:
        x_train_f /= 255.0
        x_val_f   /= 255.0

    x_train_f = np.clip(x_train_f, 0.0, 1.0)
    x_val_f   = np.clip(x_val_f,   0.0, 1.0)
    
    teacher_model = models.load_model("ImagenetModels/Proposed/WVGA/Student_sigma0.25_model1_T1.h5.h5")
    teacher_model.trainable = False
    
    p = teacher_model.predict(x_val_f[:32])
    print(p.shape, p.min(), p.max(), p.sum(axis=1).min(), p.sum(axis=1).max())


def train_cohen(sigma):

    print(f"Training RnadSmooth model for sigma {sigma}")
    
    cat_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    )
    
    model = build_resnet18(input_shape=(64,64,3), num_classes=200)
    # noisy_x_train = add_gaussian_noise_tf(x_train, std=sigma)
    noisy_x_val = add_gaussian_noise(x_val, std=sigma)
    noisy_x_train = add_gaussian_noise(x_train, std=sigma)
  
    

    base_lr = 0.05 
    model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=5e-4, momentum=0.9, nesterov=True), loss=cat_loss, metrics=['accuracy'])
    model.fit(noisy_x_train, y_train, batch_size=128, epochs=30, validation_data=(noisy_x_val, y_val),  callbacks=[lr_scheduler])
    model.save(f'ImagenetModels/RandSmoth/RandSmooth_{sigma}.h5')

def train_Sween(sigma, model_no):
    
    for i in range(4,6):
        print(f"Training SWEEN {sigma} modelNo {i}")
        
        model = build_resnet18(input_shape=(64,64,3), num_classes=200)
        # noisy_x_train = add_gaussian_noise_tf(x_train, std=sigma)
        noisy_x_val = add_gaussian_noise(x_val, std=sigma)
        noisy_x_train = add_gaussian_noise(x_train, std=sigma)
    
        loss_is = losses.CategoricalCrossentropy(label_smoothing=0.1) #
        
        # cat_loss = tf.keras.losses.CategoricalCrossentropy(
        # label_smoothing=0.1
        # )

        base_lr = 0.05 
        model.compile(optimizer=optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=5e-4, nesterov=True), loss=loss_is, metrics=['accuracy'])
        model.fit(noisy_x_train, y_train, batch_size=128, epochs=30, validation_data=(noisy_x_val, y_val),  callbacks=[lr_scheduler])
        model.save(f'ImagenetModels/Sween/Sigma_{sigma}/SWEEN{sigma}_modelNo{i}.h5')
 


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

    epochs = 30
    batch_size = 128
    
    print(f"Training proposed Without Variable Gaus {sigma} with temp {modelno}")
    noisy_x_train = add_gaussian_noise(x_train, std=sigma)
    noisy_x_val = add_gaussian_noise(x_val, std=sigma)
    

    student = build_resnet18(input_shape, num_classes) #Remove this is not needed


    
 


    def train_defensive_distillation(model, noisy_x_train, y_train, noisy_x_val, y_val, T, soft_labels):
        kld = tf.keras.losses.KLDivergence()
        def lr_schedule(epoch, lr):
            if epoch > 15:
                return 0.01
            else:
                return 0.1 
    
        
        lr_scheduler_prop = LearningRateScheduler(lr_schedule)

        # student = build_resnet18(input_shape, num_classes)
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

        model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-5, momentum=0.9, nesterov=True), loss=kd_loss, metrics=['accuracy'])
        model.fit(noisy_x_train, soft_labels, batch_size=128, epochs=25, validation_data=(noisy_x_val, y_val),  callbacks=[lr_scheduler_prop])


        return model
    print("test2")
    teacher_model = models.load_model(f"ImagenetModels/ImageNetTeacher.h5")
    #Remove for loop and set T = float(temp)
    for i in range(1,6):
        T = float(i + 1)
        eps = 1e-7

        soft_labels = teacher_model.predict(x_train).astype(np.float32)
        soft_labels = np.clip(soft_labels, eps, 1.0)
        soft_labels = soft_labels ** (1.0 / T)
        soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)

        student = build_resnet18(input_shape, num_classes)
        trained_model = train_defensive_distillation(student,noisy_x_train, y_train, noisy_x_val, y_val, T, soft_labels)
        trained_model.save(f'ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Student_network{sigma}_modelNo{i}_temp{T}.h5')
        
        val_loss, val_acc = trained_model.evaluate(noisy_x_val, y_val, batch_size=batch_size, verbose=1)
        print(f"Noisy validation: loss={val_loss:.4f}, acc={val_acc:.4f}")

        clean_loss, clean_acc = trained_model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        print(f"Clean validation: loss={clean_loss:.4f}, acc={clean_acc:.4f}")
        
        del trained_model
        tf.keras.backend.clear_session()
        gc.collect()

def load_model_with_unique_name(model, temp):
    model._name = f"Model_{temp}_DD_T_{temp}"
    return model


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
        model = tf.keras.models.load_model(f"ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Student_network{sigma}_modelNo{i}.h5", compile=False) #SOTA
        #model = tf.keras.models.load_model(f"Cifar_updated_model/WithNoise/Noise_{sigma}/Student_network{sigma}_temp{i}.h5", compile=False)
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        model = load_model_with_unique_name(model, str(i))
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
 
    
    noisy_x_train = add_gaussian_noise(x_train, 0, args.sigma)
    noisy_x_val = add_gaussian_noise(x_val, 0, args.sigma)
            
    base_model_accuracies = [base_model.evaluate(noisy_x_val, y_val)[1] for base_model in arr_models]
    # Calculate weights based on accuracies
    print(base_model_accuracies)
    weights = np.array(base_model_accuracies) / sum(base_model_accuracies)
    print("The weights are")
    print(weights)
    inputs = layers.Input(shape=(input_shape))
    outputs = layers.average([model(inputs) * weight for model, weight in zip(arr_models, weights)])
    weighted_ensemble = models.Model(inputs, outputs)
    weighted_ensemble.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    #weighted_ensemble.save(f'Cifar_updated_model/cifar_WWE_{str(args.sigma)}.h5')
    weighted_ensemble.save(f'ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Proposed_WOVGA_WE_imgnet_{sigma}.h5')



def MajorityVoting(sigma):
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
        model = tf.keras.models.load_model(f"ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Student_network{sigma}_modelNo{i}.h5", compile=False) #SOTA
        #model = tf.keras.models.load_model(f"Cifar_updated_model/WithNoise/Noise_{sigma}/Student_network{sigma}_temp{i}.h5", compile=False)
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        model = load_model_with_unique_name(model, str(i))
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
 
    
    noisy_x_train = add_gaussian_noise(x_train, 0, args.sigma)
    noisy_x_val = add_gaussian_noise(x_val, 0, args.sigma)
            
    base_model_accuracies = [base_model.evaluate(noisy_x_val, y_val)[1] for base_model in arr_models]
    # Calculate weights based on accuracies
    print(base_model_accuracies)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.5])
    print("The weights are")
    print(weights)
    inputs = layers.Input(shape=(input_shape))
    outputs = layers.average([model(inputs) * weight for model, weight in zip(arr_models, weights)])
    weighted_ensemble = models.Model(inputs, outputs)
    weighted_ensemble.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    #weighted_ensemble.save(f'Cifar_updated_model/cifar_WWE_{str(args.sigma)}.h5')
    weighted_ensemble.save(f'ImagenetModels/Proposed/WOVGA/Noise_{sigma}/Proposed_WOVGA_MV_imgnet_{sigma}.h5')


if __name__ == "__main__":
    
    print(args.WWE_or_base)
    if(args.WWE_or_base == 1):
        train_teacher()
    elif(args.WWE_or_base == 2):
        train_cohen(args.sigma)
    elif(args.WWE_or_base == 3):
        train_Sween(args.sigma, args.modelNo)
    elif(args.WWE_or_base == 4):
        WeightedEnsemble(args.sigma)
    elif(args.WWE_or_base == 5):
        train_model(args.sigma, args.modelNo, args.temp)
    elif(args.WWE_or_base == 6):
        evaluate_teacher_on_test()
    elif(args.WWE_or_base == 7):
        train_proposed_WOVGA(args.sigma, args.modelNo, args.temp)
    elif(args.WWE_or_base == 8):
        MajorityVoting(args.sigma)
    # elif(args.WWE_or_base == 4):
    #     print(f"Train Proposed without VGA: modelNo{args.modelNo}")
    #     train_proposed_WOVGA(args.sigma, args.modelNo, args.temp)
    # elif(args.WWE_or_base == 5):
    #     WeightedEnsemble(args.sigma)
    # elif(args.WWE_or_base == 6):
    #     MajorityVoting(args.sigma)
          
    
# print(y_train[0])
# print("Next")
