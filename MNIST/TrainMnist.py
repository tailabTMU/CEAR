import argparse
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
import numpy as np
import random
from tensorflow.keras.callbacks import LearningRateScheduler


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("sigma", type=float, help="noise")
parser.add_argument("WWE_or_base", type=int, help="Select 1: Teacher, 2: to train DD single networks, 3: To train Noisy sginle network, 4: Compile WE")
parser.add_argument("modelNo", type=int, default=1, help="number of the model you want to train")
parser.add_argument("temp", type=int, default=1, help="temperature used to distill the soft logits")
args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 4: Define Model Architecture
def create_base_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),  # No activation function for logits
        ])
    return model

def train_teacher():
    batch_size = 256
    
    def lr_schedule(epoch, lr):
        if epoch < 30:
            return 0.1
        elif epoch < 60:
            return 0.01
        else:
            return 0.001  
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    teacher = create_base_model()

    teacher.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.8, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    teacher.fit(x_train, y_train, batch_size=batch_size, epochs=90, validation_data=(x_test, y_test),  callbacks=[lr_scheduler])
    teacher.save(f'MNIST_updated_model/MNISTTeacher.h5')
    

def train_proposed(sigma, temp, modelno):
   

    temp = temp
    print(f"----Training network with T = {temp}")
    model = create_base_model()
    
    eps = 80

    var = (1/2)*sigma
    
    if(temp == 1):
        beta = sigma
    elif(temp % 3 == 0): 
        beta = sigma - round(random.uniform(0,var), 3)
    else:
        beta = sigma + round(random.uniform(0,var), 3)
    # Step 6: Implement Defensive Distillation
    print(beta)
    

    def train_defensive_distillation(model, teacher, x_train, y_train, x_test, y_test, temp):

        
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                        logits=predicted/temp)
    
        # noise_stddev = 0.5  # Adjust noise standard deviation as needed
        # model.add(layers.GaussianNoise(stddev=noise_stddev))
        def add_gaussian_noise(images, mean, std):
            noise = np.random.normal(loc=mean, scale=std, size=images.shape)
            noisy_images = images + noise
            return np.clip(noisy_images, 0, 1)  # Clip values to be within the valid range [0, 1]

        def lr_schedule(epoch, lr):
            if epoch < 30:
             return 0.1
            elif epoch < 60:
                return 0.01
            else:
             return 0.001  
            return lr
        noisy_x_train = add_gaussian_noise(x_train, 0, beta)
        noisy_x_test = add_gaussian_noise(x_test, 0, sigma)
        
        soft_labels = teacher.predict(x_train)
            
        lr_scheduler = LearningRateScheduler(lr_schedule)
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4, momentum=0.9),
        #             loss=fn,
        #             metrics=['accuracy'])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=fn,
                    metrics=['accuracy'])
        model.fit(noisy_x_train, soft_labels, epochs=eps, validation_data=(noisy_x_test, y_test),batch_size=256)
        #model.fit(noisy_x_train, y_train, epochs=eps, validation_data=(noisy_x_test, y_test),batch_size=256, callbacks=[lr_scheduler])
    teacher_model = models.load_model(f'MNIST_updated_model/MNISTTeacher.h5')
    # Step 5: Compile and Train the Model
    train_defensive_distillation(model, teacher_model, x_train, y_train, x_test, y_test,temp)
    
    model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    model.save(f'MNIST_updated_model/mnist_defensive_distillation_T_{modelno}_noise_{args.sigma}.h5')




def train_without_VGA(sigma, temp, modelno):
    

    temp = temp
    print(f"----Training WITHOUT VGA, network with T = {temp}")
    model = create_base_model()
    
    eps = 80

   
    

    def train_defensive_distillation(model, teacher, x_train, y_train, x_test, y_test, temp):

        
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                        logits=predicted/temp)
    
        # noise_stddev = 0.5  # Adjust noise standard deviation as needed
        # model.add(layers.GaussianNoise(stddev=noise_stddev))
        def add_gaussian_noise(images, mean, std):
            noise = np.random.normal(loc=mean, scale=std, size=images.shape)
            noisy_images = images + noise
            return np.clip(noisy_images, 0, 1)  # Clip values to be within the valid range [0, 1]

        def lr_schedule(epoch, lr):
            if epoch < 30:
             return 0.1
            elif epoch < 60:
                return 0.01
            else:
             return 0.001  
            
        noisy_x_train = add_gaussian_noise(x_train, 0, sigma)
        noisy_x_test = add_gaussian_noise(x_test, 0, sigma)
        
        soft_labels = teacher.predict(x_train)
            
        lr_scheduler = LearningRateScheduler(lr_schedule)
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4, momentum=0.9),
        #             loss=fn,
        #             metrics=['accuracy'])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=fn,
                    metrics=['accuracy'])
        model.fit(noisy_x_train, soft_labels, epochs=eps, validation_data=(noisy_x_test, y_test),batch_size=256)
        #model.fit(noisy_x_train, y_train, epochs=eps, validation_data=(noisy_x_test, y_test),batch_size=256, callbacks=[lr_scheduler])
    teacher_model = models.load_model(f'MNIST_updated_model/MNISTTeacher.h5')
    # Step 5: Compile and Train the Model
    train_defensive_distillation(model, teacher_model, x_train, y_train, x_test, y_test,temp)
    
    model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    model.save(f'MNIST_updated_model/WithoutVGA/mnist_defensive_distillation_T_{modelno}_noise_{args.sigma}.h5')



def train_RS_WE(size, sigma):
    num_ensembles = size
    for i in range(0, num_ensembles+1):
        temp = i+5
        print(f"----Training network number = {temp} with noise {sigma}")
        model = create_base_model()
        
        eps = 75
         

        def add_gaussian_noise(images, mean, std):
            noise = np.random.normal(loc=mean, scale=std, size=images.shape)
            noisy_images = images + noise
            return np.clip(noisy_images, 0, 1)  # Clip values to be within the valid range [0, 1]

        def lr_schedule(epoch, lr):
            if epoch > 0 and epoch % 30 == 0:
                return lr * 0.1  # Reduce the learning rate by a factor of 10 every 30 epochs
            return lr
        
        noisy_x_train = add_gaussian_noise(x_train, 0, sigma)
        noisy_x_test = add_gaussian_noise(x_test, 0, sigma)
        
            
        lr_scheduler = LearningRateScheduler(lr_schedule)
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4, momentum=0.9),
        #             loss=fn,
        #             metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        model.compile(optimizer=optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.8, nesterov=True),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])
        model.fit(noisy_x_train, y_train, epochs=eps, validation_data=(noisy_x_test, y_test),batch_size=128, callbacks=[lr_scheduler])
        
        

        model.save(f'MNIST_updated_model/SOTA_WE/RS_WE_modelNo{temp}_noise{args.sigma}.h5')


def load_model_with_unique_name(model, temp):
    model._name = f"Model_{temp}_DD_T_{temp}"
    return model

def WeightedEnsemble(sigma):
    arr_models = []
    for i in range(1,6):
        model = tf.keras.models.load_model(f'MNIST_updated_model/WithoutVGA/mnist_WOVGA_noise_{sigma}/mnist_defensive_distillation_T_{i}_noise_{sigma}.h5', compile=False)
        #model = tf.keras.models.load_model("MNIST_updated_model/mnist_noise_"+str(args.sigma)+"/mnist_defensive_distillation_T_" + str(i)+"_noise_"+ str(args.sigma) +".h5")
        model.compile(optimizer='sgd',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        model = load_model_with_unique_name(model, str(i))
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
    def add_gaussian_noise(images, mean, std):
        noise = np.random.normal(loc=mean, scale=std, size=images.shape)
        noisy_images = images + noise
        return np.clip(noisy_images, 0, 1)  # Clip values to be within the valid range [0, 1]
    
    noisy_x_train = add_gaussian_noise(x_train, 0, args.sigma)
    noisy_x_test = add_gaussian_noise(x_test, 0, args.sigma)
            
    base_model_accuracies = [base_model.evaluate(noisy_x_test, y_test)[1] for base_model in arr_models]
    # Calculate weights based on accuracies
    weights = np.array(base_model_accuracies) / sum(base_model_accuracies)
    print()
    inputs = layers.Input(shape=(28, 28, 1))
    outputs = layers.average([model(inputs) * weight for model, weight in zip(arr_models, weights)])
    weighted_ensemble = models.Model(inputs, outputs)

    weighted_ensemble.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    weighted_ensemble.save(f'MNIST_updated_model/WithoutVGA/Proposed_WOVGA_WWE_{str(sigma)}.h5')

# Evaluate the weighted ensemble on the test set


def MajorityVoting(sigma):
    arr_models = []
    for i in range(1,6):
        
        model = tf.keras.models.load_model(f'MNIST_updated_model/mnist_noise_{sigma}/mnist_defensive_distillation_T_{i}_noise_{sigma}.h5', compile=False)
        #model = tf.keras.models.load_model("MNIST_updated_model/mnist_noise_"+str(args.sigma)+"/mnist_defensive_distillation_T_" + str(i)+"_noise_"+ str(args.sigma) +".h5")
        model.compile(optimizer='sgd',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])
        
        model = load_model_with_unique_name(model, str(i))
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
    def add_gaussian_noise(images, mean, std):
        noise = np.random.normal(loc=mean, scale=std, size=images.shape)
        noisy_images = images + noise
        return np.clip(noisy_images, 0, 1)  # Clip values to be within the valid range [0, 1]
    
    noisy_x_train = add_gaussian_noise(x_train, 0, args.sigma)
    noisy_x_test = add_gaussian_noise(x_test, 0, args.sigma)
            
    #base_model_accuracies = [base_model.evaluate(noisy_x_test, y_test)[1] for base_model in arr_models]
    # Calculate weights based on accuracies
    weights = np.array([0.2] * 5)
    # weights = 1/5 #Set all the weights to the same value (since we have 5 networks weight = 1/5)

    inputs = layers.Input(shape=(28, 28, 1))
    outputs = layers.average([model(inputs) * weight for model, weight in zip(arr_models, weights)])
    weighted_ensemble = models.Model(inputs, outputs)

    weighted_ensemble.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    weighted_ensemble.save(f'MNIST_updated_model/Proposed_MVE/WVGA/Proposed_WVGA_MV_{str(sigma)}.h5')

# Evaluate the weighted ensemble on the test set



#Train an ensemble of n_size
if __name__ == "__main__":
    
    print(args.WWE_or_base)
    if(args.WWE_or_base == 1):
        WeightedEnsemble(args.sigma)
    elif(args.WWE_or_base == 2):
        train_teacher()
    elif(args.WWE_or_base == 3):
        n_size = 5
        train_RS_WE(n_size,args.sigma)
    elif(args.WWE_or_base == 4):
        train_without_VGA(args.sigma, args.temp, args.modelNo)
    elif(args.WWE_or_base == 5):
        MajorityVoting(args.sigma)
    else:   
        train_proposed(args.sigma, args.temp, args.modelNo)
