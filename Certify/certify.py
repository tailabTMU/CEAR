import argparse
from datasets import get_dataset, DATASETS, get_num_classes
from core_GM import Smooth #core_GM = for GM
import time
# import setGPU
#import tensorflow.compat.v1 as tf
import tensorflow as tf
# from architectures import get_architecture
import datetime
import numpy as np
import os

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved TensorFlow model of the base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size") #1000
parser.add_argument("--skip", type=int, default=20, help="how many examples to skip")
parser.add_argument("--max", type=int, default=3500, help="stop after this many examples") #6500
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use") #Change back 10 000
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

#cifar_2epoch_5_teachers_

if __name__ == "__main__":
 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


    start_ind = 2160    	
    arrFilenames = []
    #datamodel = MNISTModel if args.dataset == DATASETS[0] else CIFARModel
    arr_models = []
 
        # ------------- For Single Network ------------------------------------ #
    # model = tf.keras.models.load_model(args.base_classifier + "RandomizedSmoothing_single_Cifar10_noise_0.5.h5", compile=False)
    # model = tf.keras.models.load_model(args.base_classifier + "Proposed_WOVGA_WWE_1.0.h5", compile=False)
    # model.compile(optimizer='adam',
    #                 loss=tf.keras.losses.CategoricalCrossentropy(),
    #                 metrics=['accuracy'])
    # arr_models.append(model)
    
    #------------ For Mul Network ------------ #
    for i in range(1,6):
        model = tf.keras.models.load_model(args.base_classifier + "Student_network"+str(args.sigma)+"_modelNo"+str(i)+".h5", compile=False)
        # model = tf.saved_model.load(file_model)
        # #model = datamodel(file_model, sess)
        arr_models.append(model)
        
    base_classifier = arr_models
   
    
    print(args.sigma)
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    print(smoothed_classifier)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    dataset = get_dataset(args.dataset, args.split)
    x, label = dataset
    
    for i in range(start_ind,len(x)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max:
            print(args.max)
            break
        before_time = time.time()
        
        
        print(i)
        x_tmp = tf.expand_dims(x[i], axis=0)
        prediction, radius = smoothed_classifier.certify(x_tmp, args.N0, args.N, args.alpha, args.batch)
        after_time = time.time()
        if(args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'imagenet'):
            actual_label = 0
            for j in range(label[i].size):
                if(int(label[i][j]) == 1):
                    actual_label = j
                    #break
            print(actual_label)  
            correct = int(prediction == actual_label)

        else:
            # actual_label = np.argmax(label[i])
            # print(f"hello {actual_label}" )
            correct = int(prediction == label[i][0])

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        if(args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'imagenet'):
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, actual_label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        # elif(args.dataset == 'mnist' or args.dataset == 'cifar10' or args.dataset == 'imagnet'):
        #     print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        #         i, np.argmax(actual_label), prediction, radius, correct, time_elapsed), file=f, flush=True)
        else:
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label[i], prediction, radius, correct, time_elapsed), file=f, flush=True)
    f.close()
