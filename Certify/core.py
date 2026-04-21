import tensorflow as tf
import numpy as np
# from scipy.stats import norm, binom_test
from scipy.stats import norm
from math import ceil
import scipy.stats as stats
from setup_cifar import CIFAR, CIFARModel, CIFARDPModel
from setup_mnist import MNIST, MNISTModel, MNISTDPModel
from scipy.stats import binom
import keras as k
import gc


#import setGPU
#from statsmodels.stats.proportion import proportion_confint

class Smooth(object):
    """A smoothed classifier g"""

    # To abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma):
        """
        :param base_classifier: maps from [batch x height x width x channel] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x, n0, n, alpha, batch_size):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        be robust within an L2 ball of radius R around x.

        :param x: the input [batch x height x width x channel]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN, and the radius will be 0.0
        """
        counts_selection = 0
        global counts_estimation 
        counts_estimation = 0
        for i in range(len(self.base_classifier)):
            self.base_classifier[i].trainable = False
            
        # draw samples of f(x + epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, 1)
        #print(counts_selection)
        # draw more samples of f(x + epsilon)
        counts_estimation = (self._sample_noise(x, n, batch_size, 1))
            
        # use these samples to estimate a lower bound on pA
        #final_count_est = sum(counts_estimation)
        # use these samples to take a guess at the top class
        
        cAHat = np.argmax(counts_selection)
        print(cAHat)
        nA = counts_estimation[cAHat]
        N = n*len(self.base_classifier)
        pABar = self._lower_confidence_bound(nA, N, alpha)
        # print(pABar)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x, n, alpha, batch_size):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [batch x height x width x channel]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class or ABSTAIN
        """
        print("inside")
        count1 = 0
        count2 = 0
        for i in range(len(self.base_classifier)):
            self.base_classifier[i].compile(loss='categorical_crossentropy', optimizer='sgd')  # Compile your classifier here
            #counts = self._sample_noise(x, n, batch_size, i)
            counts = counts_estimation[i]
            top2 = np.argsort(counts)[::-1][:2]
            count1 += counts[top2[0]]
            count2 += counts[top2[1]]
        
        p_value = binom.cdf(count1, count1 + count2, 0.5)

        if p_value > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    # def _sample_noise(self, x, num, batch_size, nsamp):
    #         """ Sample the base classifier's prediction under noisy corruptions of the input x.

    #         :param x: the input [batch x height x width x channel]
    #         :param num: number of samples to collect
    #         :param batch_size:
    #         :return: a numpy array of shape (num_classes,) containing the per-class counts
    #         nsamp is one of the netowrks from the ensemble
    #         """
    #         counts = np.zeros(self.num_classes, dtype=int)
    #         for _ in range(int(np.ceil(num / batch_size))):
    #             this_batch_size = min(batch_size, num)
    #             num -= this_batch_size

    #             #batch = tf.tile(x, (this_batch_size, 1, 1, 1))
    #             batch = tf.tile(x, [this_batch_size, 1, 1, 1])
    #             noise = tf.random.normal(mean=0, shape=batch.shape, stddev=self.sigma)
    #             #noise = np.random.normal(scale=self.sigma, size=batch.shape).astype(np.float32)
    #             batch = tf.cast(batch, tf.float32)
    #             batch_with_noise = batch + noise
    #             batch_with_noise = np.clip(batch_with_noise, 0, 1) 
    #             for i in range(len(self.base_classifier)):
    #                 #predictions = self.base_classifier[i].predict(batch_with_noise).eval().argmax(axis=1)
    #                 predictions = self.base_classifier[i](batch_with_noise).numpy().argmax(axis=1) 
    #                 counts += self._count_arr(predictions, self.num_classes)
    #         del predictions
    #         del batch, noise, batch_with_noise
    #         gc.collect()
    #         return counts
    
    #Use this one for imagnet and the one above for the rest 
    def _sample_noise(self, x, num, batch_size, nsamp):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        #         :param x: the input [batch x height x width x channel]
        #         :param num: number of samples to collect
        #         :param batch_size:
        #         :return: a numpy array of shape (num_classes,) containing the per-class counts
        #         nsamp is one of the netowrks from the ensemble
        #         """
        counts = np.zeros(self.num_classes, dtype=int)

        remaining = num
        while remaining > 0:
            this_batch_size = min(batch_size, remaining)
            remaining -= this_batch_size

            batch = tf.tile(x, [this_batch_size, 1, 1, 1])
            batch = tf.cast(batch, tf.float32)

            noise = tf.random.normal(
                shape=tf.shape(batch),
                mean=0.0,
                stddev=self.sigma,
                dtype=tf.float32
            )

            batch_with_noise = tf.clip_by_value(batch + noise, 0.0, 1.0)

            for model in self.base_classifier:
                probs = model(batch_with_noise, training=False)
                predictions = tf.argmax(probs, axis=1).numpy()  # small (B,)
                counts += self._count_arr(predictions, self.num_classes)

                del probs, predictions

            del batch, noise, batch_with_noise
            gc.collect()

        return counts



    def _count_arr(self, arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts
    
    def _lower_confidence_bound(self, NA, N, alpha):
        """ Returns a (1 - alpha) lower confidence bound on a Bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true with at least (1 - alpha) probability over the samples
        """
        lower_bound = stats.beta.ppf(alpha / 2, NA, N - NA + 1)
        return lower_bound
        
    
