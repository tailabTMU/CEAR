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

'''
This core file uses the Geometric mean calculation the other 'Core_og does not

'''

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
        #N = n*len(self.base_classifier)
        N = n
        pABar = self._lower_confidence_bound(nA, N, alpha)
        # print(pABar)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    
    import numpy as np

    def project_to_simplex_batch(self,V):
        """
        Projects each row of V onto the probability simplex:
        nonnegative and sums to 1.

        V: (B, K)
        returns: (B, K)
        """
        B, K = V.shape
        U = np.sort(V, axis=1)[:, ::-1]          # sort each row descending
        cssv = np.cumsum(U, axis=1)
        j = np.arange(1, K + 1)[None, :]        # (1, K)

        cond = U * j > (cssv - 1)
        rho = cond.sum(axis=1) - 1              # (B,)
        rho = np.clip(rho, 0, K - 1)

        theta = (cssv[np.arange(B), rho] - 1) / (rho + 1.0)
        W = V - theta[:, None]
        return np.maximum(W, 0.0)


    def geometric_median_simplex_batch(self,P, max_iter=50, tol=1e-6, eps=1e-12):
        """
        Weiszfeld-style geometric median with simplex projection (batch version).

        P: (M, B, K) probability vectors (M models, batch B, K classes)
        returns: (B, K) geometric median on simplex
        """
        # init: mean then project (keeps it valid)
        q = P.mean(axis=0)                       # (B, K)
        q = self.project_to_simplex_batch(q)

        for _ in range(max_iter):
            q_old = q.copy()

            diff = P - q[None, :, :]             # (M, B, K)
            d = np.linalg.norm(diff, axis=2)     # (M, B)
            d = np.maximum(d, eps)

            w = 1.0 / d                          # (M, B)
            q = (w[:, :, None] * P).sum(axis=0) / w.sum(axis=0)[:, None]

            # enforce probability simplex constraint
            q = self.project_to_simplex_batch(q)

            if np.linalg.norm(q - q_old) < tol:
                break

        return q


    
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
        counts = self._sample_noise(x, n, batch_size, 1)
        top2 = np.argsort(counts)[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        p_value = binom.cdf(count1, count1 + count2, 0.5)

        if p_value > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]



    


    # def _sample_noise(self, x, num, batch_size, nsamp):
    #     """
    #     Sample ensemble prediction under noisy corruptions of x.
    #     Uses geometric-median aggregation of per-model probability outputs.

    #     Returns: counts over classes, shape (num_classes,)
    #     """
    #     counts = np.zeros(self.num_classes, dtype=int)

    #     for _ in range(int(np.ceil(num / batch_size))):
    #         this_batch_size = min(batch_size, num)
    #         num -= this_batch_size

    #         batch = tf.tile(x, [this_batch_size, 1, 1, 1])
    #         noise = tf.random.normal(mean=0, shape=batch.shape, stddev=self.sigma)
    #         batch = tf.cast(batch, tf.float32)
    #         batch_with_noise = batch + noise
    #         batch_with_noise = np.clip(batch_with_noise, 0, 1)

    #         # Collect per-model probabilities: list of (B, K)
    #         probs_list = []
    #         for i in range(len(self.base_classifier)):
    #             probs = self.base_classifier[i](batch_with_noise).numpy()  # (B, K), already probs
    #             probs_list.append(probs)

    #         # Stack to (M, B, K)
    #         P = np.stack(probs_list, axis=0)

    #         # Geometric median on simplex => (B, K)
    #         q_hat = self.geometric_median_simplex_batch(P)

    #         # Vote for each noisy sample
    #         preds = np.argmax(q_hat, axis=1)
    #         counts += self._count_arr(preds, self.num_classes)

    #     return counts
    
    

    def _sample_noise(self, x, num, batch_size, nsamp):
        """
        Sample ensemble prediction under noisy corruptions of x.
        Uses geometric-median aggregation of per-model probability outputs.

        Returns: counts over classes, shape (num_classes,)
        """
        counts = np.zeros(self.num_classes, dtype=int)

        remaining = int(num)
        M = len(self.base_classifier)

        while remaining > 0:
            this_batch_size = min(int(batch_size), remaining)
            remaining -= this_batch_size

            # Build noisy batch on GPU
            batch = tf.tile(x, [this_batch_size, 1, 1, 1])
            batch = tf.cast(batch, tf.float32)

            noise = tf.random.normal(shape=tf.shape(batch), mean=0.0, stddev=self.sigma, dtype=tf.float32)
            batch_with_noise = tf.clip_by_value(batch + noise, 0.0, 1.0)

            # Collect per-model probabilities as TF tensors (stay on GPU)
            probs_tensors = []
            for model in self.base_classifier:
                probs_tf = model(batch_with_noise, training=False)  # (B, K) tensor
                probs_tensors.append(probs_tf)

            # Stack to (M, B, K) on GPU, then copy once to CPU for geometric median
            P_tf = tf.stack(probs_tensors, axis=0)     # (M, B, K) tensor
            P = P_tf.numpy()                           # single host transfer

            # Free GPU tensors as early as possible
            del probs_tensors, P_tf

            # Geometric median on simplex => (B, K) on CPU (numpy)
            q_hat = self.geometric_median_simplex_batch(P)

            # Vote for each noisy sample (CPU)
            preds = np.argmax(q_hat, axis=1)
            counts += self._count_arr(preds, self.num_classes)

            # Cleanup big CPU arrays too
            del P, q_hat, preds, batch, noise, batch_with_noise
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
        
    
