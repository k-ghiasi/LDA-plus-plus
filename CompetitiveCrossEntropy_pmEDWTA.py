"""

Authors: Ramin Zarei-Sabzevar, Sayed Kamaledin Ghiasi-Shirazi
Year:   2021
		This is an extension of a code for Competitive Cross Entropy 
        written by Sayed Kamaledin Ghiasi-Shirazi, available at
        https://github.com/k-ghiasi/CompetitiveCrossEntropy
		
This code is associated with the following paper:
	R. Zarei-Sabzevar, K. Ghiasi-Shirazi and A. Harati, "Prototype-Based Interpretation of the Functionality of Neurons in Winner-Take-All Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3155174.

See also:
	https://github.com/raminzs/pmED-WTA

"""

import numpy as np
import numpy.matlib

class CompetitiveCrossEntropy_pmEDWTA:
    def __init__ (self, trainingData, learning_rate, beta_learning_rate,
                  lr_decay_mult, max_epochs,
                  initialization_epsilon, init_beta):
        self.trainingData = trainingData
        self.learning_rate = learning_rate
        self.beta_learning_rate = beta_learning_rate
        self.max_epochs = max_epochs
        self.lr_decay_mult = lr_decay_mult
        self.init_beta = init_beta
        self.initialization_epsilon = initialization_epsilon
        self.td = trainingData
        self.W_pos = self.td.subclassMeans
        if initialization_epsilon == 'mean':
            self.W_neg = np.tile(np.mean (self.td.X,axis=0),(self.td.L,1))
        elif initialization_epsilon == 'mean_samples_and_features':
            self.W_neg = 0 * self.td.subclassMeans + np.mean (self.td.X)
        else:
            self.W_neg = self.td.subclassMeans * (1 - self.initialization_epsilon)
        self.beta = self.init_beta

    def fit (self, reset=False):
        X = self.td.X
        N, dim = X.shape

        W_pos = self.W_pos
        W_neg = self.W_neg
        beta = self.beta

        if reset:
            W_pos = self.td.subclassMeans
            beta  = self.init_beta            
            if self.initialization_epsilon == 'mean':
                self.W_neg = np.tile(np.mean (self.td.X,axis=0),(self.td.L,1))
            elif self.initialization_epsilon == 'mean_samples_and_features':
                self.W_neg = 0 * self.td.subclassMeans + np.mean (self.td.X)
            else:
                self.W_neg = self.td.subclassMeans * (1 - self.initialization_epsilon)

        alpha = self.learning_rate
        iter = 0
        target_iter = N

        for i in range (self.max_epochs):
            shuffle = np.random.permutation(N)
            for j in range (N):
                n = shuffle[j]
                x = X[n,:]

                iter = iter + 1
                if (iter >= target_iter):
                    alpha = alpha * self.lr_decay_mult
                    iter = 0

                d_X_WPos = W_pos - np.matlib.repmat(x, self.td.L, 1)
                WPos_minus_WNeg = W_pos - W_neg
                WPos_minus_WNeg_sign = np.sign(W_pos - W_neg)
                d_pos_2 = 0.5 * np.sum(d_X_WPos ** 2, axis=1)

                d_X_WNeg = W_neg - np.matlib.repmat(x, self.td.L, 1)
                WNeg_minus_WPos = W_neg - W_pos
                WNeg_minus_WPos_sign = np.sign(W_neg - W_pos)
                d_neg_2 = 0.5 * np.sum(d_X_WNeg ** 2, axis=1)

                d_neg_2_minus_d_pos_2 = d_neg_2 - d_pos_2
                out = beta * d_neg_2_minus_d_pos_2

                t = self.td.y[n]
                out = out - np.max(out)
                z = np.exp(out)
                denum = np.sum(z)
                if denum <= 10e-10:
                    denum = 10e-10
                z = z / denum

                tau = np.zeros(len(z))
                split_grad_pos = np.zeros(len(z))
                split_grad_neg = np.ones(len(z))

                idx1 = t * self.td.K
                idx2 = (t+1) * self.td.K
                denum = np.sum (z[idx1:idx2])
                if (denum <= 10e-10):
                    denum = 10e-10
                tau[idx1:idx2] = z[idx1:idx2] / denum
                split_grad_pos[idx1:idx2] = 1
                split_grad_neg[idx1:idx2] = 0

                dE_dwk = z - tau

                grad_pos = alpha * split_grad_pos * beta * -dE_dwk
                grad_neg = alpha * split_grad_neg * beta * dE_dwk

                W_pos = W_pos - grad_pos[:,np.newaxis] * d_X_WPos
                        
                W_neg = W_neg - grad_neg[:,np.newaxis] * d_X_WNeg 

                dE_d_beta = dE_dwk * d_neg_2_minus_d_pos_2

                beta = beta - self.beta_learning_rate * sum(dE_d_beta)

        self.W_pos = W_pos
        self.W_neg = W_neg 
        self.beta = beta

        self.learning_rate = alpha 


    def classifyByMaxClassifier(self, X_test):
        
        X_test = X_test
        N, dim = X_test.shape
        W_pos = self.W_pos
        W_neg = self.W_neg

        W_pos_2 = np.sum(W_pos**2, axis=1)
        W_neg_2 = np.sum(W_neg**2, axis=1)

        beta = self.beta
        A = beta * (W_pos - W_neg)
        b = - 0.5 * beta * (W_pos_2 - W_neg_2)
        out = X_test @ A.T + np.matlib.repmat(b,N,1)

        y_pred = self.td.label[out.argmax(axis=1)]
        return y_pred
                    
    def transform (self, XTest):
        A = self.beta * (self.W_pos - self.W_neg)
        return XTest @ A.T

    def GenerateImagesOfWeights(self, width, height, color='color',
                        n_images=1, rows=None, cols=None, eps=0, weight='diff'):
        if weight == 'diff':
            A = self.W_pos - self.W_neg
        elif weight == 'pos':
            A = self.W_pos
        elif weight == 'neg':
            A = self.W_neg
        else:
            assert(0)

        n_features_per_image = rows * cols  
        if rows == None or cols == None:
            cols = int(np.sqrt(A.shape[0] - 1)) + 1
            rows = (A.shape[0] + cols - 1) // cols
        images = []
        for picture in range(n_images):
            img = np.ones([rows * (height + 1), cols * (width + 1), 3])
            for nn in range(n_features_per_image):
                n = picture * n_features_per_image + nn
                if (n >= A.shape[0]):
                    continue
                j = nn // rows
                i = nn % rows
                idx1 = i * (height + 1)
                idx2 = j * (width + 1)
                T = max(-np.min(A[n, :]), np.max(A[n, :])) + eps
                if color == 'color':
                    arr_pos = np.maximum(A[n,:] / T, 0)
                    arr_neg = np.maximum(-A[n,:] / T, 0)
                    mcimg_pos = np.reshape(arr_pos, [height, width])  
                    mcimg_neg = np.reshape(arr_neg, [height, width])  
                    mcimg_oth = 0
                elif color == 'gray':
                    if weight == 'diff':
                        arr = A[n, :] / (2 * T) + 0.5
                    else:
                        arr = A[n, :] / T
                    arr = np.maximum(0,arr)
                    mcimg_pos = np.reshape(arr, [height, width])
                    mcimg_neg = mcimg_pos
                    mcimg_oth = mcimg_pos
                else:
                    assert(0)
                    
                img[idx1:idx1 + height, idx2:idx2 + width, 0] = mcimg_pos
                img[idx1:idx1 + height, idx2:idx2 + width, 1] = mcimg_neg
                img[idx1:idx1 + height, idx2:idx2 + width, 2] = mcimg_oth
            images.append(img)
        return images
