"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:   2019

"""


import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from LDATrainingDataPreparation import TrainingData

class LinearDiscriminantAnalysis:
    def __init__ (self, trainingData, solver, S1_, S2_, regCoef, minSingVal):
        self.solver = solver
        self.S1_ = S1_
        self.S2_ = S2_
        self.regCoef = regCoef
        self.minSingVal = minSingVal

        self.classes = trainingData.classes
        self.Ni = trainingData.Ni
        self.C  = trainingData.C
        self.X = trainingData.X
        self.N = trainingData.N
        self.d = trainingData.d
        self.classMeans = trainingData.classMeans
        self.M = trainingData.M
        self.Ki = trainingData.Ki
        self.NKi = trainingData.NKi
        self.subclassMeans = trainingData.subclassMeans
        self.L = trainingData.L
        self.Li = trainingData.Li
        self.label = trainingData.label

        
    def computeSb(self):
        if 'svd' not in self.solver:
            self.Sb = 0
            for c in range (self.C):
                for k in range (self.Ki[c]):
                    Mck = self.subclassMeans[c,k,:] - self.M
                    self.Sb = self.Sb + self.NKi[c,k] / self.N * np.outer (Mck, Mck)
            result = self.Sb
        else:
            result = None

        self.subclassWeight = np.zeros(self.L)
        idx = 0
        for c in range (self.C):
            for k in range (self.Ki[c]):
                self.subclassWeight[idx] = self.NKi[c,k] / self.N
                idx += 1

        return result


    def computeSw(self):
        if 'svd' not in self.solver:
            self.Sw = 0
            Xnew = (self.X).copy()
            idx1 = 0
            for c in range (self.C):
                for k in range (self.Ki[c]):
                    Mck = self.subclassMeans[c, k, :]
                    idx2 = idx1 + self.NKi[c, k]
                    Xnew[idx1:idx2,:] -= Mck
                    idx1 = idx2
            self.Sw = Xnew.T @ Xnew  / self.N
            self.Sw = self.Sw + self.regCoef * np.diag (np.ones(self.d))
            result = self.Sw
        elif 'svd' in self.solver:
            A = np.zeros([self.N, self.d])
            idx = 0
            for c in range (self.C):
                for k in range (self.Ki[c]):
                    n = self.NKi[c, k]
                    A[idx:idx+n, :] = (self.X[idx:idx+n,:] - self.subclassMeans[c, k, :])
                    idx += n
            A = A / np.sqrt(self.N)
            U, S, VH = np.linalg.svd( A, full_matrices=False)
            rank = np.sum(S > self.minSingVal)
            self.Sw_eigvec = VH[:rank].T
            self.Sw_singval = np.sqrt(S[:rank] ** 2 + self.regCoef)
            result = (self.Sw_eigvec, self.Sw_singval)
        else:
            assert (0)
        return result


    def computeSt(self):
        if 'svd' not in self.solver:
            X = self.X - self.M
            self.St = 1 / self.N * X.T @ X
            self.St = self.St + self.regCoef * np.diag (np.ones(self.d))
            result = self.St
        elif 'svd' in self.solver:
            X = (self.X - self.M)/ np.sqrt(self.N)
            U, S, VH = np.linalg.svd( X, full_matrices=False)
            rank = np.sum(S > self.minSingVal)
            self.St_eigvec = VH[:rank].T
            self.St_singval = np.sqrt(S[:rank]**2 + self.regCoef)
            result = (self.St_eigvec, self.St_singval)
        else:
            assert (0)
        return result


    def fitFeatureExtractor(self):
        idx = 0
        self.Mi = np.zeros([self.L, self.d])
        for c in range (self.C):
            for k in range (self.Ki[c]):
                self.Mi[idx, :] = self.subclassMeans[c,k] - self.M
                idx += 1
                    
        if self.solver == 'orthogonal_centroid':
            Q, R = np.linalg.qr (self.Mi.T, mode = 'reduced')
            self.model = Q.T
            return
        
        if self.S1_ == 'Sb':
            if 'S1' not in self.__dict__:
                self.S1 = self.computeSb()
        else:
            assert (0)

        if self.S2_ == 'St':
            if 'St' not in self.__dict__:
                self.S2 = self.computeSt()
        elif self.S2_ == 'Sw':
            if 'Sw' not in self.__dict__:
                self.S2 = self.computeSw()
        else:
            assert (0)

        A = None
        if self.solver == 'eigen':
            (eigvals, eigvecs) = scipy.linalg.eigh(self.S1, self.S2)
            idx = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:,idx]
            eigvecs /= np.linalg.norm(eigvecs, axis=0)
            A = eigvecs[:,0:self.L-1].T
        elif self.solver == 'eigen_np': # Yields the same result as eigen but uses numpy
            (eigvals, eigvecs) = np.linalg.eig(np.linalg.inv(self.S2) @ self.S1)
            idx = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:,idx]
            eigvecs /= np.linalg.norm(eigvecs, axis=0)
            A = eigvecs[:,0:self.L-1].T
        elif self.solver == 'ghiasi_lstsq':
            v = np.linalg.lstsq(self.S2, self.Mi.T, rcond=None)[0]
            A = v.T[0:self.L,:]
        elif self.solver == 'ghiasi_pinv':
            v = np.linalg.pinv(self.S2) @ self.Mi.T
            A = v.T[0:self.L,:]
        elif self.solver == 'svd':
            (U2, singVal) = self.S2
            scaledU2 = U2 / singVal
            X = np.zeros([self.L, U2.shape[0]])
            if self.S1_ == 'Sb':
                idx = 0
                for c in range(self.C):
                    for k in range(self.Ki[c]):
                        X[idx, :] = np.sqrt(self.NKi[c, k] / self.N) * (self.subclassMeans[c, k, :] - self.M)
                        idx += 1
            else:
                assert (0)

            B = X @ scaledU2
            U, S, V = np.linalg.svd(B, full_matrices=False)
            eigvecs = scaledU2 @ V.T
            eigvecs /= np.linalg.norm(eigvecs, axis=0)
            A = eigvecs[:,0:self.L-1].T
        elif self.solver == 'ghiasi_svd':
            (eigenvec, singval) = self.S2
            A = self.Mi @ eigenvec
            A *= np.array(1/singval ** 2)
            A = A @ eigenvec.T
        elif self.solver == 'ghiasi_svd_pca':
            (eigenvec, singval) = self.S2
            A = self.Mi @ eigenvec
            A *= np.array(1/singval)
            A = A @ eigenvec.T    # Kept to allow visualization			
        else:
            assert (0)
        
        A = A.T / np.linalg.norm(A, axis=1)
        self.model = A.T
        return


    def fitBayesLinearClassifier(self):
        if 'svd' not in self.solver:
            if 'SwInv' not in self.__dict__:
                if 'Sw' not in self.__dict__:
                    self.Sw = self.computeSw()
                self.SwInv = np.linalg.pinv(self.Sw)
            self.linclass_weights = self.Mi @ self.SwInv
        elif 'svd' in self.solver:
            if 'Sw_eigvec' not in self.__dict__:
                self.computeSw()
            self.linclass_weights = (self.Mi @ self.Sw_eigvec) @ \
                                    (np.diag(1/self.Sw_singval ** 2) @ self.Sw_eigvec.T)
        self.linclass_biases = -0.5 * (np.diag(self.linclass_weights @ self.Mi.T)) +\
                                np.log (self.Li / self.N)
        self.linclass_biases = self.linclass_biases.reshape ([self.L, 1])

    def classifyByBayesClassifier(self, Xtest):
        A = self.linclass_weights
        b = self.linclass_biases
        d = A @ (Xtest - self.M).T + b
        if 'Sw_eigvec' not in self.__dict__:
            U, S2, VH = np.linalg.svd( self.Sw, full_matrices=False)
            rank = np.sum(S2 > self.minSingVal ** 2)
            self.Sw_eigvec = VH[:rank].T
            self.Sw_singval = np.sqrt(S2[:rank])

        XTimesSigmaHalfInv = (Xtest - self.M) @ self.Sw_eigvec @ np.diag(1 / self.Sw_singval)
        X2 = np.sum (XTimesSigmaHalfInv**2, axis = 1)
        d -= 0.5 * X2
        dClass = np.zeros ([Xtest.shape[0], self.C])
        for c in range (self.C):
            dClass[:,c] = scipy.special.logsumexp(d[self.label == c, :], axis = 0)
        y_pred = dClass.argmax(axis=1)
        return y_pred

    def classifyByMaxClassifier(self, Xtest):
        A = self.linclass_weights
        b = self.linclass_biases
        d = A @ (Xtest - self.M).T + b
        y_pred = self.label[d.argmax(axis=0)]
        return y_pred

    def transform (self, XTest):
        A = self.model
        #if 'ghiasi' in self.solver:
        #    A = A * self.subclassWeight.reshape([A.shape[0], 1])
        return XTest @ A.T

    def transformByProjection (self, XTest):
        A = self.model[0:self.L-1, :].T
        Q, R = np.linalg.qr (A)
        return XTest @ Q
                
    def computeFeatureSpaceScatterMatrices(self):
        A = self.model[0:self.L-1, :]
        # The objective function is invariant to any non-singular transformation
        #if 'ghiasi' in self.solver:
        #    A = A * self.subclassWeight.reshape([A.shape[0], 1])

        if 'St' not in self.__dict__:
            self.St = self.computeSt()
        if 'S1' not in self.__dict__:    # for orthogonal_centroid
            self.S1 = self.computeSb()

        if 'svd' not in self.solver:
            St_Y = A @ self.St @ A.T
            S1_Y = A @ self.S1 @ A.T
        elif 'svd' in self.solver:
            eigvec, singval = self.St
            AVS2 = A @ eigvec @ np.diag(singval)
            St_Y = AVS2 @ AVS2.T

            S1_Y = 0
            if self.S1_ == 'Sb':
                idx = 0
                for c in range(self.C):
                    for k in range(self.Ki[c]):
                        v = np.sqrt(self.NKi[c, k] / self.N) * A @ (self.subclassMeans[c, k, :] - self.M)
                        v = np.reshape(v, [v.shape[0], 1])
                        S1_Y = S1_Y + v @ v.T
                        idx += 1
            else:
                assert (0)
        else:
            assert (0)
        return (S1_Y, St_Y)
    
    def objective(self):
        (S1_Y, St_Y) = self.computeFeatureSpaceScatterMatrices()
        objective = np.trace(np.linalg.pinv(St_Y) @ S1_Y)
        return objective

    #Objective in Generalized Discriminant Analysis
    def objective2(self):
        (S1_Y, S2_Y) = self.computeFeatureSpaceScatterMatrices()
        S2_Y += np.diag(np.ones([S2_Y.shape[0]])) * 1e-10
        (eigvals, eigvecs) = scipy.linalg.eigh(S1_Y, S2_Y)
        objective = np.sum(eigvals)
        return objective
        
    def GenerateImagesOfLinearFeatureExtractorWeights(self, width, height, color = 'color', nImages=1, rows=None, cols=None):
        A = self.model
        nFeaturesPerImage = rows * cols  # (A.shape[0] + 1) // nImages
        if rows == None or cols == None:
            cols = int(np.sqrt(A.shape[0] - 1)) + 1
            rows = (A.shape[0] + cols - 1) // cols
        images = []
        for picture in range(nImages):
            img = np.ones([rows * (height + 1), cols * (width + 1), 3])
            for nn in range(nFeaturesPerImage):
                n = picture * nFeaturesPerImage + nn
                if (n >= A.shape[0]):
                    continue
                j = nn // rows
                i = nn % rows
                idx1 = i * (height + 1)
                idx2 = j * (width + 1)
                T = max(-np.min(A[n, :]), np.max(A[n, :]))
                if color == 'color':
                    arr_pos = np.maximum(A[n,:] / T, 0)
                    arr_neg = np.maximum(-A[n,:] / T, 0)
                    mcimg_pos = np.reshape(arr_pos, [height, width])  
                    mcimg_neg = np.reshape(arr_neg, [height, width])  
                    mcimg_oth = 0
                elif color == 'gray':
                    arr = A[n, :] / (2 * T) + 0.5              
                    mcimg_pos = np.reshape(arr, [height, width])
                    mcimg_neg = mcimg_pos
                    mcimg_oth = mcimg_pos
                elif color == 'gray_abs':
                    #arr = np.minimum(np.abs(A[n,:] / T),1.0)
                    arr = np.abs(A[n,:]) / T
                    mcimg_pos = np.reshape(arr, [height, width])
                    mcimg_neg = mcimg_pos
                    mcimg_oth = mcimg_pos                      
                else:
                    assert (0)
                    
                img[idx1:idx1 + height, idx2:idx2 + width, 0] = mcimg_pos
                img[idx1:idx1 + height, idx2:idx2 + width, 1] = mcimg_neg
                img[idx1:idx1 + height, idx2:idx2 + width, 2] = mcimg_oth
            images.append(img)
        return images
