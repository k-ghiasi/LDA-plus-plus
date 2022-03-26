"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:   2019

"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from LDATrainingDataPreparation import TrainingData

class KernelLinearDiscriminantAnalysis:
    def __init__ (self, trainingData, kernelFunc, solver, regCoef, minSingVal, approximateKernel = False):
        self.kernelFunc = kernelFunc
        self.solver = solver
        self.regCoef = regCoef
        self.minSingVal = minSingVal
        self.classes = trainingData.classes
        self.Ni = trainingData.Ni
        self.C  = trainingData.C
        self.X = trainingData.X
        self.N = trainingData.N
        self.d = trainingData.d
        self.M = trainingData.M
        self.Ki = trainingData.Ki
        self.NKi = trainingData.NKi
        self.L = trainingData.L
        self.Li = trainingData.Li
        self.label = trainingData.label
        self.approximateKernel = approximateKernel

    def computeKernelMatrix (self):
        kerMat = np.zeros([self.N, self.N])
        for i in range (self.N):
           for j in range (self.N):
              kerMat[i,j] = self.kernelFunc(self.X[i,:],self.X[j,:])

        self.sumKerMatCols = np.ones([1,self.N]) @ kerMat
        auxMat  = np.ones([self.N, self.N])/self.N
        sumK    = auxMat @ kerMat
        self.centeredKerMat = kerMat - sumK.T - sumK + sumK @ auxMat
        if self.approximateKernel:
            (eigval, eigvec) = scipy.linalg.eig(self.centeredKerMat)
            minVal = eigval[0]/1000  #define the lowest eigen value used
            m = eigval.shape[0]
            for i in range (eigval.shape[0]):
               if eigval[i] < minVal:
                  eigval[i] = 0
                  m -= 1
            eigvec = eigvec[:,0:m]
            eigval = eigval[0:m]
            self.centeredKerMat=eigvec @ np.diag(eigval) @ eigvec.T

    def computeW(self):
        W = np.zeros([self.N, self.N])
        stBloc = 0
        endBloc = 0
        for c in range(self.C):
            for k in range(self.Ki[c]):
                endBloc = endBloc + self.NKi[c, k]
                for i in range(stBloc, endBloc):
                    for j in range(stBloc, endBloc):
                        W[i, j] = 1 / self.NKi[c, k]
                stBloc = stBloc + self.NKi[c, k]
        return W

    def fitFeatureExtractor(self):
        A = None

        self.computeKernelMatrix()

        if self.solver == 'kfda':
            self.computeCenteredKerMatEigendecomposition()
            eigvecs = self.centeredKerMatEigvecs
            eigvals = self.centeredKerMatEigvals
            W = self.computeW()

            [valBeta, vecBeta] = scipy.linalg.eig(eigvecs.T @ W @ eigvecs)
            # To make the results similar to the GDA paper
            vecBeta = -vecBeta
            #obj = np.sum(valBeta)
            alpha= eigvecs @ np.diag(1/eigvals) @ vecBeta
            A  = np.zeros([self.L-1, self.N])
            for i in range (self.L-1):
                A[i,:] = alpha[:,i] / np.sqrt(alpha[:,i].T @ self.centeredKerMat @ alpha[:,i])

        elif self.solver == 'ghiasi_kfda':
            A  = np.zeros([self.L, self.N])
            self.computeCenteredKerMatEigendecomposition()
            eigvecs = self.centeredKerMatEigvecs
            eigvals = self.centeredKerMatEigvals
            idx = 0
            idxSample = 0
            for c in range (self.C):
                for k in range (self.Ki[c]):
                    v = np.zeros([self.N,1])
                    v[idxSample:idxSample+self.NKi[c,k]] = 1
                    alpha = (eigvecs @ np.diag (1/eigvals)) @ (eigvecs.T @ v)
                    A[idx, :] = alpha[:, 0] / np.sqrt(alpha.T @ self.centeredKerMat @ alpha)
                    idxSample += self.NKi[c,k]
                    idx+=1
        else:
            assert (0)
        self.model = A
        return A

    def computeCenteredKerMatEigendecomposition(self):
        (eigvals, eigvecs) = scipy.linalg.eig(self.centeredKerMat)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        idx = eigvals > self.minSingVal ** 2
        self.centeredKerMatEigvals = eigvals = eigvals[idx]
        self.centeredKerMatEigvecs = eigvecs = eigvecs[:, idx]
        self.centeredKerMatPowHalf = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def mapTrainingDataToExplicitFeatureMap(self):
        if 'centeredKerMatPowHalf' not in self.__dict__:
            self.computeCenteredKerMatEigendecomposition()
        return self.centeredKerMatPowHalf

    def mapTestingDataToExplicitFeatureMap(self, XTest):
        centeredKerMatTest = self.computeCenterdTestKernel(XTest)
        if 'centeredKerMatPowNegHalf' not in self.__dict__:
            eigvals = self.centeredKerMatEigvals
            eigvecs = self.centeredKerMatEigvecs
            self.centeredKerMatPowNegHalf = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
        return  centeredKerMatTest @ self.centeredKerMatPowNegHalf

    def mapFeatureVectorsToExplicitFeatureMap(self):
        if 'centeredKerMatPowHalf' not in self.__dict__:
            self.computeCenteredKerMatEigendecomposition()
        alpha = self.model
        A = self.centeredKerMatPowHalf @ alpha.T
        return A.T

    def computeCenterdTestKernel(self, XTest):
        NTest = XTest.shape[0]
        unit = np.ones([self.N, self.N]) / self.N
        unitTest = np.ones([NTest, self.N]) / self.N
        repeatedSumKerMatCols = np.tile(self.sumKerMatCols,[NTest,1]) / self.N
        KerMatTest = np.zeros([NTest, self.N])
        for i in range (NTest):
            for j in range (self.N):
                KerMatTest[i, j] = self.kernelFunc(XTest[i,:], self.X[j,:])
        centeredKerMatTest = KerMatTest - repeatedSumKerMatCols - \
                             KerMatTest @ unit +  repeatedSumKerMatCols @ unit
        return centeredKerMatTest

    def transform(self, XTest):
        centeredKerMatTest = self.computeCenterdTestKernel(XTest)
        return  centeredKerMatTest @ self.model.T
