"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:   2019

"""

import numpy as np

class TrainingData:
    def __init__(self, X, y):
        self.classes, y = np.unique(y, return_inverse=True)
        self.Ni = np.bincount(y)
        self.C  = len(self.classes)
        self.X = np.zeros (X.shape)
        self.y = np.zeros (X.shape[0], dtype = int)
        self.XMap = np.zeros (X.shape[0], dtype = int)
        for c in range(self.C):
            idx1 = sum(self.Ni[0:c])
            idx2 = sum(self.Ni[0:c + 1])
            self.X[idx1:idx2, :] = X[y == c,:]
            self.y[idx1:idx2] = y[y == c]
            self.XMap[y == c] = range (idx1, idx2)

    def setSubclasses(self, Ki, NKi):
        self.Ki = Ki
        self.NKi = NKi

        self.L = np.sum(self.Ki)
        self.label = np.zeros(self.L)
        self.Li    = np.zeros(self.L)
        idx = 0
        for c in range (self.C):
            for k in range (self.Ki[c]):
                self.label[idx] = c
                self.Li[idx] = self.NKi[c, k]
                idx += 1

        self.computeMeans()

    def findSubclasses(self, KMax=1, clusAlg=None, forceKMax=True):
        self.N, self.d = self.X.shape
        self.Ki = np.zeros (self.C, dtype = 'int')
        KMaxMax = np.max(KMax)
        self.NKi = np.zeros ([self.C, KMaxMax], dtype = 'int')
        self.subclassMeans = np.zeros ([self.C, KMaxMax, self.d])
        self.classMeans = np.zeros([self.C,self.d])


        if (KMax == 1).all():
            for c in range (self.C):
                self.Ki[c] = 1
                self.NKi[c,0] = self.Ni[c]
        else:
            if forceKMax:
                for c in range (self.C):
                    idx1 = sum(self.Ni[0:c])
                    idx2 = sum(self.Ni[0:c+1])
                    Xi = self.X[idx1:idx2,:].copy()
                    clusAlg.n_clusters = KMax[c]
                    clusAlg.fit(Xi)
                    skip = 0
                    for k in range(KMax[c]):
                        idx = clusAlg.labels_ == k
                        self.NKi[c,k-skip] = np.sum(idx)
                        if self.NKi[c,k-skip] > 0:
                            subIdx1 = sum(self.NKi[c,0:k-skip])
                            subIdx2 = sum(self.NKi[c,0:k-skip+1])
                            self.X[idx1+subIdx1:idx1+subIdx2, :] = Xi[idx,:]
                        else:
                            skip += 1
                    self.Ki[c] = KMax[c] - skip
            else:
                for c in range (self.C):
                    idx1 = sum(self.Ni[0:c])
                    idx2 = sum(self.Ni[0:c+1])
                    Xi = self.X[idx1:idx2,:].copy()
                    self.classMeans[c, :] = np.mean(Xi, axis = 0)
                    AED = np.zeros(KMax[c])
                    for K in range (KMax[c]):
                        clusAlg.n_clusters = K+1
                        clusAlg.fit(Xi)
                        TED = 0
                        for k in range (K):
                            idx = clusAlg.labels_ == k
                            self.subclassMeans[c,k,:] = np.mean(Xi[idx,:], axis = 0)
                            EDi = np.linalg.norm (self.subclassMeans[c,k,:] - self.classMeans[c,:])
                            TED += EDi
                        AED[K] = TED / K
                    self.Ki[c] = np.argmax(AED) + 1
                    clusAlg.n_clusters = self.Ki[c]
                    clusAlg.fit(Xi)
                    for k in range (self.Ki[c]):
                        idx = clusAlg.labels_ == k
                        self.NKi[c,k] = np.sum(idx)
                        subIdx1 = sum(self.NKi[c, 0:k])
                        subIdx2 = sum(self.NKi[c, 0:k + 1])
                        self.X[idx1+subIdx1:idx1+subIdx2, :] = Xi[idx,:]

        self.L = np.sum(self.Ki)
        self.label = np.zeros(self.L)
        self.Li    = np.zeros(self.L)
        idx = 0
        for c in range (self.C):
            for k in range (self.Ki[c]):
                self.label[idx] = c
                self.Li[idx] = self.NKi[c, k]
                idx += 1
        self.computeMeans()

    def computeMeans (self):
        self.N, self.d = self.X.shape
        self.subclassMeans = np.zeros ([self.C, np.max(self.Ki), self.d])
        self.classMeans = np.zeros([self.C,self.d])
        self.M = np.mean (self.X, axis=0)
        for c in range(self.C):
            idx1 = sum(self.Ni[0:c])
            idx2 = sum(self.Ni[0:c + 1])
            Xi = self.X[idx1:idx2, :]
            self.classMeans[c, :] = np.mean(Xi, axis=0)
            for k in range (self.Ki[c]):
                subIdx1 = sum(self.NKi[c, 0:k])
                subIdx2 = sum(self.NKi[c, 0:k + 1])
                self.subclassMeans[c,k,:] = np.mean(Xi[subIdx1:subIdx2,:], axis=0)
