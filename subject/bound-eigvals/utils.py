from collections import UserDict
import numpy as np
import scipy.linalg as sla


class Arguments(UserDict):
    def __getattr__(self, name):
        return self.data[name]

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value


class Record(Arguments):
    def append(self, name, val):
        val = np.atleast_1d(val)[np.newaxis, :]
        if name not in self:
            self[name] = val
        else:
            self[name] = np.append(self[name], val, axis=0)


def eig_thr(A, thr):
    eigs, eigv = sla.eig(A)
    sort = np.argsort(np.real(eigs))  # sort in ascending order
    eigs = np.real(eigs[sort])
    eigv = np.real(eigv[:, sort])
    eigs[eigs < thr] = 0
    return eigs, eigv
