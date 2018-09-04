import numpy as np
import scipy.linalg as sla
import scipy.optimize as sop

from utils import Arguments, eig_thr

import matplotlib.pyplot as plt


class BeAgent():
    def __init__(self, args):
        self.args = args
        self.lmax = args.lmax

    def _make_eargs(self, A, y):
        args = self.args

        eigs, eigv = eig_thr(A, args.thr)

        if np.any(eigs == 0):
            redind = ([np.argmax(eigv[:, eigs == 0].T.dot(y)**2)]
                      + np.where(eigs != 0)[0].tolist())
        else:
            redind = eigs != 0

        eigs = eigs[redind]
        eigv = eigv[:, redind]

        l1, ln = eigs[[0, -1]]
        g1, gn = np.diff(eigs)[[0, -1]]

        v = eigv.T.dot(y)
        v1, vn = v[[0, -1]]
        nv = sla.norm(y)

        k1 = (l1 + v1**2/nv**2 * g1) / (l1 + g1)
        kn = (ln - vn**2/nv**2 * gn) / (ln - gn)

        return Arguments(l1=l1, ln=ln, g1=g1, gn=gn, nv=nv, k1=k1, kn=kn,
                         v1=v1, vn=vn)

    def findab(self, A, y, lmax):
        eargs = self._make_eargs(A, y)
        l1, ln, g1, gn, k1, kn, v1, vn, nv = map(
            eargs.get, ['l1', 'ln', 'g1', 'gn', 'k1', 'kn', 'v1', 'vn', 'nv'])

        p = lmax / ln * np.sqrt(
            (1 - 1/kn) * ln / (ln - gn) * (ln / kn / (ln - gn) - 1))
        s0 = np.arctan2(
            lmax/ln * (2 * (1 - 1/kn) * ln - gn) / 2 / (ln - gn), p)
        eargs.update(lmax=lmax, p=p, s0=s0)

        if gn == ln:
            return lmax / 2 / ln, lmax / 2 / nv**2

        s = sop.minimize_scalar(
            self._findr,
            args=eargs,
            method='bounded',
            bounds=(- np.pi/2 + s0 + 1e-8,
                    np.pi/2 + s0 - 1e-8))

        if s.fun < - eargs.l1:
            a, b = self._findab_from_s(np.clip(s.x, -1e-7, None), eargs)
        else:
            a, b = 1, 0

        return a, b

    def _findr(self, s, eargs):
        l1, g1, nv, v1 = map(
            eargs.get, ['l1', 'g1', 'nv', 'v1'])

        a, b = self._findab_from_s(s, eargs)
        r = (
            a * (l1 + 0.5 * g1)
            + 0.5 * b * nv**2
            - 0.5 * np.sqrt((a * g1 + b * nv**2)**2 - 4 * a * b * g1 * v1**2)
        )

        return -r

    def _findab_from_s(self, s, eargs):
        p, s0, kn, ln, gn, nv, lmax = map(
            eargs.get, ['p', 's0', 'kn', 'ln', 'gn', 'nv', 'lmax'])

        a = - p / np.cos(s - s0) - p * np.tan(s - s0) + lmax / (kn * (ln - gn))
        b = 2 * ln / kn / nv**2 * (
            p * np.tan(s - s0)
            + lmax/ln * (2 * (1 - 1/kn) * ln - gn) / 2 / (ln - gn))

        return a, b

    def _f_from_ab(self, a, b, c, l1, l2, v1, nv):
        # for ln: l1 = l_n, l2 = l_{n-1}, v1 = v_n
        k = l1/l2 - v1**2/nv**2 * (l1/l2 - 1)
        ct1 = (
            a**2 + k * nv**2 / l1 * a * b - (l1 + l2) / l1 / l2 * a * c
            - nv**2 / l1 / l2 * b * c + 1 / l1 / l2 * c**2
        )
        ct2 = a + nv**2 / (l1 + l2) * b - 2 * c / (l1 + l2)
        if l1 <= l2:
            return (self._c_from_ab(a, b, l1, l2, v1, nv)
                    * ((ct1 >= 0) & (ct2 >= 0)))
        else:
            return (self._c_from_ab(a, b, l1, l2, v1, nv)
                    * ((ct1 >= 0) & (ct2 <= 0)))

    def _c_from_ab(self, a, b, l1, l2, v1, nv):
        g = l2 - l1
        return a * l1 + 0.5 * (
            a * g + b * nv**2
            - np.sign(g) * np.sqrt((a * g + b * nv**2)**2
                                   - 4 * a * b * g * v1**2)
        )


if __name__ == '__main__':
    args = Arguments(
        thr=1e-8,
        lmax=10,
    )
    agent = BeAgent(args)

    np.random.seed(1)
    a = np.random.rand(5, 5)
    A = a.T.dot(a)
    y = np.random.rand(5)

    agent.drawab(A, y)
