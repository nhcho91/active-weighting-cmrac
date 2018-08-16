import numpy as np
from numpy import tan, cos, arctan, arcsin
import scipy.linalg as sla
import scipy.optimize

import matplotlib.pyplot as plt


def findab(s):
    a = - p / cos(s - s0) - p * tan(s - s0) + lmax / (kn * (ln - gn))
    b = 2 * ln / kn / nv**2 * (
        p * tan(s - s0) + lmax/ln * (2 * (1 - 1/kn) * ln - gn) / 2 / (ln - gn))
    # print(s, a, b)
    return a, b


def findr(s):
    a, b = findab(s)
    r = a * (1 + 0.5 * g1 / l1) + 0.5 * b * nv**2 / l1 - 0.5 / l1 * np.sqrt(
        (a * g1 + b * nv**2)**2 - 4 * a * b * g1 * v1**2)
    return r


def find_maxr():
    tmp = lmax / kn / (ln - gn) / p
    max_s = s0 + arcsin((tmp**2 - 1) / (tmp**2 + 1))
    if False:
        s = scipy.optimize.minimize_scalar(
            lambda s: - findr(s), method='bounded', bounds=(0, max_s))
        res = s.x
    elif False:
        s = scipy.optimize.minimize_scalar(
            lambda s: (1 - findr(s))**2,
            method='bounded', bounds=(0, max_s), tol=1e-9)
        res = np.random.rand() * s.x
        # print(max_s, s.x, res)
    elif True:
        s = scipy.optimize.minimize_scalar(
            lambda s: - findr(s),
            method='bounded',
            bounds=(- np.pi/2 + s0 + 1e-7, np.pi/2 + s0 - 1e-7))
        res = np.clip(s.x, -1e-2, None)

    # print(res)

    return res


D = np.diag([0.01, 1, 6, 8, 10])
V, _ = sla.qr(np.random.rand(5, 5))
A = V.dot(D).dot(V.T)

lmax = 15

minset = []
maxset = []
eigset = np.empty((0, 5))
abset = np.empty((0, 2))

for i in range(10000):
    eigs, eigv = sla.eig(A)
    sort = np.argsort(np.real(eigs))
    eigs = np.real(eigs[sort])
    eigv = np.real(eigv[:, sort])

    minset.append(eigs[0])
    maxset.append(eigs[-1])
    eigset = np.append(eigset, [eigs], axis=0)

    l1 = eigs[0]
    ln = eigs[-1]

    g1 = np.diff(eigs)[0]
    gn = np.diff(eigs)[-1]

    v = np.random.rand(5)
    # v = np.array([-0.3, 0.2, -0.1, 0.5, 0])
    w = eigv.T.dot(v)

    v1 = w[0]
    vn = w[-1]
    nv = sla.norm(v)

    k1 = (l1 + v1**2/nv**2 * g1) / (l1 + g1)
    kn = (ln - vn**2/nv**2 * gn) / (ln - gn)

    # p = np.sqrt((1 - 1/kn) * ln / (ln - gn)**2 * (gn - (1 - 1/kn) * ln))
    p = lmax / ln * np.sqrt((1 - 1/kn) * ln / (ln - gn)
                            * (ln / kn / (ln - gn) - 1))
    s0 = np.arctan2(lmax/ln * (2 * (1 - 1/kn) * ln - gn) / 2 / (ln - gn), p)

    a, b = findab(find_maxr())
    # print(a, b)
    a = np.clip(a, 0, None)
    # b = np.clip(b, 0, None)

    # if abs(g1) < 1e-7:
    #     print('yes')
    #     a, b = 0.99, 0.01
    # if v1**2 / l1 > vn**2 / ln:
    #     a, b = findab(find_maxr())
    # else:
    #     a, b = 1, 0.1
    # if v1**2 / l1 < vn**2 / ln:
    #     a, b = 1, -0.01

    abset = np.append(abset, [[a, b]], axis=0)

    A = a * A + b * np.outer(v, v)

# plt.plot(minset, 'r', label='minimum')
# plt.plot(maxset, 'b', label='maximum')
plt.figure()
plt.plot(eigset)

plt.figure()
plt.plot(abset)

plt.show()
