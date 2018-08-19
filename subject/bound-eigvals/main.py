import numpy as np
import scipy.linalg as sla
import scipy.signal
from collections import deque, UserDict

import matplotlib.pyplot as plt

import ipdb


class System():
    def __init__(self, args):
        self.args = args
        self.unc = Uncertainty(args)

    def reset(self):
        return np.zeros(2)

    def step(self, t, x, u):
        args = self.args

        next_x = x + args.t_step * (
            args.A.dot(x[:, np.newaxis])
            + args.B.dot(self.unc.Lambda).dot(
                (u + self.unc.delta(x))[:, np.newaxis])
        ).ravel()

        return next_x


class Uncertainty():
    def __init__(self, args):
        self.W = args.W
        self.Lambda = args.Lambda

    def basis(self, x):
        return np.hstack((x, np.abs(x)*x[1], x[0]**3))

    def delta(self, x):
        return self.W.T.dot(self.basis(x))


class DirectMrac():
    def __init__(self, system):
        self.basis = system.unc.basis
        self.args = system.args

        self.P = sla.solve_lyapunov(self.args.A.T, self.args.Q_lyap)


class Cmrac():
    def __init__(self, system):
        self.basis = system.unc.basis
        self.args = system.args

        self.P = sla.solve_lyapunov(self.args.A.T, -self.args.Q_lyap)
        self.Bhat = sla.pinv(args.B)

    def reset(self):
        args = self.args

        return Arguments(
            xr=np.zeros(args.ndim_state),
            what=np.zeros((args.ndim_basis, args.ndim_input)),
            phif=np.zeros(args.ndim_basis),
            z=np.zeros(args.ndim_input),
            basissum=np.zeros((args.ndim_basis, args.ndim_basis)),
            estsum=np.zeros((args.ndim_basis, args.ndim_input)),
            a=1, b=0,
        )

    def get_inputs(self, t, x, uvar):
        args = self.args

        what = uvar.what
        c = self.command(t)

        u_n = args.Kr.dot(c)
        u_a = - what.T.dot(self.basis(x))

        return u_n + u_a

    def get_ua(self, t, x, uvar):
        what = uvar.what
        return - what.T.dot(self.basis(x))

    def step(self, t, x, uvar):
        args = self.args

        xr = uvar.xr
        what = uvar.what
        phif = uvar.phif
        z = uvar.z
        basissum = uvar.basissum
        estsum = uvar.estsum
        a, b = uvar.a, uvar.b

        u = self.get_ua(t, x, uvar)

        e = x - xr
        c = self.command(t)
        phi = self.basis(x)
        esterror = z + self.Bhat.dot(e[: np.newaxis]).ravel() / args.tau_f

        next_xr = xr + args.t_step * (
            args.A.dot(xr[:, np.newaxis]) + args.Br.dot(c[:, np.newaxis])
        ).ravel()

        next_what = what + args.t_step * (
            args.g_direct * phi[:, np.newaxis]
            * e.T.dot(self.P).dot(args.B)
            - args.g_indirect * (basissum.dot(what) - estsum)
        )

        next_phif = phif + args.t_step * (
            - (phif - phi) / args.tau_f
        ).ravel()

        next_z = z + args.t_step * (
            (- u[:, np.newaxis] - self.Bhat.dot(
                np.eye(args.ndim_state) / args.tau_f + args.A
            ).dot(e[:, np.newaxis]) - z[:, np.newaxis]) / args.tau_f
        ).ravel()

        eigs, _ = eig_thr(basissum, args.thr)
        self.lmax = args.lambda_max * np.tanh(0.6*t) + 1e-3
        # self.lmax = np.clip(1.001 * max(eigs), 0.1, 3)

        if max(eigs) != 0:
            a, b = self.findab(basissum, phif)
        elif sla.norm(phif) > args.thr:
            a, b = 0.999, self.lmax / sla.norm(phif)**2

        # print(a, b)

        next_basissum = a * basissum + b * np.outer(phif, phif)
        next_estsum = a * estsum + b * np.outer(phif, esterror)

        return Arguments(xr=next_xr, what=next_what, phif=next_phif,
                         z=next_z, basissum=next_basissum, estsum=next_estsum,
                         a=a, b=b)

    def command(self, t):
        c = 5 * np.deg2rad(
            scipy.signal.square([t * 2*np.pi / 20])
            # - scipy.signal.square([t * 2*np.pi / 20 - np.pi/4])
        )
        return c

    def findab(self, A, y):
        # ipdb.set_trace()
        args = self.args

        eigs, eigv = eig_thr(A, args.thr)

        # if np.any(eigs == 0):
        #     nz = np.where(eigs == 0)[0][-1]
        # else:
        #     nz = 0

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

        lmax = self.lmax

        if gn == ln:
            return lmax / 2 / ln, lmax / 2 / nv**2

        k1 = (l1 + v1**2/nv**2 * g1) / (l1 + g1)
        kn = (ln - vn**2/nv**2 * gn) / (ln - gn)

        p = lmax / ln * np.sqrt(
            (1 - 1/kn) * ln / (ln - gn) * (ln / kn / (ln - gn) - 1))
        s0 = np.arctan2(
            lmax/ln * (2 * (1 - 1/kn) * ln - gn) / 2 / (ln - gn), p)

        eargs = Arguments(l1=l1, ln=ln, g1=g1, gn=gn, nv=nv, k1=k1, kn=kn, p=p,
                          s0=s0, v1=v1, lmax=lmax)
        self.eargs = eargs

        s = scipy.optimize.minimize_scalar(
            self._findr,
            method='bounded',
            bounds=(- np.pi/2 + s0 + 1e-8, np.pi/2 + s0 - 1e-8))

        if s.fun < 0:
            a, b = self._findab_from_s(np.clip(s.x, -1e-7, None))
        else:
            a, b = 1, 0

        if a <= 0:
            ipdb.set_trace()

        return a, b

    def _findr(self, s):
        p, s0, kn, l1, g1, nv, v1 = map(
            self.eargs.get, ['p', 's0', 'kn', 'l1', 'g1', 'nv', 'v1'])

        a, b = self._findab_from_s(s)
        r = (
            a * (1 + 0.5 * g1)
            + 0.5 * b * nv**2
            - 0.5 * np.sqrt((a * g1 + b * nv**2)**2 - 4 * a * b * g1 * v1**2)
        )

        return -r

    def _findab_from_s(self, s):
        p, s0, kn, ln, gn, nv, lmax = map(
            self.eargs.get, ['p', 's0', 'kn', 'ln', 'gn', 'nv', 'lmax'])

        a = - p / np.cos(s - s0) - p * np.tan(s - s0) + lmax / (kn * (ln - gn))
        b = 2 * ln / kn / nv**2 * (
            p * np.tan(s - s0)
            + lmax/ln * (2 * (1 - 1/kn) * ln - gn) / 2 / (ln - gn))

        return a, b


class Arguments(UserDict):
    def __getattr__(self, name):
        return self.data[name]

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value


class Data(Arguments):
    def append(self, name, val):
        val = np.atleast_1d(val)[np.newaxis, :]
        if name not in self:
            self[name] = val
        else:
            self[name] = np.append(self[name], val, axis=0)

    def ele_plot(self, name):
        x = self.time
        y = self[name]

        plt.plot(x, y)


def eig_thr(A, thr):
    eigs, eigv = sla.eig(A)
    sort = np.argsort(np.real(eigs))  # sort in ascending order
    eigs = np.real(eigs[sort])
    eigv = np.real(eigv[:, sort])
    eigs[eigs < thr] = 0
    return eigs, eigv


def plot_basis(data):
    system = data.system
    x = data.time

    plt.figure()
    plt.plot(x, data.reference_model, 'k--')
    plt.plot(x, data.state, 'r')
    plt.title('States')

    plt.figure()
    for w in args.W.ravel():
        plt.axhline(y=w, c='k', ls='--')
    plt.plot(x, data.what, 'r')
    plt.title('Parameter')

    plt.figure()
    basis = np.array([system.unc.basis(state).tolist()
                      for state in data.state])
    plt.plot(x, basis)

    plt.show()


def plot_phif(data):
    x = data.time
    args = data.args

    plt.figure()
    plt.plot(x, list(map(
        lambda phif: args.W.T.dot(phif), data.phif)), 'k')
    plt.plot(x, list(map(
        lambda z, e: z - 1/args.tau_f * data.control.Bhat.dot(e),
        data.z, data.e)), 'b--')
    plt.plot(x, list(map(
        lambda what, phif: what.T.dot(phif), data.what, data.phif)), 'r--')

    plt.figure()
    y = np.zeros_like(data.phif)
    for i, (A, phif) in enumerate(zip(data.basissum, data.phif)):
        V = eig_thr(A, args.thr)[1]
        y[i] = V.T.dot(phif)

    plt.plot(x, y[:, [0, -1]])
    # plt.legend(['1', '2', '3', '4', '5'])

    plt.show()


def plot_eigvals(data):
    x = data.time
    # args = data.args

    plt.figure()
    plt.plot(x, list(map(
        lambda basissum: eig_thr(basissum, data.args.thr)[0], data.basissum)))

    plt.figure()
    plt.subplot(211)
    plt.plot(x, data.a)
    plt.subplot(212)
    plt.plot(x, data.b)

    plt.figure()
    plt.plot(x, list(map(
        lambda basissum: eig_thr(basissum, data.args.thr)[0][0],
        data.basissum)))

    plt.show()


def main(args):
    system = System(args)
    control = Cmrac(system)

    x = system.reset()
    uvar = control.reset()

    data = Data()
    data.args = args
    data.system = system
    data.control = control

    for t in args.ts:
        # get a control input
        u = control.get_inputs(t, x, uvar)

        # steps
        next_x = system.step(t, x, u)
        next_uvar = control.step(t, x, uvar)

        data.append('time', t)
        data.append('state', x)
        data.append('input', u)
        data.append('reference_model', uvar.xr)
        data.append('what', uvar.what.ravel())
        data.append('basissum', uvar.basissum)
        data.append('phif', uvar.phif)
        data.append('z', uvar.z)
        data.append('e', x - uvar.xr)
        data.append('a', uvar.a)
        data.append('b', uvar.b)

        x = next_x
        uvar = next_uvar

    # plot_basis(data)
    # plot_phif(data)
    # plot_eigvals(data)

    return data


if __name__ == '__main__':
    args = Arguments()
    args.A = np.array([[0, 1], [-30, -11]])
    args.B = np.array([[0, 1]]).T
    args.Br = np.array([[0, 1]]).T
    args.Kr = np.array([[1]])
    args.Q_lyap = 1*np.eye(2)
    args.g_direct = 10000
    args.g_indirect = 1
    args.t_step = 0.01
    args.t_final = 40
    args.ndim_state = 2
    args.ndim_input = 1
    args.ndim_basis = 5
    args.ts = np.arange(0, args.t_final, args.t_step)
    args.W = np.array([-18.59521, 15.162375, -62.45153,
                       9.54708, 21.45291])[:, np.newaxis]
    args.Lambda = np.diag([1])
    args.tau_f = 1e-1
    args.lambda_max = 3
    args.thr = 1e-20

    data = main(args)
