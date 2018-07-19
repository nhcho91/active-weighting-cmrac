import numpy as np
import scipy.linalg as sla
from collections import deque, UserDict

import matplotlib.pyplot as plt

import ipdb


class System():
    def __init__(self, args):
        self.args = args
        self.unc = Uncertainty(args)

    def reset(self):
        return np.zeros(2)

    def step(self, x, u):
        args = self.args

        next_x = x + args.t_step * (
            args.A.dot(x[:, np.newaxis])
            + args.B.dot(self.unc.Lambda).dot(
                (u + self.unc.delta(x))[:, np.newaxis])
        ).ravel()

        return next_x


class Uncertainty():
    def __init__(self, args):
        self.W = np.array([-18.59521, 15.162375, -62.45153,
                           9.54708, 21.45291])[:, np.newaxis]
        self.Lambda = np.diag([0.7])

    def basis(self, x):
        return np.hstack((x, np.abs(x)*x[1], x[0]**3))

    def delta(self, x):
        return self.W.T.dot(self.basis(x))


class Cmrac():
    def __init__(self, system):
        self.basis = system.unc.basis
        self.args = system.args

        delta_num = int(self.args.delta / self.args.t_step)
        self.memory = deque(maxlen=delta_num)

        self.P = sla.solve_lyapunov(self.args.A.T, self.args.Q_lyap)

    def reset(self):
        args = self.args
        self.xr = np.zeros(args.ndim_state)
        self.v1 = np.zeros(args.ndim_input)
        self.v2 = np.zeros(args.ndim_basis)
        self.v3 = np.zeros(args.ndim_state)
        self.lambdahat = np.eye(args.ndim_input)
        self.vhat = self.lambdahat.dot(
            np.zeros((args.ndim_basis, args.ndim_input)).T
        )
        self.what = np.zeros((args.ndim_basis, args.ndim_input))
        # self.vhat = self.lambdahat.dot(self.what.T)

        self.memory.clear()

        return self.xr, self.v1, self.v2, self.v3, self.lambdahat, self.vhat

    def get_inputs(self, t, x):
        args = self.args

        lambdahat = self.lambdahat
        vhat = self.vhat

        if args.use_cmrac:
            what = vhat.T.dot(np.diag(1 / np.diag(lambdahat)))
        else:
            what = self.what

        c = self.command(t)

        # u_n = 0*np.diag(1 / np.diag(lambdahat)).dot(args.Kr).dot(c)
        u_n = args.Kr.dot(c)
        u_a = - what.T.dot(self.basis(x))

        return u_n + u_a

    def update(self, t, x):
        args = self.args

        # realize variables
        xr = self.xr
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3
        lambdahat = self.lambdahat
        vhat = self.vhat
        c = self.command(t)

        e = x - xr

        if args.use_cmrac:
            what = vhat.T.dot(np.diag(1 / np.diag(lambdahat)))
        else:
            what = self.what

        self.memory.append((x, e, what))

        x_delta, e_delta, what_delta = self.memory[0]

        y = e - e_delta - args.A.dot(v3[:, np.newaxis]).ravel()
        yhat = args.B.dot(
            - lambdahat.dot(v1[:, np.newaxis])
            + vhat.dot(v2[:, np.newaxis])
        ).ravel()

        # update reference model
        next_xr = xr + args.t_step * (
            args.A.dot(xr[:, np.newaxis]) + args.Br.dot(c[:, np.newaxis])
        ).ravel()

        next_v1 = v1 + args.t_step * (
            what.T.dot(self.basis(x)[:, np.newaxis])
            - what_delta.T.dot(self.basis(x_delta)[:, np.newaxis])
        ).ravel()

        next_v2 = v2 + args.t_step * (
            self.basis(x) - self.basis(x_delta)
        )

        next_v3 = v3 + args.t_step * (e - e_delta)

        next_lambdahat = lambdahat + args.t_step * (
            args.g1 * np.diag(v1)
            * np.diag(args.B.T.dot((yhat - y)[:, np.newaxis]))
        )

        next_vhat = vhat + args.t_step * (
            - args.g2 * args.B.T.dot((yhat - y)[:, np.newaxis]) * v2
        )

        next_what = what + args.t_step * (
            args.g3 * self.basis(x)[:, np.newaxis]
            * e.T.dot(self.P).dot(args.B)
        )

        self.xr = next_xr
        self.v1 = next_v1
        self.v2 = next_v2
        self.v3 = next_v3
        self.lambdahat = next_lambdahat
        self.vhat = next_vhat
        self.what = next_what

        # next_what = - args.g1 * what.dot(np.diag(self.

    def command(self, t):
        if t < 10:
            c = 1
        elif t < 20:
            c = -1
        elif t < 30:
            c = 1
        elif t < 40:
            c = -1
        else:
            c = 0

        return np.deg2rad(5*np.array([c]))


class Arguments(UserDict):
    def __getattr__(self, name):
        return self.data[name]

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value


class Data():
    def __init__(self, timebase=None, base=None):
        if type(base) is not int:
            self.content = np.zeros(timebase.shape + base.shape)
        else:
            self.content = np.zeros(timebase.shape + (base, ))

        self.ts = timebase

    def append(self, index, value):
        self.content[index] = value

    def plot(self):
        fig, ax = plt.subplots()

        x = self.ts
        y = self.content

        ax.plot(x, y)

        return fig


def main():
    args = Arguments()
    args.A = np.array([[-2, -1], [1, -1]])
    args.B = np.array([[0, 1]]).T
    args.Br = np.array([[0, 1]]).T
    args.Kr = np.array([[1]])
    args.Q_lyap = np.eye(2)
    args.g1 = 10
    args.g2 = 10
    args.g3 = 100
    args.delta = 3
    args.t_step = 0.01
    args.t_final = 40
    args.ndim_state = 2
    args.ndim_input = 1
    args.ndim_basis = 5
    args.ts = np.arange(0, args.t_final, args.t_step)
    args.use_cmrac = False

    system = System(args)
    control = Cmrac(system)

    data = Arguments()

    x = system.reset()
    data.xs = Data(args.ts, x)

    control.reset()

    data.us = Data(args.ts, 1)

    for i in range(args.ts.size):
        t = args.ts[i]
        u = control.get_inputs(t, x)

        # step
        next_x = system.step(x, u)

        # controller update
        control.update(t, x)

        data.xs.append(i, x)
        data.us.append(i, u)

        x = next_x

    data.xs.plot()
    data.us.plot()

    plt.show()


if __name__ == '__main__':
    main()
