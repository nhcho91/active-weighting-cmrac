import numpy as np
import scipy.linalg as sla
import cvxpy as cvx
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


class Clmrac():
    def __init__(self, system):
        self.args = system.args

        self.hstack = deque(maxlen=self.args.ndim_basis)

    def update(self, new_data):
        hstack = self.hstack

        if len(hstack) >= hstack.maxlen:
            for i in range(len(hstack)):
                new_hstack = hstack.copy()
                new_hstack[i] = new_data

        else:
            hstack.apend(new_data)


class Cmrac():
    def __init__(self, system):
        self.basis = system.unc.basis
        self.args = system.args

        delta_num = int(self.args.delta / self.args.t_step)
        self.memory = deque(maxlen=delta_num)

        self.P = sla.solve_lyapunov(self.args.A.T, -self.args.Q_lyap)

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

        # lambdahat = self.lambdahat
        # vhat = self.vhat

        # if args.use_cmrac:
        #     what = vhat.T.dot(np.diag(1 / np.diag(lambdahat)))
        # else:
        #     what = self.what

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
        # v1 = self.v1
        # v2 = self.v2
        # v3 = self.v3
        # lambdahat = self.lambdahat
        # vhat = self.vhat
        c = self.command(t)

        e = x - xr

        # if args.use_cmrac:
        #     what = vhat.T.dot(np.diag(1 / np.diag(lambdahat)))
        # else:
        #     what = self.what

        what = self.what

        self.memory.append((x, e, what))

        x_delta, e_delta, what_delta = self.memory[0]

        # y = e - e_delta - args.A.dot(v3[:, np.newaxis]).ravel()
        # yhat = args.B.dot(
        #     - lambdahat.dot(v1[:, np.newaxis])
        #     + vhat.dot(v2[:, np.newaxis])
        # ).ravel()

        # update reference model
        next_xr = xr + args.t_step * (
            args.A.dot(xr[:, np.newaxis]) + args.Br.dot(c[:, np.newaxis])
        ).ravel()

        # next_v1 = v1 + args.t_step * (
        #     what.T.dot(self.basis(x)[:, np.newaxis])
        #     - what_delta.T.dot(self.basis(x_delta)[:, np.newaxis])
        # ).ravel()

        # next_v2 = v2 + args.t_step * (
        #     self.basis(x) - self.basis(x_delta)
        # )

        # next_v3 = v3 + args.t_step * (e - e_delta)

        # next_lambdahat = lambdahat + args.t_step * (
        #     args.g1 * np.diag(v1)
        #     * np.diag(args.B.T.dot((yhat - y)[:, np.newaxis]))
        # )

        # next_vhat = vhat + args.t_step * (
        #     - args.g2 * args.B.T.dot((yhat - y)[:, np.newaxis]) * v2
        # )

        next_what = what + args.t_step * (
            args.g_direct * self.basis(x)[:, np.newaxis]
            * e.T.dot(self.P).dot(args.B)
        )

        self.xr = next_xr
        # self.v1 = next_v1
        # self.v2 = next_v2
        # self.v3 = next_v3
        # self.lambdahat = next_lambdahat
        # self.vhat = next_vhat
        self.what = next_what

        return dict(reference_model=xr, w_hat=what.ravel())

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


# class Data():
#     def __init__(self, timebase=None, base=None):
#         if type(base) is not int:
#             self.content = np.zeros(timebase.shape + base.shape)
#         else:
#             self.content = np.zeros(timebase.shape + (base, ))

#         self.ts = timebase

#     def append(self, index, value):
#         self.content[index] = value

#     def plot(self):
#         fig, ax = plt.subplots()

#         x = self.ts
#         y = self.content

#         ax.plot(x, y)

#         return fig


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
    plt.plot(x, data.w_hat, 'r')
    plt.title('Parameter')

    plt.figure()
    basis = np.array([system.unc.basis(state).tolist()
                      for state in data.state])
    plt.plot(x, basis)

    plt.show()


def main(args):

    system = System(args)
    control = Cmrac(system)

    x = system.reset()
    control.reset()

    data = Data()
    for i in range(args.ts.size):
        t = args.ts[i]
        u = control.get_inputs(t, x)

        # step
        next_x = system.step(t, x, u)

        # controller update
        current_data = control.update(t, x)

        data.append('time', t)
        data.append('state', x)
        data.append('input', u)

        [data.append(*item) for item in current_data.items()]

        x = next_x

    data.system = system
    data.control = control

    # Plot
    # plot_basis(data)

#     # SDP
#     N = cvx.Parameter(nonneg=True)
#     mineig = []
#     for n in range(10, 21):
#         N.value = n
#         a = cvx.Variable(N.value, nonneg=True)
#         s = cvx.Variable(1, nonneg=True)
#         constr = [sum(a) == 5]

#         basis = np.array([system.unc.basis(state).tolist()
#                           for state in data.state])
#         expr = 0
#         for ai, phi in zip(a, basis[:N.value]):
#             expr += phi[:, np.newaxis] * phi * ai

#         constr += [expr - s * np.eye(5) >> 0]

#         obj = cvx.Maximize(s)
#         prob = cvx.Problem(obj, constr)
#         prob.solve(solver=cvx.CVXOPT, verbose=True)

#         mineig.append(s.value)

    # SDP
    N = cvx.Parameter(nonneg=True)
    mineig = []
    for n in range(10, 22):
        N.value = n
        a = cvx.Variable(N.value, nonneg=True)
        # s = cvx.Variable(1, nonneg=True)
        constr = [sum(a) == 5]

        basis = np.array([system.unc.basis(state).tolist()
                          for state in data.state])
        expr = 0
        for ai, phi in zip(a, basis[:N.value]):
            expr += phi[:, np.newaxis] * phi * ai

        # constr += [expr - s * np.eye(5) >> 0]

        obj = cvx.Maximize(cvx.lambda_min(expr))
        prob = cvx.Problem(obj, constr)
        result = prob.solve(solver=cvx.CVXOPT, verbose=True)

        mineig.append(result)

    plt.figure()
    plt.plot(mineig)

    plt.show()

    return data


if __name__ == '__main__':
    args = Arguments()
    args.A = np.array([[-2, -1], [1, -1]])
    args.B = np.array([[0, 1]]).T
    args.Br = np.array([[0, 1]]).T
    args.Kr = np.array([[1]])
    args.Q_lyap = 100*np.eye(2)
    args.g_direct = 1000
    args.g_indirect = 0
    args.delta = 3
    args.t_step = 0.01
    args.t_final = 40
    args.ndim_state = 2
    args.ndim_input = 1
    args.ndim_basis = 5
    args.ts = np.arange(0, args.t_final, args.t_step)
    args.W = np.array([-18.59521, 15.162375, -62.45153,
                       9.54708, 21.45291])[:, np.newaxis]
    args.Lambda = np.diag([1])

    data = main(args)
