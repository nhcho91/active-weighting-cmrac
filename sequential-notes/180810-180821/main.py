import numpy as np
import scipy.linalg as sla
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, InsetPosition

from collections import deque, UserDict

from tqdm import tqdm

import ipdb


class System():
    def __init__(self, args):
        self.args = args
        self.unc = Uncertainty(args)

    def reset(self):
        return self.args.xinit

    def step(self, t, x, u):
        args = self.args

        next_x = x + args.t_step * (
            args.A.dot(x[:, np.newaxis])
            + args.B.dot(self.unc.Lambda).dot(
                (u + self.unc.delta(x))[:, np.newaxis])
            + args.Br.dot(self.command(t)[:, np.newaxis])
        ).ravel()

        return next_x


class Uncertainty():
    def __init__(self, args):
        self.W = args.W
        self.Lambda = args.Lambda

    def basis(self, x):
        return np.hstack((x[:2], np.abs(x[:2])*x[1], x[0]**3))

    def delta(self, x):
        return self.W.T.dot(self.basis(x))


class Cmrac():
    def __init__(self, system):
        self.basis = system.unc.basis
        self.args = system.args

        self.P = sla.solve_lyapunov(self.args.A.T, -self.args.Q_lyap)
        self.Bhat = sla.pinv(args.B)
        self.system = system

    def reset(self):
        args = self.args

        x = self.system.reset()
        xr = x.copy()
        what = np.zeros((args.ndim_basis, args.ndim_input))
        e = x - xr
        z = (what.T.dot(self.basis(x))[:, np.newaxis]
             - self.Bhat.dot(np.eye(args.ndim_state)/args.tau_f + args.A).dot(
                 e[:, np.newaxis])).ravel()
        basissum = np.zeros((args.ndim_basis, args.ndim_basis))
        best_basissum = basissum.copy()
        estsum = np.zeros((args.ndim_basis, args.ndim_input))
        best_estsum = estsum.copy()

        return Arguments(
            xr=xr,
            what=what,
            phif=self.basis(np.zeros(args.ndim_state)),
            z=z,
            basissum=basissum,
            estsum=np.zeros((args.ndim_basis, args.ndim_input)),
            a=0, b=1,
            best_basissum=best_basissum,
            best_estsum=best_estsum,
        )

    def get_inputs(self, t, x, uvar):
        what = uvar.what
        u_a = - what.T.dot(self.basis(x))
        return u_a

    def get_ua(self, t, x, uvar):
        return - uvar.what.T.dot(self.basis(x))

    def step(self, t, x, uvar):
        args = self.args

        xr = uvar.xr
        what = uvar.what
        phif = uvar.phif
        z = uvar.z
        basissum = uvar.basissum
        best_basissum = uvar.best_basissum
        estsum = uvar.estsum
        best_estsum = uvar.best_estsum
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

        next_z = z + args.t_step * ((
            - u[:, np.newaxis]
            - self.Bhat.dot(
                np.eye(args.ndim_state) / args.tau_f + args.A
            ).dot(e[:, np.newaxis])
            - z[:, np.newaxis]) / args.tau_f
        ).ravel()

        if args.case == 'MRAC':
            a, b = 0, 0
        elif args.case == 'BECMRAC':
            eigs, _ = eig_thr(basissum, args.thr)
            self.lmax = args.lambda_max * np.tanh(args.lmax_speed*t) + 1e-3
            # self.lmax = np.clip(1.001 * max(eigs), 0.1, 3)

            if max(eigs) != 0:
                a, b = self.findab(basissum, phif)
            elif sla.norm(phif) > args.thr:
                a, b = 0.999, self.lmax / sla.norm(phif)**2
        elif args.case == 'SLSCMRAC':
            a, b = 0, 1
        elif args.case == 'FECMRAC':
            # ipdb.set_trace()
            b = args.t_step
            alp = (
                0.1
                + (10 - 0.1)*np.tanh(0.1 * sla.norm((phif - phi)/args.tau_f))
            )
            a = 1 - alp * args.t_step

            if eig_thr(basissum, 0)[0][0] > eig_thr(best_basissum, 0)[0][0]:
                best_basissum = basissum.copy()
                best_estsum = estsum.copy()
                # print(t, eig_thr(basissum, 0)[0][0])

            next_what = what + args.t_step * (
                args.g_direct * phi[:, np.newaxis]
                * e.T.dot(self.P).dot(args.B)
                - args.g_indirect * (best_basissum.dot(what) - best_estsum)
            )

        next_basissum = a * basissum + b * np.outer(phif, phif)
        next_estsum = a * estsum + b * np.outer(phif, esterror)

        return Arguments(xr=next_xr, what=next_what, phif=next_phif,
                         z=next_z, basissum=next_basissum, estsum=next_estsum,
                         a=a, b=b, best_basissum=best_basissum,
                         best_estsum=best_estsum)

    def command(self, t):
        if t > 10:
            c = scipy.signal.square([(t - 10) * 2*np.pi / 20])
        else:
            c = np.array([0])

        return c

    def findab(self, A, y):
        # ipdb.set_trace()
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

        if s.fun < -l1:
            a, b = self._findab_from_s(np.clip(s.x, -1e-7, None))
        else:
            a, b = 1, 0

        if a <= 0:
            ipdb.set_trace()

        return a, b

    def _findr(self, s):
        l1, g1, nv, v1 = map(
            self.eargs.get, ['l1', 'g1', 'nv', 'v1'])

        a, b = self._findab_from_s(s)
        r = (
            a * (l1 + 0.5 * g1)
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


class Record(Arguments):
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


def plot_basis(env):
    args = env.args
    # system = env.system
    x = args.time

    plt.figure()
    plt.subplot(311)
    plt.plot(x, args.state[:, 0], 'r')
    plt.plot(x, args.reference_model[:, 0], 'k--')
    plt.plot(x, list(map(lambda t: env.control.command(t), x)), 'k--')
    plt.title('States')

    plt.subplot(312)
    plt.plot(x, args.state[:, 1], 'r')
    plt.plot(x, args.reference_model[:, 1], 'k--')

    plt.subplot(313)
    plt.plot(x, args.input, 'r')
    plt.plot(x, args.input, 'r')

    plt.figure()
    for w in args.W.ravel():
        plt.axhline(y=w, c='k', ls='--')
    plt.plot(x, args.what, 'r')
    plt.title('Parameter')

    # plt.figure()
    # basis = np.array([system.unc.basis(state).tolist()
    #                   for state in args.state])
    # plt.plot(x, basis)

    plt.show()


def plot_phif(env):
    args = env.args
    x = args.time

    plt.figure()
    plt.plot(x, list(map(
        lambda phif: args.W.T.dot(phif), args.phif)), 'k')
    plt.plot(x, list(map(
        lambda z, e: z + 1/args.tau_f * env.control.Bhat.dot(e),
        args.z, args.e)), 'b--')
    plt.plot(x, list(map(
        lambda what, phif: what.T.dot(phif), args.what, args.phif)), 'r--')

    plt.figure()
    y = np.zeros_like(args.phif)
    for i, (A, phif) in enumerate(zip(args.basissum, args.phif)):
        V = eig_thr(A, args.thr)[1]
        y[i] = V.T.dot(phif)

    plt.plot(x, y[:, [0, -1]])

    plt.show()


def plot_eigvals(env):
    args = env.args
    x = args.time

    plt.figure()
    plt.plot(x, list(map(
        lambda basissum: eig_thr(basissum, args.thr)[0],
        args.basissum)))

    plt.figure()
    plt.subplot(211)
    plt.plot(x, args.a)
    plt.subplot(212)
    plt.plot(x, args.b)

    plt.figure()
    plt.plot(x, list(map(
        lambda basissum: eig_thr(basissum, args.thr)[0][0],
        args.basissum)))

    if args.case == 'FECMRAC':
        plt.figure()
        plt.plot(x, list(map(
            lambda basissum: eig_thr(basissum, args.thr)[0][0],
            args.best_basissum)))
        plt.title('FE')

    plt.show()


def save(args, filename):
    np.savez(filename, **args)


def load(filename):
    tmp = np.load(filename)
    args = Arguments(**tmp)
    env = make_env(args)
    return env


def make_env(args):
    env = Arguments()
    system = System(args)
    control = Cmrac(system)
    system.command = control.command

    env.system = system
    env.control = control
    env.args = args

    return env


def main(args):
    env = make_env(args)
    system = env.system
    control = env.control

    x = system.reset()
    uvar = control.reset()

    record = Record()

    args.ts = np.arange(0, args.t_final, args.t_step)
    for t in tqdm(args.ts, mininterval=1):
        # get a control input
        u = control.get_inputs(t, x, uvar)

        # steps
        next_x = system.step(t, x, u)
        next_uvar = control.step(t, x, uvar)

        record.append('time', t)
        record.append('state', x)
        record.append('input', u)
        record.append('reference_model', uvar.xr)
        record.append('what', uvar.what.ravel())
        record.append('basissum', uvar.basissum)
        record.append('phif', uvar.phif)
        record.append('z', uvar.z)
        record.append('e', x - uvar.xr)
        record.append('a', uvar.a)
        record.append('b', uvar.b)
        record.append('best_basissum', uvar.best_basissum)

        x = next_x
        uvar = next_uvar

    args.update(**record)

    return env


def compare():
    mrac = load('data/record_mrac.npz')
    # sls = load('data/record_slscmrac.npz')
    be = load('data/record_becmrac.npz')
    fe = load('data/record_fecmrac.npz')

    x = mrac.args.time

    basic_kwargs = dict(c='k', lw=0.7)
    mrac.kwargs = dict(basic_kwargs, c='b', ls='--', label='MRAC')
    fe.kwargs = dict(basic_kwargs, c='b', label='FECMRAC')
    be.kwargs = dict(basic_kwargs, c='r', label='BECMRAC')

    # =================
    # States and inputs
    # =================
    plt.figure(figsize=(12.31, 7.57))

    ax = plt.subplot(311)
    plt.plot(x, list(map(lambda t: mrac.control.command(t), x)),
             **basic_kwargs, ls='-.')
    plt.plot(x, mrac.args.reference_model[:, 0], **basic_kwargs, ls='--')
    plt.plot(x, mrac.args.state[:, 0], **mrac.kwargs)
    plt.plot(fe.args.time, fe.args.state[:, 0], **fe.kwargs)
    plt.plot(be.args.time, be.args.state[:, 0], **be.kwargs)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel(r'$x_1$')
    plt.legend(loc='best')

    ax2 = plt.subplot(312, sharex=ax)
    plt.plot(x, mrac.args.reference_model[:, 1], **basic_kwargs, ls='--')
    plt.plot(x, mrac.args.state[:, 1], **mrac.kwargs)
    plt.plot(fe.args.time, fe.args.state[:, 1], **fe.kwargs)
    plt.plot(be.args.time, be.args.state[:, 1], **be.kwargs)
    plt.ylabel(r'$x_2$')
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax)
    plt.plot(x, mrac.args.input, **mrac.kwargs)
    plt.plot(fe.args.time, fe.args.input, **fe.kwargs)
    plt.plot(be.args.time, be.args.input, **be.kwargs)
    plt.ylabel(r'$u$')
    plt.xlabel('Time, sec')

    axins = zoomed_inset_axes(ax2, 2.5, loc='lower center')
    axins.set_axes_locator(InsetPosition(ax2, [0.15, -0.3, 0.2, 0.68]))
    axins.plot(x, mrac.args.reference_model[:, 1], **basic_kwargs, ls='--')
    axins.plot(x, mrac.args.state[:, 1], **mrac.kwargs)
    axins.plot(fe.args.time, fe.args.state[:, 1], **fe.kwargs)
    axins.plot(be.args.time, be.args.state[:, 1], **be.kwargs)
    axins.set_xlim(0, 2.5)
    axins.set_ylim(-1, 0.35)
    plt.yticks(visible=False)
    mark_inset(ax2, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    plt.savefig('images/state_input.png', dpi=400)

    # ====================
    # Parameter estimation
    # ====================
    plt.figure(figsize=(4.51, 4.14))

    for w in mrac.args.W.ravel():
        plt.axhline(y=w, **basic_kwargs, ls='--', label='Real')

    plt.plot(x, mrac.args.what, **mrac.kwargs)
    plt.plot(fe.args.time, fe.args.what, **fe.kwargs)
    plt.plot(be.args.time, be.args.what, **be.kwargs)

    plt.ylabel(r'$W^\ast, W$')
    plt.xlabel('Time, sec')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='lower right')

    plt.tight_layout()

    plt.savefig('images/parameter_estimation.png', dpi=400)

    # ===========================
    # Parameter estimation (norm)
    # ===========================
    plt.figure(figsize=(8.02, 3.74))

    plt.plot(x, sla.norm(mrac.args.what - mrac.args.W.T, axis=1), **mrac.kwargs)
    plt.plot(fe.args.time, sla.norm(fe.args.what - fe.args.W.T, axis=1),
             **fe.kwargs)
    plt.plot(be.args.time, sla.norm(be.args.what - be.args.W.T, axis=1),
             **be.kwargs)

    plt.ylabel(r'$||\tilde{W}||$')
    plt.xlabel('Time, sec')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig('images/parameter_estimation_normed.png', dpi=400)

    # ===========
    # Eigenvalues
    # ===========
    fe.eigs = np.array(list(map(
        lambda A: eig_thr(A, args.thr)[0],
        fe.args.g_indirect * fe.args.best_basissum))).T
    be.eigs = np.array(list(map(
        lambda A: eig_thr(A, args.thr)[0], be.args.basissum))).T

    plt.figure(figsize=(7.03, 7.53))

    ax1 = plt.subplot(211)

    fe.kwargs_tmp = dict(fe.kwargs, lw=1.3, alpha=0.7)
    be.kwargs_tmp = dict(be.kwargs, lw=1.3, alpha=0.7)

    plt.fill_between(fe.args.time.ravel(), *fe.eigs[[-1, 0]],
                     facecolor='blue', alpha=0.3)
    plt.fill_between(be.args.time.ravel(), *be.eigs[[-1, 0]],
                     facecolor='red', alpha=0.3)
    plt.plot(fe.args.time.ravel(), fe.eigs[[-1, 0]].T, **fe.kwargs_tmp)
    plt.plot(fe.args.time.ravel(), fe.eigs[1:-1].T, **fe.kwargs,
             ls='--', alpha=0.5)
    plt.plot(be.args.time.ravel(), be.eigs[[-1, 0]].T, **be.kwargs_tmp)
    plt.plot(be.args.time.ravel(), be.eigs[1:-1].T, **be.kwargs,
             ls='--', alpha=0.5)

    ax2 = plt.subplot(212, sharex=ax1)
    plt.fill_between(fe.args.time.ravel(), *fe.eigs[[-1, 0]],
                     facecolor='blue', alpha=0.3)
    plt.fill_between(be.args.time.ravel(), *be.eigs[[-1, 0]],
                     facecolor='red', alpha=0.3)
    plt.plot(fe.args.time.ravel(), fe.eigs[[-1, 0]].T, **fe.kwargs_tmp)
    plt.plot(fe.args.time.ravel(), fe.eigs[1:-1].T, **fe.kwargs,
             ls='--', alpha=0.5)
    plt.plot(be.args.time.ravel(), be.eigs[[-1, 0]].T, **be.kwargs_tmp)
    plt.plot(be.args.time.ravel(), be.eigs[1:-1].T, **be.kwargs,
             ls='--', alpha=0.5)

    ax1.set_ylim([50, 3050])
    ax2.set_ylim([-0.5, 18])
    ax2.set_xlabel('Time, sec')

    handles, labels = ax1.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax1.legend(handles, labels, loc='center right')

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')
    ax2.xaxis.tick_bottom()

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, d), (-d, d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    plt.savefig('images/eigenvalues.png', dpi=400)

    # ===========
    # The a and b
    # ===========
    plt.figure(figsize=(11.57, 5.56))

    fe.kwargs, fe.kwargs_tmp = dict(fe.kwargs, lw=1.3), fe.kwargs
    be.kwargs, be.kwargs_tmp = dict(be.kwargs, lw=1.3), be.kwargs

    ax1 = plt.subplot(121)
    plt.plot(fe.args.time, fe.args.a, **fe.kwargs)
    plt.plot(be.args.time, be.args.a, **be.kwargs)
    plt.ylabel(r'$a$')
    plt.xlabel('Time, sec')
    plt.legend(loc='best')

    ax2 = plt.subplot(122)
    plt.plot(fe.args.time, fe.args.b, **fe.kwargs)
    plt.plot(be.args.time, be.args.b, **be.kwargs)
    plt.ylabel(r'$b$')
    plt.xlabel('Time, sec')
    plt.ylim(-10, 50)

    axins = zoomed_inset_axes(ax1, 1)
    axins.set_axes_locator(InsetPosition(ax1, [0.2, 0.6, 0.38, 0.2]))
    axins.plot(fe.args.time, fe.args.a, **fe.kwargs)
    axins.plot(be.args.time, be.args.a, **be.kwargs)
    axins.set_xlim(-0.001, 0.02)
    axins.set_ylim(-0.1, 5.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    axins2 = zoomed_inset_axes(ax1, 1)
    axins2.set_axes_locator(InsetPosition(ax1, [0.55, 0.3, 0.38, 0.2]))
    axins2.plot(fe.args.time, fe.args.a, **fe.kwargs)
    axins2.plot(be.args.time, be.args.a, **be.kwargs)
    axins2.set_xlim(20, 22)
    axins2.set_ylim(0.995, 1.005)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    mark_inset(ax1, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

    axins3 = zoomed_inset_axes(ax2, 1)
    axins3.set_axes_locator(InsetPosition(ax2, [0.5, 0.3, 0.45, 0.25]))
    axins3.plot(fe.args.time, fe.args.b, **fe.kwargs)
    axins3.plot(be.args.time, be.args.b, **be.kwargs)
    axins3.set_xlim(35, 40)
    axins3.set_ylim(-0.001, 0.001)
    plt.xticks(visible=False)
    plt.yticks(fontsize=8)
    mark_inset(ax2, axins3, loc1=3, loc2=4, fc="none", ec="0.5")

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, wspace=0.3)

    plt.savefig('images/a_and_b.png', dpi=400)


if __name__ == '__main__':
    args = Arguments()
    args.A = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
    args.B = np.array([[0, 1, 0]]).T
    args.Br = np.array([[0, 0, -1]]).T
    args.ndim_state = 3
    args.ndim_input = 1
    args.ndim_basis = 5
    args.xinit = np.array([0.3, 0, 0])
    args.Kr = np.array([[1]])
    args.Q_lyap = 1*np.eye(args.ndim_state)
    args.g_direct = 10000
    args.g_indirect = 0
    args.case = 'MRAC'
    args.t_step = 1e-3
    args.t_final = 40
    args.W = np.array([-18.59521, 15.162375, -62.45153,
                       9.54708, 21.45291])[:, np.newaxis]
    args.Lambda = np.diag([1])
    args.tau_f = 1e-3
    args.thr = 1e-8

    # MRAC
    if False:
        env = main(args)
        save(env.args, 'data/record_mrac.npz')
        print('MRAC - end')

    if False:
        env = load('data/record_mrac.npz')

    args.g_indirect = 1

    # Bounding Eigenvalue CMRAC
    if False:
        args.t_step = 5e-4
        args.tau_f = 1e-3
        args.lambda_max = 2.9e3
        args.lmax_speed = 2
        args.thr = 1e-8
        args.case = 'BECMRAC'
        env = main(args)
        # save(env.args, 'data/record_becmrac.npz')
        print('BEMRAC - end')

    if False:
        env = load('data/record_becmrac.npz')

    # Standard LS CMRAC
    if False:
        args.case = 'SLSCMRAC'
        env = main(args)
        save(env.args, 'data/record_slscmrac.npz')
        print('SLSMRAC - end')

    if False:
        env = load('data/record_slscmrac.npz')

    # Finite Excited CMRAC (N. Cho)
    if False:
        args.case = 'FECMRAC'
        args.t_step = 5e-4
        args.tau_f = 1e-3
        args.g_indirect = 1e3
        env = main(args)
        save(env.args, 'data/record_fecmrac.npz')
        print('FEMRAC - end')

    if False:
        env = load('data/record_fecmrac.npz')

    # Compare
    if False:
        compare()
