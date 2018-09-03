import numpy as np
import scipy.linalg as sla
import scipy.signal

from collections import deque, UserDict

from tqdm import tqdm

from utils import Arguments, Record, eig_thr
from plots import compare


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
        args.case = 'BECMRAC'
        args.t_step = 5e-4
        args.tau_f = 1e-3
        args.lambda_max = 2.9e3
        args.lmax_speed = 0.15
        args.thr = 1e-8
        env = main(args)
        save(env.args, 'data/record_becmrac.npz')
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
        mrac = load('data/record_mrac.npz')
        be = load('data/record_becmrac.npz')
        fe = load('data/record_fecmrac.npz')
        compare(mrac=mrac, be=be, fe=fe)
