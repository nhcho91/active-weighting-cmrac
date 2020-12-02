import numpy as np
import scipy.signal
import scipy.optimize
from types import SimpleNamespace as SN
from pathlib import Path
import itertools

from fym.core import BaseEnv, BaseSystem
import fym.logging

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

# np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)

cfg = SN()

style = SN()
style.base = dict(c="k", lw=0.7)


def load_config():
    cfg.Am = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
    cfg.B = np.array([[0, 1, 0]]).T
    cfg.Br = np.array([[0, 0, -1]]).T
    cfg.x_init = np.vstack((0.3, 0, 0))
    cfg.Q_lyap = np.eye(3)
    cfg.P = scipy.linalg.solve_lyapunov(cfg.Am.T, -cfg.Q_lyap)
    cfg.final_time = 50

    cfg.Wcirc = np.vstack((-18.59521, 15.162375, -62.45153, 9.54708, 21.45291))

    cfg.vareps = SN()
    cfg.vareps.freq = 5
    cfg.vareps.amp = 2  # 2
    cfg.vareps.offset = 0

    cfg.tauf = 1e-3
    cfg.Gamma1 = 1e4
    cfg.threshold = 1e-10

    cfg.bF = 5000
    cfg.bh = 1500
    cfg.LF_speed = 0.05
    cfg.Lh_speed = 0.05
    cfg.LF_init = 10
    cfg.Lh_init = 10

    cfg.dir = "data"

    # MRAC
    cfg.MRAC = SN()
    cfg.MRAC.env_kwargs = dict(
        solver="odeint", dt=20, max_t=cfg.final_time, ode_step_len=int(20/0.01))

    # BECMRAC
    cfg.BECMRAC = SN()
    cfg.BECMRAC.env_kwargs = dict(
        solver="rk4", dt=0.01, max_t=cfg.final_time, ode_step_len=10)
    cfg.BECMRAC.Gamma2 = 1

    # FECMRAC
    cfg.FECMRAC = SN()
    cfg.FECMRAC.env_kwargs = dict(
        solver="rk4", dt=1e-2, max_t=cfg.final_time, ode_step_len=10)
    cfg.FECMRAC.Gamma2 = 500
    cfg.FECMRAC.kL = 0.1
    cfg.FECMRAC.kU = 10
    cfg.FECMRAC.theta = 0.1


def get_eig(A, threshold=0):
    eigs, eigv = np.linalg.eig(A)
    sort = np.argsort(np.real(eigs))  # sort in ascending order
    eigs = np.real(eigs[sort])
    eigv = np.real(eigv[:, sort])
    eigs[eigs < threshold] = 0
    return eigs, eigv


class System(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)
        self.unc = Uncertainty()

    def set_dot(self, t, u, c):
        x = self.state
        self.dot = (
            cfg.Am.dot(x)
            + cfg.B.dot(u + self.unc(t, x))
            + cfg.Br.dot(c)
        )


class ReferenceSystem(BaseSystem):
    def __init__(self):
        super().__init__(cfg.x_init)
        self.cmd = Command()

    def set_dot(self, c):
        xr = self.state
        self.dot = cfg.Am.dot(xr) + cfg.Br.dot(c)


class Uncertainty():
    def __init__(self):
        self.Wcirc = cfg.Wcirc

    def basis(self, x):
        return np.vstack((x[:2], np.abs(x[:2])*x[1], x[0]**3))

    def parameter(self, t):
        return self.Wcirc

    def vareps(self, t, x):
        # vareps = cfg.vareps.amp * np.sin(cfg.vareps.freq * t) + cfg.vareps.offset
        # vareps = np.tanh(np.sum(np.abs(x[:2]) * x[0]) + x[1]**3)
        vareps = np.tanh(x[1])
        vareps += np.exp(-t/10) * np.sin(5 * t)
        return cfg.vareps.amp * vareps

    def __call__(self, t, x):
        Wcirc = self.parameter(t)
        return Wcirc.T.dot(self.basis(x)) + self.vareps(t, x)


class Command():
    def __call__(self, t):
        if t > 10:
            c = scipy.signal.square([(t - 10) * 2*np.pi / 20])
        else:
            c = 0
        return np.atleast_2d(c)


class CMRAC(BaseSystem):
    def __init__(self):
        super().__init__(shape=cfg.Wcirc.shape)

    def set_dot(self, e, phi, composite_term):
        self.dot = (
            cfg.Gamma1 * phi.dot(e.T).dot(cfg.P).dot(cfg.B)
            + composite_term
        )


class MRACEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.MRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()
        self.W = CMRAC()

        self.basis = self.x.unc.basis
        self.cmd = Command()

        self.logger = fym.logging.Logger(Path(cfg.dir, "mrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def step(self):
        *_, done = self.update()

        return done

    def set_dot(self, t):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        phi = self.basis(x)
        c = self.cmd(t)

        e = x - xr
        u = - W.T.dot(phi)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, 0)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W = self.observe_list(y)
        phi = self.basis(x)
        c = self.cmd(t)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = - W.T.dot(phi)

        return dict(t=t, x=x, xr=xr, W=W, Wcirc=Wcirc, e=e, c=c, u=u)


class Filter(BaseEnv):
    def __init__(self, basis):
        super().__init__()
        self.xi = BaseSystem(shape=(5, 1))
        self.z = BaseSystem()

        self.Bdagger = np.linalg.pinv(cfg.B)

    def set_dot(self, e, phi, u):
        xi = self.xi.state
        z = self.z.state

        self.xi.dot = -1 / cfg.tauf * (xi - phi)
        self.z.dot = (
            1 / cfg.tauf * (self.Bdagger.dot(e) - z)
            + self.Bdagger.dot(cfg.Am).dot(e)
            + u
        )

    def get_y(self, e, z):
        return 1 / cfg.tauf * (self.Bdagger.dot(e) - z)


class BECMRACAgent():
    def __init__(self):
        self.F = np.zeros((5, 5))
        self.G = np.zeros((5, 1))
        self.h = np.zeros((1, 1))

        self.logger = fym.logging.Logger(Path(cfg.dir, "becmrac-agent.h5"))

    def close(self):
        self.logger.close()

    def get_L(self, t, ln, h):
        LF = (cfg.bF - cfg.LF_init) * np.tanh(cfg.LF_speed * t) + cfg.LF_init
        Lh = (cfg.bh - cfg.Lh_init) * np.tanh(cfg.Lh_speed * t) + cfg.Lh_init
        # LF = (cfg.bF - ln) * np.tanh(cfg.LF_speed * t) + ln
        # Lh = (cfg.bh - h) * np.tanh(cfg.Lh_speed * t) + h
        return LF, Lh

    def get_action(self, obs):
        self.update(obs)

        F, G, h = self.F, self.G, self.h

        t, *_ = obs
        eigs, _ = get_eig(F, cfg.threshold)
        self.logger.record(t=t, eigs=eigs, h=h)

        return F, G, h

    def update(self, obs):
        t, y, xi = obs

        p, q = self._get_pq(obs)

        self.F = q * self.F + p * xi.dot(xi.T)
        self.G = q * self.G + p * xi.dot(y.T)
        self.h = q * self.h + np.abs(p) * np.linalg.norm(xi)

    def _get_pq(self, obs):
        t, y, xi = obs

        eigs, eigv = get_eig(self.F)
        ln = eigs[-1]

        h = self.h[0][0]
        LF, Lh = self.get_L(t, ln, h)

        v = eigv.T.dot(xi)
        v1, vn = v[[0, -1]].ravel()
        nv = np.linalg.norm(xi)

        p, q = 0, 1

        if ln == 0:
            if nv > cfg.threshold:
                p, q = LF / nv**2, 1
            else:
                p, q = 0, 1
            return p, q

        if nv < cfg.threshold:
            p, q = 1, LF / ln
            return p, q

        if np.any(eigs == 0):
            reduced_indices = (
                [np.argmax(v[eigs == 0]**2)]
                + np.where(eigs != 0)[0].tolist())
        else:
            reduced_indices = eigs != 0

        eigs = eigs[reduced_indices]
        eigv = eigv[:, reduced_indices]

        l1, ln = eigs[[0, -1]]
        g1, gn = np.diff(eigs)[[0, -1]]

        if np.abs(gn - ln) < cfg.threshold:
            p, q = LF / 2 / nv**2, LF / 2 / ln
            return p, q

        p0 = - Lh / (nv + 1e-10)
        p1 = Lh / (nv + 1e-10)

        args = l1, ln, g1, gn, v1, vn, nv, LF, Lh, h
        psol = scipy.optimize.minimize_scalar(
            self._minus_f,
            method="bounded",
            bounds=(p0, p1),
            args=args,
        )

        if psol.fun <= -l1:
            p = psol.x
            q = self._find_q(p, *args)

        return p, q

    def _find_q(self, p, *args):
        l1, ln, g1, gn, v1, vn, nv, LF, Lh, h = args

        tmp = LF * (2 * ln - gn) - (ln * nv**2 - gn * vn**2) * p

        qF = 1 / (2 * ln * (ln - gn)) * (
            tmp - np.sqrt(tmp**2 - 4 * ln * (ln - gn) * LF * (LF - nv**2 * p)))
        qh = (Lh - np.abs(p) * nv) / h

        return min(qF, qh)

    def _minus_f(self, p, *args):
        l1, ln, g1, gn, v1, vn, nv, LF, Lh, h = args

        q = self._find_q(p, *args)
        f = (
            q * (l1 + 0.5 * g1)
            + 0.5 * p * nv**2
            - 0.5 * np.sqrt((q * g1 + p * nv**2)**2 - 4 * q * p * g1 * v1**2)
        )

        return -f


class BECMRACEnv(BaseEnv):
    def __init__(self):
        super().__init__(**cfg.BECMRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()
        self.W = CMRAC()
        self.filter = Filter(basis=self.x.unc.basis)

        self.basis = self.x.unc.basis
        self.cmd = Command()

        self.logger = fym.logging.Logger(Path(cfg.dir, "becmrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def reset(self):
        super().reset()
        return self.observation()

    def observation(self):
        x, xr, W, (xi, z) = self.observe_list()
        e = x - xr
        y = self.filter.get_y(e, z)
        t = self.clock.get()
        return t, y, xi

    def step(self, action):
        F, G, _ = action
        *_, done = self.update(F=F, G=G)
        return self.observation(), done

    def set_dot(self, t, F, G):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        phi = self.basis(x)
        c = self.cmd(t)

        e = x - xr
        u = - W.T.dot(phi)

        composite_term = - cfg.BECMRAC.Gamma2 * (F.dot(W) - G)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, composite_term)
        self.filter.set_dot(e, phi, u)

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        x, xr, W, (xi, z) = self.observe_list(y)
        phi = self.basis(x)
        c = self.cmd(t)

        Wcirc = self.x.unc.parameter(t)

        e = x - xr
        u = - W.T.dot(phi)

        return dict(t=t, x=x, xr=xr, W=W, xi=xi, z=z,
                    Wcirc=Wcirc, e=e, c=c, u=u)


class FECMRACAgent(BECMRACAgent):
    def __init__(self):
        self.F = np.zeros((5, 5))
        self.G = np.zeros((5, 1))
        self.h = np.zeros((1, 1))

        self.best_F = self.F.copy()
        self.best_G = self.G.copy()
        self.best_h = self.h.copy()

        self.logger = fym.logging.Logger(Path(cfg.dir, "fecmrac-agent.h5"))

    def get_action(self, obs):
        self.update(obs)

        F, G, h = self.best_F, self.best_G, self.best_h

        t, *_ = obs
        eigs, _ = get_eig(F, cfg.threshold)
        self.logger.record(t=t,
                           eigs=cfg.FECMRAC.Gamma2 * eigs,
                           h=cfg.FECMRAC.Gamma2 * h)

        return F, G, h

    def update(self, obs):
        t, y, xi, phi = obs

        p, q = self._get_pq(obs)

        self.F = q * self.F + p * xi.dot(xi.T)
        self.G = q * self.G + p * xi.dot(y.T)
        self.h = q * self.h + np.abs(p) * np.linalg.norm(xi)

        if get_eig(self.F)[0][0] > get_eig(self.best_F)[0][0]:
            self.best_F = self.F
            self.best_G = self.G
            self.best_h = self.h

    def _get_pq(self, obs):
        t, y, xi, phi = obs

        p = cfg.FECMRAC.env_kwargs["dt"]
        xidot = - (xi - phi) / cfg.tauf
        nxidot = np.linalg.norm(xidot)
        k = self._get_k(nxidot)
        q = 1 - k * p

        return p, q

    def _get_k(self, nxidot):
        return (cfg.FECMRAC.kL
                + ((cfg.FECMRAC.kU - cfg.FECMRAC.kL)
                   * np.tanh(cfg.FECMRAC.theta * nxidot)))


class FECMRACEnv(BECMRACEnv, BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self, **cfg.FECMRAC.env_kwargs)
        self.x = System()
        self.xr = ReferenceSystem()
        self.W = CMRAC()
        self.filter = Filter(basis=self.x.unc.basis)

        self.basis = self.x.unc.basis
        self.cmd = Command()

        self.logger = fym.logging.Logger(Path(cfg.dir, "fecmrac-env.h5"))
        self.logger.set_info(cfg=cfg)

    def observation(self):
        x, xr, W, (xi, z) = self.observe_list()
        e = x - xr
        y = self.filter.get_y(e, z)
        t = self.clock.get()
        phi = self.basis(x)
        return t, y, xi, phi

    def set_dot(self, t, F, G):
        x = self.x.state
        xr = self.xr.state
        W = self.W.state
        phi = self.basis(x)
        c = self.cmd(t)

        e = x - xr
        u = - W.T.dot(phi)

        composite_term = - cfg.FECMRAC.Gamma2 * (F.dot(W) - G)

        self.x.set_dot(t, u, c)
        self.xr.set_dot(c)
        self.W.set_dot(e, phi, composite_term)
        self.filter.set_dot(e, phi, u)


def run_mrac():
    env = MRACEnv()
    env.reset()

    while True:
        env.render()

        done = env.step()

        if done:
            break

    env.close()


def run_becmrac():
    env = BECMRACEnv()
    agent = BECMRACAgent()

    obs = env.reset()

    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, done = env.step(action)

        if done:
            break

        obs = next_obs

    env.close()
    agent.close()


def run_fecmrac():
    env = FECMRACEnv()
    agent = FECMRACAgent()

    obs = env.reset()

    while True:
        env.render()

        action = agent.get_action(obs)
        next_obs, done = env.step(action)

        if done:
            break

        obs = next_obs

    env.close()
    agent.close()


def plot_compare():
    # Style setting
    style.ref = dict(style.base, ls="--")
    style.mrac = dict(style.base, c="b", ls="--", label="MRAC")
    style.becmrac = dict(style.base, c="r", label="Proposed")
    style.fecmrac = dict(style.base, c="b", label="FE-CMRAC")

    # Data loading
    mrac = fym.logging.load(cfg.MRAC.env_path)
    becmrac = fym.logging.load(cfg.BECMRAC.env_path)
    becmrac_agent = fym.logging.load(cfg.BECMRAC.agent_path)
    fecmrac = fym.logging.load(cfg.FECMRAC.env_path)
    fecmrac_agent = fym.logging.load(cfg.FECMRAC.agent_path)

    # =================
    # States and inputs
    # =================
    plt.figure(figsize=(12.31, 7.57))

    ax = plt.subplot(311)
    plt.plot(mrac["t"], mrac["xr"][:, 0], **style.ref)
    plt.plot(mrac["t"], mrac["x"][:, 0], **style.mrac)
    plt.plot(becmrac["t"], becmrac["x"][:, 0], **style.becmrac)
    plt.plot(fecmrac["t"], fecmrac["x"][:, 0], **style.fecmrac)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel(r"$x_1$")
    plt.legend(loc="best")

    plt.subplot(312, sharex=ax)
    plt.plot(mrac["t"], mrac["xr"][:, 1], **style.ref)
    plt.plot(mrac["t"], mrac["x"][:, 1], **style.mrac)
    plt.plot(becmrac["t"], becmrac["x"][:, 1], **style.becmrac)
    plt.plot(fecmrac["t"], fecmrac["x"][:, 1], **style.fecmrac)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel(r"$x_2$")

    plt.subplot(313, sharex=ax)
    plt.plot(mrac["t"], mrac["u"][:, 0], **style.mrac)
    plt.plot(becmrac["t"], becmrac["u"][:, 0], **style.becmrac)
    plt.plot(fecmrac["t"], fecmrac["u"][:, 0], **style.fecmrac)
    plt.ylabel(r'$u$')
    plt.xlabel("Time, sec")

    # =================================
    # Parameter estimation error (norm)
    # =================================
    plt.figure(figsize=(8.02, 3.74))

    plt.plot(mrac["t"],
             np.linalg.norm((mrac["W"] - cfg.Wcirc).squeeze(), axis=1),
             **style.mrac)
    plt.plot(becmrac["t"],
             np.linalg.norm((becmrac["W"] - cfg.Wcirc).squeeze(), axis=1),
             **style.becmrac)
    plt.plot(fecmrac["t"],
             np.linalg.norm((fecmrac["W"] - cfg.Wcirc).squeeze(), axis=1),
             **style.fecmrac)

    plt.ylabel(r"$||\tilde{W}||$")
    plt.xlabel('Time, sec')
    plt.legend(loc='best')
    plt.tight_layout()

    # ===========
    # Eigenvalues
    # ===========
    plt.figure(figsize=(8.75, 6.1))

    ax1 = plt.subplot(211)
    plot_eigvals(becmrac_agent, style.becmrac)
    plot_eigvals(fecmrac_agent, style.fecmrac)

    ax2 = plt.subplot(212, sharex=ax1)
    plot_eigvals(becmrac_agent, style.becmrac)
    plot_eigvals(fecmrac_agent, style.fecmrac)

    ax1.set_ylim([10, 800])
    ax2.set_ylim([-0.5, 1.5])
    ax2.set_xlabel('Time, sec')

    plt.show()


def plot_eigvals(data, internal=True):
    plt.fill_between(data.agent["t"],
                     data.agent["eigs"][:, -1],
                     data.agent["eigs"][:, 0],
                     # **fill_style)
                     facecolor=data.style["c"],
                     alpha=0.3)
    plt.plot(data.agent["t"], data.agent["eigs"][:, [0, -1]],
             **dict(data.style, alpha=0.7))
    if internal:
        plt.plot(data.agent["t"], data.agent["eigs"][:, 1:-1],
                 **dict(data.style, label="_", ls="--", alpha=0.5))


def plot_tracking_error(data):
    plt.plot(data.env["t"],
             np.linalg.norm(data.env["e"].squeeze(), axis=1),
             **data.style)


def plot_estimation_error(data):
    plt.plot(data.env["t"],
             np.linalg.norm(
                 (data.env["W"] - data.env["Wcirc"]).squeeze(), axis=1),
             **data.style)


def plot_h(data):
    plt.plot(data.agent["t"], data.agent["h"].squeeze(),
             **dict(data.style))


def plot_parameters(data):
    lines = plt.plot(data.env["t"], data.env["W"].squeeze(), **data.style)
    plt.setp(lines[1:], label=None)


def plot_states_and_input(data, key, index):
    return plt.plot(data.env["t"], data.env[key][:, index], **data.style)


def plot_posing(n, subsize, width, top, bottom, left, hspace):
    refpoint = (bottom, left)
    figsize = (width, refpoint[1] + subsize[1] * n + hspace * (n - 1) + top)
    sub = np.divide(subsize, figsize)
    ref = np.divide(refpoint, figsize)

    h = hspace / figsize[1]
    poses = []
    for i in range(n):
        subref = ref + np.array([0, (h + sub[1]) * (n - 1 - i)])
        pos = np.vstack((subref, sub))
        poses.append(pos.ravel())

    return figsize, poses


def exp1():
    """Changing b_h"""
    basedir = Path("data/exp1")

    bh_list = [1e0, 1e1, 1e2, 1e3]
    bF_list = [1e0, 1e1, 1e2, 1e3]

    def run(exp_cfg):
        load_config()
        cfg.dir = exp_cfg.dir
        cfg.bh = exp_cfg.bh
        cfg.bF = exp_cfg.bF
        cfg.label = exp_cfg.label
        run_becmrac()

    for i, (bh, bF) in enumerate(itertools.product(bh_list, bF_list)):
        data = SN()
        data.dir = Path(basedir, f"data{i:02d}")
        data.bh = bh
        data.bF = bF
        data.label = rf"$b_h = {bh}$, $b_F = {bF}$"
        run(data)


def exp1_plot():
    def get_data(datadir):
        data = SN()
        env, info = fym.logging.load(Path(datadir, "becmrac-env.h5"),
                                     with_info=True)
        data.env = env
        data.info = info
        data.agent = fym.logging.load(Path(datadir, "becmrac-agent.h5"))
        data.style = dict(label=rf"$b_h = {info['cfg'].bh}$, $b_F = {info['cfg'].bF}$")
        return data

    data = []
    for p in sorted(Path("data", "exp1").iterdir()):
        d = get_data(p)
        data.append(d)

    basestyle = dict(c="k", lw=0.7)
    for d, (c, ls) in zip(data, itertools.product("rgbk", ["-", "--", "-.", ":"])):
        d.style.update(basestyle, c=c, ls=ls)

    # =================================
    # Parameter estimation error (norm)
    # =================================
    plt.figure(figsize=(8.02, 3.74))

    for d in data:
        plot_estimation_error(d)

    plt.ylabel(r"$||\tilde{W}||$")
    plt.xlabel('Time, sec')
    plt.legend(loc='best')
    plt.tight_layout()

    # ===========
    # Eigenvalues
    # ===========
    plt.figure(figsize=(8.75, 6.1))

    ax1 = plt.subplot(211)
    # for d in data:
    #     plot_eigvals(d)

    ax2 = plt.subplot(212, sharex=ax1)
    for d in data:
        plot_eigvals(d)

    ax1.set_ylim([10, 800])
    ax2.set_ylim([-0.5, 1.5])
    ax2.set_xlabel('Time, sec')
    # ax1.legend(loc='best')

    plt.show()


def exp2():
    """Changing b_h"""
    basedir = Path("data/exp2")

    Gamma2_list = [1e0, 1e1, 1e2, 1e3]

    for i, Gamma2 in enumerate(Gamma2_list):
        load_config()
        cfg.dir = Path(basedir, f"data{i:02d}")
        cfg.FECMRAC.Gamma2 = Gamma2
        cfg.label = rf"FE-CMRAC ($\Gamma_2 = {Gamma2}$)"
        run_fecmrac()

    i += 1
    load_config()
    cfg.dir = Path(basedir, f"data{i:02d}")
    cfg.label = "Proposed"
    run_becmrac()


def exp2_plot():
    def get_data(datadir):
        data = SN()
        env, info = fym.logging.load(list(datadir.glob("*env.h5"))[0],
                                     with_info=True)
        data.env = env
        data.info = info
        data.agent = fym.logging.load(list(datadir.glob("*agent.h5"))[0])
        data.style = dict(label=info["cfg"].label)
        return data

    data = []
    for p in sorted(Path("data", "exp2").iterdir()):
        d = get_data(p)
        data.append(d)

    basestyle = dict(c="k", lw=0.7)
    for d, (c, ls) in zip(data, itertools.product("rgbk", ["-", "--", "-.", ":"])):
        d.style.update(basestyle, c=c, ls=ls)

    # =================================
    # Parameter estimation error (norm)
    # =================================
    plt.figure(figsize=(8.02, 3.74))

    for d in data:
        plot_estimation_error(d)

    plt.ylabel(r"$||\tilde{W}||$")
    plt.xlabel('Time, sec')
    plt.legend(loc='best')
    plt.tight_layout()

    # ===========
    # Eigenvalues
    # ===========
    plt.figure(figsize=(8.75, 6.1))

    ax1 = plt.subplot(211)
    # for d in data:
    #     plot_eigvals(d)

    ax2 = plt.subplot(212, sharex=ax1)
    for d in data:
        plot_eigvals(d)

    ax1.set_ylim([10, 800])
    ax2.set_ylim([-0.5, 1.5])
    ax2.set_xlabel('Time, sec')
    # ax1.legend(loc='best')

    plt.show()


def exp3():
    """Chaning vareps"""
    basedir = Path("data/exp3")

    amp_list = [1e-1, 1e0, 1e1]
    freq_list = [1e0, 1e1, 1e2]

    for i, (amp, freq) in enumerate(itertools.product(amp_list, freq_list)):
        load_config()
        cfg.dir = Path(basedir, f"data{i:02d}")
        cfg.vareps.amp = amp
        cfg.vareps.freq = freq

        cfg.label = rf"FE-CMRAC (amp: {amp}, freq: {freq})"
        run_fecmrac()

        cfg.label = rf"Proposed (amp: {amp}, freq: {freq})"
        run_becmrac()


def exp3_plot():
    def get_data(datadir, cnt):
        data = SN()
        env, info = fym.logging.load(Path(datadir, cnt + "-env.h5"),
                                     with_info=True)
        data.env = env
        data.info = info
        data.agent = fym.logging.load(Path(datadir, cnt + "-agent.h5"))
        data.style = dict(label=info["cfg"].label)
        return data

    data = []
    for datadir in sorted(Path("data", "exp3").iterdir()):
        d = (get_data(datadir, "fecmrac"),
             get_data(datadir, "becmrac"))
        data.append(d)

    basestyle = dict(lw=0.7)
    colors = {}
    colors.update(matplotlib.colors.BASE_COLORS)
    colors.update(matplotlib.colors.TABLEAU_COLORS)
    for (fecmrac, becmrac), c in zip(data, colors.values()):
        fecmrac.style.update(basestyle, c=c, ls="--")
        becmrac.style.update(basestyle, c=c, ls="-")

    # =================================
    # Parameter estimation error (norm)
    # =================================
    plt.figure(figsize=(8.02, 3.74))

    for d in data:
        plot_estimation_error(d[0])
        plot_estimation_error(d[1])

    plt.ylabel(r"$||\tilde{W}||$")
    plt.xlabel('Time, sec')
    plt.legend(loc='best')
    plt.tight_layout()

    # ===========
    # Eigenvalues
    # ===========
    plt.figure(figsize=(8.75, 6.1))

    ax1 = plt.subplot(211)
    # for d in data:
    #     plot_eigvals(d)

    ax2 = plt.subplot(212, sharex=ax1)
    for d in data:
        plot_eigvals(d[0])
        plot_eigvals(d[1])

    ax1.set_ylim([10, 800])
    ax2.set_ylim([-0.5, 1.5])
    ax2.set_xlabel('Time, sec')
    # ax1.legend(loc='best')

    plt.show()


def exp4():
    """Compare"""
    basedir = Path("data/exp4")

    load_config()
    cfg.dir = Path(basedir, "data00")
    cfg.label = "MRAC"
    run_mrac()

    load_config()
    cfg.dir = Path(basedir, "data01")
    cfg.label = r"FE-CMRAC"
    run_fecmrac()

    load_config()
    cfg.dir = Path(basedir, "data03")
    cfg.label = "Proposed"
    run_becmrac()


def exp4_plot():
    def get_data(datadir):
        data = SN()
        env, info = fym.logging.load(list(datadir.glob("*env.h5"))[0],
                                     with_info=True)
        data.env = env
        data.info = info
        agentlist = list(datadir.glob("*agent.h5"))
        if agentlist != []:
            data.agent = fym.logging.load(agentlist[0])
        data.style = dict(label=info["cfg"].label)
        return data

    plt.rc("font", family="Times New Roman")
    plt.rc("text", usetex=True)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", grid=True)
    plt.rc("grid", linestyle="--", alpha=0.8)

    datadir = Path("data", "exp4")
    mrac = get_data(Path(datadir, "data00"))
    fecmrac = get_data(Path(datadir, "data01"))
    becmrac = get_data(Path(datadir, "data03"))
    data = [mrac, fecmrac, becmrac]
    data_no_mrac = [fecmrac, becmrac]

    basestyle = dict(c="k", lw=0.7)
    cmdstyle = dict(basestyle, ls="--", label="Command")
    mrac.style.update(basestyle, c="g", ls="-")
    fecmrac.style.update(basestyle, c="b", ls="-.")
    becmrac.style.update(basestyle, c="r", ls="-")

    # Figure common setup
    t_range = (0, 50)

    # All in inches
    subsize = (4.05, 0.946)
    width = 4.94
    top = 0.2
    bottom = 0.671765
    left = 0.5487688
    hspace = 0.2716

    # =================
    # States and inputs
    # =================
    figsize, pos = plot_posing(3, subsize, width, top, bottom, left, hspace)

    plt.figure(figsize=figsize)

    ax = plt.subplot(311, position=pos[0])
    lines = []
    lines += plt.plot(mrac.env["t"], mrac.env["c"][:, 0], **cmdstyle)
    lines += [plot_states_and_input(d, "x", 0)[0] for d in data]
    plt.ylabel(r"$x_1$")
    plt.ylim(-2, 2)
    plt.figlegend(
        lines,
        [line.get_label() for line in lines],
        bbox_to_anchor=(0.99, 0.78)
    )

    plt.subplot(312, sharex=ax, position=pos[1])
    [plot_states_and_input(d, "x", 1) for d in data]
    plt.ylabel(r"$x_2$")
    plt.ylim(-2, 2)

    plt.subplot(313, sharex=ax, position=pos[2])
    [plot_states_and_input(d, "u", 0) for d in data]
    plt.ylabel(r'$u$')
    plt.xlabel("Time, sec")
    plt.xlim(t_range)
    plt.ylim(-80, 80)

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    # ==============================================
    # Tracking and parameter estimation errors (norm)
    # ==============================================
    figsize, pos = plot_posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax = plt.subplot(211, position=pos[0])
    [plot_tracking_error(d) for d in data]
    plt.ylabel(r"$||e||$")
    plt.ylim(0, 0.6)
    plt.legend(loc='best')

    plt.subplot(212, sharex=ax, position=pos[1])
    [plot_estimation_error(d) for d in data]
    plt.ylabel(r"$||\tilde{W}||$")
    plt.xlabel("Time, sec")
    plt.xlim(t_range)
    plt.ylim(0, 85)

    # ===========
    # Eigenvalues
    # ===========
    figsize, pos = plot_posing(2, subsize, width, top, bottom, left, hspace)
    plt.figure(figsize=figsize)

    ax1 = plt.subplot(211, position=pos[0])
    [plot_eigvals(d, internal=True) for d in data_no_mrac]

    plt.ylabel("Eigenvalues")
    plt.yscale("log")
    plt.ylim([1e-5, 3000])

    plt.subplot(212, sharex=ax1, position=pos[1])
    [plot_h(d) for d in data_no_mrac]
    plt.ylabel(r"$h$")
    plt.xlabel("Time, sec")
    plt.xlim(t_range)
    plt.legend(loc="best")

    for ax in plt.gcf().get_axes():
        ax.label_outer()

    basedir = Path("img")
    basedir.mkdir(exist_ok=True)

    plt.figure(1)
    plt.savefig(Path(basedir, "figure_1.pdf"), bbox_inches="tight")

    plt.figure(2)
    plt.savefig(Path(basedir, "figure_2.pdf"), bbox_inches="tight")

    plt.figure(3)
    plt.savefig(Path(basedir, "figure_3.pdf"), bbox_inches="tight")

    plt.show()


def main():
    # run_mrac()
    # run_becmrac()
    # run_fecmrac()
    # plot_compare()

    # exp1()
    # exp1_plot()

    # exp2()
    # exp2_plot()

    # exp3()
    # exp3_plot()

    # exp4()
    exp4_plot()


if __name__ == "__main__":
    main()
