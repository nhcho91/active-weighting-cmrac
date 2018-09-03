import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, InsetPosition
import numpy as np
import scipy.linalg as sla

from utils import eig_thr


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


def compare(mrac, be, fe, is_save=False):
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

    if is_save:
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

    if is_save:
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

    if is_save:
        plt.savefig('images/parameter_estimation_normed.png', dpi=400)

    # ===========
    # Eigenvalues
    # ===========
    fe.eigs = np.array(list(map(
        lambda A: eig_thr(A, fe.args.thr)[0],
        fe.args.g_indirect * fe.args.best_basissum))).T
    be.eigs = np.array(list(map(
        lambda A: eig_thr(A, be.args.thr)[0], be.args.basissum))).T

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

    if is_save:
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
    axins.set_ylim(-0.1, 3.2)
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
    axins3.set_axes_locator(InsetPosition(ax2, [0.65, 0.3, 0.32, 0.20]))
    axins3.plot(fe.args.time, fe.args.b, **fe.kwargs)
    axins3.plot(be.args.time, be.args.b, **be.kwargs)
    axins3.set_xlim(20, 22)
    axins3.set_ylim(-0.5, 3.1)
    plt.xticks(visible=False)
    plt.yticks(fontsize=8)
    mark_inset(ax2, axins3, loc1=3, loc2=4, fc="none", ec="0.5")

    axins4 = zoomed_inset_axes(ax2, 1)

    axins4.set_axes_locator(InsetPosition(ax2, [0.4, 0.6, 0.55, 0.3]))
    axins4.plot(fe.args.time, fe.args.b, **fe.kwargs)
    axins4.plot(be.args.time, be.args.b, **be.kwargs)
    # axins4.set_xlim(0, 10)
    # axins4.set_ylim(-1e10, 1e12)

    # plt.xticks(visible=False)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # mark_inset(ax2, axins4, loc1=1, loc2=4, fc="none", ec="0.5")

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, wspace=0.3)

    if is_save:
        plt.savefig('images/a_and_b.png', dpi=400)

    plt.show()
