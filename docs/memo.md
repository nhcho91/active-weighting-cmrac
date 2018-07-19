---
title: Composite MRAC
author: Seong-hun Kim
documentclass: my-handout
bibliography: memo.bib --filter pandoc-citeproc
header-includes: |
...

# Purpose

* To suggest a data-efficient adaptive control algorithm with stability proof.
* To consider time-varying stochastic uncertainties like wind velocity.
* Relieve the persistent / interval excitation condition by exploiting the
	non-single-step algorithms.

# Related Works

* Yongping Pan [@pan_composite_2016], [@pan_composite_2018]
	- Composite learning control

* Girish Chowdhary [@kamalapurkar_concurrent_2017],
[@chowdhary_concurrent_2013-1]
	- Concurrent learning

# Parameter Estimation

Consider the system
$$
\dot{x}(t) = A x(t) + B_r c(t) + B \Lambda \qty(u(t) + W_0^T \phi_0(t, x)),
$$
and the reference system
$$
\dot{x}_r(t) = A x_r(t) + B_r c(t),
$$
with the system matrix $A$ being Hurwitz. Then, we have the following error
dynamics
$$
\dot{e} = A e(t) + B \Lambda \qty(u(t) + W^T \phi(t, x, c)).
$$
The unknowns are a diagonal matrix $\Lambda$ possessing positive elements, and
a parameter matrix $W$.

<!-- $$ -->
<!-- \dot{e}(t) = A_r e(t) + B \Lambda u_a + B \Lambda W^T \phi(t, x) - B \Lambda (I - -->
<!-- \Lambda^{-1}) K x(t) + B \Lambda (I - \Lambda^{-1}) K_r c(t). -->
<!-- $$ -->

<!-- $$ -->
<!-- \dot{x}(t) = A x(t) + B_r c(t) + B \Lambda \qty(K_r c(t) - \Lambda^{-1} K_r -->
<!-- c(t) + \hat{\Lambda}^{-1} K_r c(t) - \hat{W}^T \phi(t, x) + W^T \phi(t, x)) -->
<!-- $$ -->

Integrating both side of the equation yields
$$
e(t) - e(t-\delta) = A \int_{t-\delta}^t e(\tau) \dd{\tau} - B \Lambda v_1(t) +
B \Lambda W^T v_2(t)
$$
with
$$
\begin{aligned}
	v_1(t) &= \int_{t-\delta}^t u(\tau) \dd{\tau}, \\
	v_2(t) &= \int_{t-\delta}^t \phi(\cdot) \dd{\tau}, \\
\end{aligned}
$$
with $\delta > 0$, and let
$$
y(t) = e(t) - e(t - \delta) - A \int_{t - \delta}^{t} e(\tau) \dd{\tau}.
$$
Then, for a sequence $\{t_n\}$ such that $t_n > \delta$, a dataset $\MC{D}$ of
which the input is $(v_1(t_k), v_2(t_k))$ and the output is $y(t_k)$ for
$k=0,1,\dotsc$ is generated.

We can define a regressor for $\MC{D}$ as follows
$$
\begin{aligned}
	\hat{y}(t) &= - B \hat{\Lambda}(t) v_1(t) + B \hat{V}(t) v_2(t),\\
	&= \bmqty{ - v_1^T \otimes B & v_2^T \otimes B}
	\bmqty{\mathrm{vec}(\hat{\Lambda}) \\ \mathrm{vec}(\hat{V})}
\end{aligned},
$$
where $\hat{V} = \hat{\Lambda} \hat{W}^T$. To formulate the problem as a
stochastic gradient problem, we define an error as
$$
\epsilon(t) = \norm{\hat{y}(t) - y(t)}^2.
$$
Then, we have
$$
\dot{\hat{\Lambda}} = \gamma_1 \mathrm{diag}\qty(v_1) \mathrm{diag}\qty(B^T
(\hat{y} - y)),
$$
and
$$
\begin{aligned}
	\dot{\hat{V}} &=  - \gamma_2 \qty(v_2 \otimes B^T) (\hat{y} - y) \\
	&= - \gamma_2 B^T (\hat{y} - y) v_2^T.
\end{aligned}
$$
Since $\dot{\hat{V}} = \dot{\hat{\Lambda}} \hat{W}^T + \hat{\Lambda}
\dot{\hat{W}}^T$,
$$
\dot{\hat{W}} = - \gamma_1 \hat{W} \mathrm{diag}\qty(v_1) \mathrm{diag}\qty(B^T
(\hat{y} - y)) \hat{\Lambda}^{-1} - \gamma_2 v_2 (\hat{y} - y)^T B
\hat{\Lambda}^{-1}.
$$


# Review of Parameter Estimation Techniques

* **Time-Varying Parameter Identification Algorithm** [@rios_time-varying_2017]

Consider
$$
\begin{aligned}
	\dv{\theta(t)}{t} &= \Theta(wt), \\
	y(t) &= \Gamma^T(wt) \theta(t) + \varepsilon(t).
\end{aligned}
$$
In order to estimate the parameter, the following update law was introduced.
$$
\dot{\hat{\theta}}(t) = - K \Gamma(wt) \left\lceil \Gamma^T(wt) \hat{\theta}(t)
- y(t) \right\rfloor^\gamma,
$$
where $\lceil \cdot \rfloor^\gamma \coloneqq \abs{\cdot}^\gamma
\mathrm{sign}(\cdot)$.

* **Stochastic Gradient Descent in Continuous Time** 
[@sirignano_stochastic_2017]

Consider
$$
\dd{X_t} = f^\ast (X_t) \dd{t} + \sigma \dd{W_t},
$$
where the goal is to statistically estimate a model $f(x, \theta)$ for
$f^\ast$, where $\theta \in \MB{R}^n$.

# References
