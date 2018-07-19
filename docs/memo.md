---
title: Composite MRAC
author: Seong-hun Kim
documentclass: my-handout
header-includes: |
...

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
dynamcis
$$
\dot{e} = A e(t) + B \Lambda \qty(u(t) + W^T \phi(t, x, c)).
$$
The unknowns are a diagonal matrix $\Lambda$ posessing positive elements, and a
parameter matrix $W$.

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
