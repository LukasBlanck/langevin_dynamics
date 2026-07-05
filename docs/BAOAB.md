## BAOAB Langevin Integrator

We consider the underdamped Langevin equations

$$
dq_j = \frac{p_j}{m}\,dt
$$

and

$$
dp_j =
F_j(q)\,dt
-\gamma_j p_j\,dt
+
\sqrt{2\gamma_j m k_B T_j}\,dW_j ,
$$

where

$$
F_j(q)=-\frac{\partial H}{\partial q_j}.
$$

For a bath coupled only to the left boundary, we choose

$$
\gamma_j=\gamma \delta_{j1}.
$$

For two boundary baths, one may instead use

$$
\gamma_j=\gamma_L\delta_{j1}+\gamma_R\delta_{jN}.
$$

The BAOAB method is obtained by splitting the Langevin dynamics into three exactly solvable parts:

$$
A:\quad dq_j=\frac{p_j}{m}\,dt,
$$

$$
B:\quad dp_j=F_j(q)\,dt,
$$

$$
O:\quad dp_j=-\gamma_jp_j\,dt+\sqrt{2\gamma_j m k_B T_j}\,dW_j.
$$

The $A$-step is the free streaming of positions, the $B$-step is the deterministic force kick, and the $O$-step is the Ornstein-Uhlenbeck thermostat.

The $O$-step can be solved exactly over one time step $\Delta t$:

$$
p_j \leftarrow
e^{-\gamma_j\Delta t}p_j
+
\sqrt{m k_B T_j\left(1-e^{-2\gamma_j\Delta t}\right)}\,Z_j,
$$

where

$$
Z_j\sim \mathcal N(0,1).
$$

If $\gamma_j=0$, this step leaves $p_j$ unchanged.

The BAOAB propagator is the symmetric Strang splitting

$$
e^{\Delta t(A+B+O)}
\approx
e^{\frac{\Delta t}{2}B}
e^{\frac{\Delta t}{2}A}
e^{\Delta t O}
e^{\frac{\Delta t}{2}A}
e^{\frac{\Delta t}{2}B}.
$$

Thus one full time step is:

$$
p_j \leftarrow p_j+\frac{\Delta t}{2}F_j(q),
$$

$$
q_j \leftarrow q_j+\frac{\Delta t}{2}\frac{p_j}{m},
$$

$$
p_j \leftarrow
e^{-\gamma_j\Delta t}p_j
+
\sqrt{m k_B T_j\left(1-e^{-2\gamma_j\Delta t}\right)}\,Z_j,
$$

$$
q_j \leftarrow q_j+\frac{\Delta t}{2}\frac{p_j}{m},
$$

$$
p_j \leftarrow p_j+\frac{\Delta t}{2}F_j(q).
$$

The force has to be evaluated twice per step, once before the first momentum kick and once after the second position drift.

The scheme is weakly second-order accurate for Langevin dynamics. This means that for sufficiently smooth observables $A(q,p)$,

$$
\left| \mathbb{E}[A(q_n,p_n)] - \mathbb{E}[A(q(t_n),p(t_n))] \right| = \mathcal O(\Delta t^2)
$$

This is the relevant notion of convergence for ensemble-averaged quantities such as kinetic-energy profiles, local energy profiles, and correlation functions.

In the limit $\gamma_j=0$, the $O$-step becomes the identity and BAOAB reduces to the standard velocity Verlet method for Hamiltonian dynamics.