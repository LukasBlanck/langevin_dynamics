# Langevin

## Theoretical Background

The Langevin Equation in it's simplest form can be written as

$$
\frac{dp(t)}{dt} = -\gamma p(t) + \xi(t)
$$

or depending on the convention of the damping factor $\gamma = \frac{\lambda}{m}$:

$$
m \frac{dv(t)}{dt} = -\lambda v(t) + \xi(t)
$$

$\xi(t)$ is the Gaussian Noise, created by the thermal energy of the smaller particles (molecules). It obeys the fluctuation-dissipation theorem:

$$
<\xi_i(t) , \xi_j(t')> = 2\gamma k_B T \delta_{i,j} \delta(t-t')
$$

and

$$
<\xi> = 0
$$

This means, that there is no time dependent correlation between forces at different times t and t'. On the time scale of the collision of the molecules this is of course not valid, but since we are only interested in the movement of the bigger grain, which has a bigger characteristic time scale, this approximation is valid.


### Hamiltonian

We write the most generic 1D Hamiltonian in the following form:

$$
H(q,p) = \sum_{j=1}^{N}\frac{p_j^2}{2m} + \sum_{j=1}^{N} V(q_{j}-q_{j+1})
$$

The potential V can be either an anharmonic oscillator:

$$
V(r) = \frac{w^2 r^2}{2} + \frac{\beta r^4}{4}
$$


or for the Josephson-Junction:

$$
V(r) = - E_J\cos(r)
$$

We used $\boxed{r = q_{j}-q_{j+1}}$.


#### Equations of Motions
###### À la Hamilton:

$$
\frac{dp_j}{dt} = -\frac{\partial H}{\partial q_j}, \qquad \frac{\partial q_j}{\partial t} = \frac{\partial H}{\partial p_j}
$$

Therfore:

$$
\frac{\partial p_j}{\partial t} = - [ \frac{\partial V(r_{j-1})}{\partial q_j} + \frac{\partial V(r_{j})}{\partial q_j}] -\lambda v(t) + \xi(t)
$$

or expressed explicitly:

$$
\boxed{
dp_j =  {\color{green}{ -\left[ \frac{\partial V(r_{j-1})}{\partial q_j} + \frac{\partial V(r_{j})}{\partial q_j} \right] }} dt -\lambda v(t)dt + \sqrt{2\lambda m k_B T_j(t)} {\color{red}{dW_j(t)}} }
$$

The red part might be expressed as "Wiener Increments", satisfying $<dW_i, dW_j> = \delta_{i,j}dt$


and 

$$
\boxed{
\frac{\partial q_j}{\partial t} = \frac{p_j}{m}
}
$$

The green part expresses the force $\color{green}{F_j}$.

This is now a Stochastic Differential Equation of the form $dX_t = a(X_t, t)dt +b(X_t,t)dW_t$


##### Force evaluation

$$
F_j(q)=-\frac{\partial H}{\partial q_j}.
$$

For interior sites,

$$
F_j=V'(r_{j-1})-V'(r_j),
\qquad 2\le j\le N-1
$$


At the boundaries,

$$
F_1=-V'(r_1), \qquad
F_N=V'(r_{N-1}).
$$

---

#### Quartic Potential

Explicit form:

$$
dp_j =
\left[
{\color{green}{
w^2(q_{j-1}-2q_j+q_{j+1})
+
\beta\left((q_{j-1}-q_j)^3-(q_j-q_{j+1})^3\right)
}}
-\lambda v(t)
\right]dt
+
\sqrt{2\lambda k_B T_j(t)}\,dW_j
$$


#### Josephson Potential
Explicit Form:

$$
dp_j = {[E_J \sin(q_{j-1} - q_j) - E_J \sin(q_{j} - q_{j+1}) - \lambda v(t)]}dt +  \sqrt{2\lambda k_B T_j(t)} dWj
$$

#### Thermal Bath
At the left end, a dissipation and random energy injection is provided with:

$$
\lambda_j = \lambda \delta_{1j}
$$


---
## Fokker Planck Equation
To verify numerical correctness, the Fokker Planck Equation could be solved. It should indicate a phase-space probability that should coincide with the results of the Langevin Solver.
I.e. if the 1D chain is in a thermal bath at temperature T, the canonical distribution should be valid: 

$$
\rho_{eq} = \frac{1}{Z} \exp{(-\frac{H}{k_B T})}
$$

which implies the numerical test

$$
<\frac{p_j^2}{m}> = k_B T
$$


--- 
## Numerical Solver
As a first step the 

- [BAOAB](docs/BAOAB.md) Splitting Method

is implemented. It should provide second order weak convergence.


#### Weak convergence
Weak convergence means that we can only determine the error/convergence for the ensemble avaerage of the determined quantity A:

$$
|\mathbb{E}[A(q_n, p_n)] - \mathbb{E}[A(q(t_n), p(t_n))]| = \mathrm{O}(\Delta t^2)
$$

#### Testing

###### Local Energy (symmetric)

$$
e_j = \frac{p_j^2}{2m} + 0.5V(r_j) + 0.5V(r_{j+1})
$$

$$H(t) = \sum _j <e_j(t)>$$

###### Scaling

$$
\sigma^2 = \frac{\sum_j(j-\bar{j})^2 \Delta e_j(t)}{\sum_j \Delta e_j(t)}
$$

$$
\sigma^2 \propto t^{\alpha}
$$

For $T=0 $ and $\lambda =0$ (only Quartic Potential) global energy conservation can be tested.


## Compile

```
cmake -S . -B build
cmake --build build

./build/langevin_dynamics
```