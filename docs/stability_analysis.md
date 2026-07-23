# Stability Analysis

We have two Potentials:


| Bounded | Not-bounded |
|---|---|
| Josephson | FPU |

These two properties are also valid for their curvatures/stiffnesses/second_derivatives and are important when analysing our Verlet Algorithm [BAOAB](BAOAB.md) for a global stability bound.

---

**Remarks:**
The following analysis is based on the analysis performed by `Niels Grønbech-Jensen` in [**Linear Analysis of Stochastic Verlet-Type Integrators for Langevin Equations**](https://link.springer.com/article/10.1007/s10955-025-03553-3?utm_source=chatgpt.com).
>`Grønbech-Jensen` uses a harmonic potential (and therefore a linear force relation). FPU and Josephson are **not** harmonic and therefore get linearized to use `Grønbech-Jensen's` stability condition. To obtain a good approximation, we do this linearization over a physically argumented range $r \in [r_{min}, r_{max}]$.

---

We write our SDE as:

$$
\frac{dp(t)}{dt} + \gamma p(t) = -\nabla H+ \xi(t)
$$

with 

$$
H(r) = \frac{p^2}{2m} + V(r)
$$

## Harmonic Potential

#### Single Harmonic Oscillator

When performing the stability analysis for a **single** harmonic Potential of the form

$$
V(r) = \frac{1}{2} \kappa r^2
$$

the big adavantage is that you get a **linear** map for the force

$$
f(r^n) = -\nabla V = -\kappa r^n 
$$

Knowing the solution of a harmonic oscillator

$$
\omega_0 = \sqrt{\frac{\kappa}{m}}
$$

we can  also write 

$$
f(r^n) = -\nabla V = -m\omega^2_0 r^n 
$$

The global stability bound for [BAOB](docs/BAOAB.md) (performed in APPENDIX C to D in  [**Reference**](https://link.springer.com/article/10.1007/s10955-025-03553-3?utm_source=chatgpt.com)) is then given by the Eigenmode $\omega_0$ and $\Delta t$ with

$$
\boxed{
    |\omega_0 \Delta t| < 2
}
$$

---
---
Now, the reference to `Niels Grønbech-Jensen` is finished and we try to derive a stability criterion for the `FPU` and `Josephson` potential.
Both potentials do **not** result in a linear map for the force:

$$
f_{FPU}(r^n) = -\kappa r^n - \beta r^{3n}
$$

$$
f_{Jos}(r^n) = -E_J \cdot \sin(r)
$$

Since we can physically expect a regime $[r_{min}, r_{max}]$ (found by observing r vals from CPU implementation?) we can linearize our forces around an expectable $r_*$ and look at in the range of our r to determine our stability.

#### Linearization

We evaluate the potential  in a given range $r \in [r_{min}, r_{max}] := R$ and linearize the force at every possible $r \in R$. The force itself is given with

$$
f(r) = -\nabla V
$$

Linearization yelds:

$$
f(r + \delta r) = f(r) - V''(r)\delta r 
$$

where

$$
\left\{V''(r):
r\in[r_{\min},r_{\max}]
\right\}.
$$

Assuming a Lipschitz bounded force, we can use this as a global stability condition for $r\in R$:

$$
\boxed{ k_{\max} = \sup_{r\in[r_{\min},r_{\max}]} |V''(r)|}
$$

---

Explicit force linearization:

$$
f(r + \delta r) = f(r) + f'(r)\delta r + \mathrm{O}(\delta r^2)
$$

Since 

$$
-V''(r_{*}) = f'(r_{*})
$$

we write the linearized force:

$$
f(r_{*} + \delta r) = f(r_{*}) - V''(r_{*})\delta r 
$$

This is now again a Hooke-type force of homogenous part:

$$
m\ddot{\delta r} = - V''(r_{*})\delta r 
$$

and inhomogeneous part:

$$
m\ddot{\delta r} = - f(r_{*})
$$

We can write the homogeneous part as:

$$
\delta\ddot r+\frac{\kappa}{m}\delta r=0
$$

with analog solution

$$
\boxed{
\omega_{*}^2=\frac{V''(r_{*})}{m}.
}
$$


## FPU

$$
V''(r) = \kappa + 3\beta r^2
$$

The Eigenmode can then be written as

$$
\omega (r) = \sqrt{\frac{V''(r)}{m}} = \sqrt{\frac{\kappa + 3\beta r^2}{m}}
$$

We can therefore extract our stability condition with

$$
|\omega(r) \Delta t| < 2
$$

and 

$$
\Delta t < \frac{2}{w(r)} = 2 \sqrt{\frac{m}{\kappa + 3\beta r^2}} 
$$

We have to resolve up to the biggest Eigenmode and therefore use:

$$
\boxed{
    \Delta t < \min_{r\in[r_{min}, r_{max}]}2 \sqrt{\frac{m}{\kappa + 3\beta r^2}} 
}
$$

## Josephson

$$
-V''(r) = E_J \cdot \cos(r)
$$

We can again say

$$
\omega (r) = \sqrt{\frac{V''(r)}{m}} = \sqrt{\frac{E_J \cdot \cos(r)}{m}}
$$

and therefore

$$
\Delta t < \max_{r\in[r_{min}, r_{max}]}2 \sqrt{\frac{m}{E_J \cdot \cos(r)}} 
$$

Since cos( ) is bounded we can therfore assure a **globally** bounded Eigenmode and stability condition

$$
\boxed{
    \Delta t < 2 \sqrt{\frac{m}{E_J}} 
}
$$

---
---

Those are now the stabilty bounds for a **single** linearized force. Let's inspect what happens, when using a chain of nearest-neighbour coupled forces/sites.

## Coupled Chain

Let's switch to matrix form:

Define:

$$
B=
\begin{pmatrix}
1&-1&0&\cdots&0\\
0&1&-1&\cdots&0\\
\vdots&&\ddots&\ddots&\vdots\\
0&\cdots&0&1&-1
\end{pmatrix}
$$

for 

$$
(Bq)_j=q_j-q_{j+1}=r_j
$$

The potential this is:

$$
V(q)=\sum_{j=1}^{N-1}V((Bq)_j)
$$


The force is calculated with

$$
\nabla V(q) = B^{T} \begin{pmatrix} V'(r_1)\\ \vdots\\ V'(r_{N-1}) \end{pmatrix}
$$

Differentiating once more gives

$$
H(q)=\nabla^2U(q)=B^T W(q)B
$$

where

$$
W(q)=\operatorname{diag}
\left(
k_1,\ldots,k_{N-1}
\right),
\qquad
k_j=V''(r_j)
$$

$H$ is the so called weighted Laplacian. It has size [N x N].


$$
\begin{pmatrix}
1&0&\cdots&0\\
-1&1&0&\vdots\\
\vdots&-1&\ddots&\vdots\\
0&\cdots&\ddots&1\\
0&\cdots&0&-1
\end{pmatrix}\cdot 
\begin{pmatrix}
k_1&0&\cdots&0\\
0&k_2&0&\\
\vdots&&\ddots&\vdots\\
0&\cdots&0&k_{N-1}
\end{pmatrix}
\cdot
\begin{pmatrix}
1&-1&0&\cdots&0\\
0&1&-1&\cdots&0\\
\vdots&&\ddots&\ddots&\vdots\\
0&\cdots&0&1&-1
\end{pmatrix} 
$$

Therefore

$$
H=
\begin{pmatrix}
k_1&-k_1&0&\cdots&0\\
-k_1&k_1+k_2&-k_2&\cdots&0\\
0&-k_2&k_2+k_3&\ddots&\vdots\\
\vdots&\vdots&\ddots&\ddots&-k_{N-1}\\
0&0&\cdots&-k_{N-1}&k_{N-1}
\end{pmatrix}
$$

We now try to estimate an upper bound for our Eigenvalues. We could of course calculate the Eigenvalues themselves, but this would probably be too expensive (but let's see).

### PDE

The PDE is now:

$$
M\,\delta\ddot q=-H(q)\,\delta q.
$$

We can find an Eigenvalue problem with assuming a solution of form:

$$
\delta q(t)=u\,a(t),
$$

Then:

$$
\delta\ddot q(t)=u\,\ddot a(t)
$$

Assuming an oscillatory (stable) Eigenmode:

$$
a(t)=e^{i\omega t}
$$

we get

$$
-M\,uw^2=-H(q)\,u.
$$

Therefor:

$$
M^{-1}H\,  u= \omega^2\, u
$$

With 
* $u\in\mathbb R^N$ is a fixed spatial displacement pattern;
* $a(t)$ is its time-dependent amplitude.

We therefore need to resolve every possible Eigenmode (=Eigenvalue) and search for the biggest Eigenvalue.
The stability condition is:

$$
\boxed{
    \Delta t^2 < \max_{\lambda}\frac{4}{\lambda^2(M^{-1}H)}
}
$$

#### Upper bound

Using Rayleigh-Quotient we can find our biggest Eigenvalue $\lambda_{max}$ and can avoid calculating the Eigenvalues explicitly. For small N this would probably be absolutely doable, but since the curvature gives us an even stricter stability condition, the calculation of Eigenvalues is not necessary.
We of course now don't calculate any Rayleigh-Quotient numerically, but upper bound it`s maximum with the maximum curvature of the Potential.

##### Assumption

Let's assume:

$$
|k_j|\le k_{\max}
$$

This is strictly valid for the Josephson potential, but no "real" upper bound can be found for the curvature of the FPU potential, as mentioned before. We use the predefined Range $R$ to resolve this problem.

For any vector $x \in \mathbb{R}^N$, we can write:

$$
x^THx=x^TB^TWBx= (Bx)^TW(Bx).
$$

Since

$$
(Bx)_j=x_j-x_{j+1},
$$

we get

$$
x^THx=\sum_{j=1}^{N-1}k_j(x_j-x_{j+1})^2.
$$

In the general case $x^THx$ is not positive semidefinit and we use 

$$
|x^THx|=\sum_{j=1}^{N-1}|k_j(x_j-x_{j+1})^2|
$$

$$
|x^THx| \le \sum_{j=1}^{N-1}|k_j|(x_j-x_{j+1})^2
$$

We can upper bound this further with our biggest curvature $k_{max}$:

$$
|x^THx| \le \sum_{j=1}^{N-1}|k_{max}|(x_j-x_{j+1})^2
$$

Now using

$$
(a - b)^2 \le 2a^2 + 2b^2
$$

we obtain

$$
|x^THx| \le 2\cdot |k_{max}|\sum_{j=1}^{N-1}(x_j^2 + x_{j+1}^2)
$$

Let's explicitly look at 

$$
\sum_{j=1}^{N-1}(x_j^2 + x_{j+1}^2) = x_1^2 + 2 \sum_{j=2}^{N-1}x_j^2 + x_{N-1}^2
$$

We can upper bound this with

$$
x_1^2 + 2 \sum_{j=2}^{N-1}x_j^2 + x_{N-1}^2 \le 2x_1^2 + 2 \sum_{j=2}^{N-1}x_j^2 + 2x_{N-1}^2 = 2 \sum_{j=1}^{N}x_j^2 = 2 x^Tx
$$

Therefore:

$$
|x^THx| \le 4\cdot |k_{max}| x^Tx
$$

The Rayleigh Quotient is:

$$
\frac{|x^THx|}{x^Tx} \le 4\cdot |k_{max}| 
$$

We know from [Courant-Fischer-Theorem](https://en.wikipedia.org/wiki/Min-max_theorem), that the biggest Rayleigh Quotient is the biggest Eigenvalue for real symmetric Matrices:

$$
\boxed{ \max_{x\neq0} \frac{|x^THx|}{x^Tx} = \max_i|\lambda_i(H)| }
$$


We can therefore find our stability condition with
$$
|\lambda_{max}| = R_{A,max} \le4\cdot |k_{max}| 
$$

FPU:

$$
\boxed{
    |\lambda_{max}| \le 4 (\kappa + 3\beta r_{max}^2)
}
$$

Josephson:

$$
\boxed{
    |\lambda_{max}| \le 4 |E_J|
}
$$

---
---


$$
\Delta t \le \sqrt{\frac{m}{E_J}}
$$

$$
\Delta t \le \sqrt{\frac{m}{\kappa + 3\beta r_{max}^2}}
$$

---
---
