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
H(q,p) = \sum_{j=1}^{N}\frac{p_j^2}{2m} + \sum_{j=1}^{N-1} V(q_{j}-q_{j+1})
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

### CPU

```
cmake -S . -B build-cpu
cmake --build build-cpu

./build-cpu/langevin_dynamics
```

### GPU

On the UNINA cluster, CMake always chooses the only cluster wide available C++ compiler as host compiler for nvcc, which is GNU 4.8.5 from 2015. This is too old and unnecessary as the micromamba environment provides a modern C++ compiler (g++ 12.4.). Therfore before configuring the build, export the `CUDAHOSTCXX` variable into the environment with 

```
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
```
Then, run:

```
cmake -S . -B build-gpu -DBUILD_GPU=ON
cmake --build build-gpu

./build-gpu/langevin_dynamics
```

## Plot

```
python -m venv .venv
source .venv/bin/activate
pip install -r extern/requirements.txt
```

The generic plotting file is provided with 
```
python scripts/plot.py <path>
```
To inspect all possibilities, run:

```
python scripts/plot.py <path> -h              // provides help
```

#### Automated Script
To run the simulation for FPU and Josephson potential and save the plots (energy heatmaps and pearson correlations), execute

```
python scripts/run.py
``` 


## Tests

To run tests, execute:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=DEBUG -DBUILD_TESTING=ON
cmake --build build

ctest --test-dir build --output-on-failure

```

## UNINA Cluster

Micromamba is provided under `~/.local/bin/micromamba`. The lightweight micromamba environment is called **`langevin_dynamics`**.

Inspect content with

```
micromamba list -n langevin_dynamics
```

To update according to .yml, run:

```
micromamba env update -n langevin_dynamics -f extern/unina_environment.yml
```

If you want to re-create the environment:

```
micromamba create -f extern/unina_environment.yml
```

To acivate the environment:

```
micromamba activate langevin_dynamics
```

## Memory bottlenecks

### of the GPU implementation

#### Block `__shared__` memory

Scales with

$$
\propto 8 \cdot N \quad \text{Bytes}
$$

I.e.:

```
N = 1,000  → 8 KB per block
N = 4,000  → 32 KB per block
N = 6,000  → 48 KB per block
N = 8,000  → 64 KB per block
```

Concretely for the `Tesla V100-SXM2-32GB`:

Theoretically 96KB, but practically (source: CUDA): 48 KiB. So this alone requires:

```
N < 6144
```

Practically, we can assume

```
shared per block     theoretical blocks per SM from shared memory alone
8 KiB                up to 12
16 KiB               up to 6
24 KiB               up to 4
32 KiB               up to 3
48 KiB               up to 2
>48 KiB              usually 1
```

>TODO: Test sensitivity on runtime (Block parallelism) for N>1000.


#### Global GPU memory

>**Is completely independent of `N_ensemble`.** The runtime of course not, but GPU memory is.

On GPU:

```
d_q                   [batch_size, N]
d_p                   [batch_size, N]
d_tot_e_temporary     [batch_size, N]
d_tot_e               [n_save, N]
```

So total GPU memory can be estimated with 8*

$$
\approx 2 \cdot [\text{batch-size} \cdot N] + \text{obs-arrays}  \cdot [\text{batch-size} \cdot N] +  \text{obs-arrays}  \cdot [\text{n-save} \cdot N]
$$

The CPU currently implements ~20 obs-arrays (=arrays for observables). This would mean 8*

$$
\boxed{
\approx 2 \cdot [\text{batch-size} \cdot N] + 20 \cdot ([\text{batch-size} \cdot N] +   [\text{n-save} \cdot N])
}
$$

or

$$
\boxed{
\boxed{
M_{\mathrm{GPU}}
\approx
8N \cdot \left[
2\cdot \text{batch-size}+20\cdot \text{batch-size}+20\cdot \text{n-save}
\right]\ \text{bytes}
}
}
$$

Standard vals:

$$
M_{\mathrm{GPU}} \approx 8 * 128 * [2 * 256 + 20 * (256 + 1000)] \approx 26 \text{MB} << 32GB
$$

 ---

##### n_save

Is determined with

```cpp
n_save = 1 + (N_time + save_every - 1) / save_every;
```

So we therefore need

$$
\boxed{
\lim_{N_t \to \infty} \frac{\text{N}_t+ \text{save-every}}{\text{save-every}} \approx 1000
}
$$

1000 in order to visually have a fine enough time resolution and at the same time a bounded memory demand.

>TODO: Remove user chosen save_every field and let the solver pick save_every dependent on $N_t$ to ensure bounded memory demand.

---

##### batch_size

The `batch_size` should be chosen small enough to keep global memory small enough and big enough to minimize kernel launch overhead and keep the GPU (SM) occupied with sufficient block-level parallelism. Let`s say 256 < batch_size < ?1024.

>TODO: Test for optimal `batch_size`.

The optimum will depend on how many trajectory blocks can reside on each SM, which depends strongly on N, shared memory, and register use.

>TODO: Test for optimal `threads_per_block`. Must be multiple of 32. Maybe perform_reduction should use a different threads_per_block setting since it is more dependent on current_batch_size, then N.

