---
title: "The only Muon Optimizer guide you need"

date: 2026-01-25
draft: false
math: true
toc: true
---

All neural networks uses a form of gradient descent for updating their parameters. The fundamental intuition to all neural net's parameter optimization seems obvious to us, i.e., to move opposite to the gradient. However, there are important caveats to the obvious intuition of following direction opposite to the gradient for optimization. For instance, what curvature to follow along the steepest descent? the scale to which we should move at each step? and the stability of each movement over an unoptimized loss landscape.

To allow for a controlled gradient descent that tackles around these caveats, most optimizers follow a template of **constrained linearized improvement** such that, we solve:
$$ \min_{\Delta\theta} \langle g, \Delta\theta \rangle \quad \text{subject to} \quad |\Delta\theta| \leq \eta $$

where,

* $\theta \rightarrow$ current parameters
* $g = \nabla_\theta \mathcal{L} \rightarrow$ gradient
* $|\cdot| \rightarrow$ a notion of step size via some norm

which implies to find a direction that decreases the linearized loss the most, given that we're not stepping "too far" according to the bounds of our chosen constrain.

Before we dive deep into Muon's architecture, it serves us well to look at one of the most fundamental algorithms, Stochastic Gradient descent as a solution to the **constrained linearized improvement** and understand where the solution lags behind in achieving the optimal loss minimization curve.

## Stochastic Gradient Descent (SGD) as simplified steepest descent under Euclidean metrics.

SGD is a simplified technique for steepest gradient descent where we use a standard Euclidean (L2) norm for constrained optimization, i.e., $|\Delta\theta|_2 = \sqrt{\sum_i \Delta\theta_i^2}$.

### Solving the constrained problem

We want:
$\min_{\Delta\theta} \langle g, \Delta\theta \rangle \quad \text{subject to} \quad \lVert\Delta\theta \rVert_2 \leq \eta$

The geometrically obvious solution to this is *to move in the opposite direction to the gradient*.

$\Delta\theta^* = -\eta \frac{g}{|g|_2}$

With a given learning rate of $\alpha$, this becomes:

$\theta_{t+1} = \theta_t - \alpha g_t$

<div style="margin-left: 2em; font-size: 0.85em;"><em>
Let's take a look at a 2D quadratic loss (elliptical contours) tracing a SGD path. The blue polyline is the sequence of SGD steps from the start (green) to orange (end).

Increasing the learning rate here leads to faster progress (and potential overshoot), raising the step count lengthens the trajectory, and moving the start point samples different regions of the curvature.</em></div>

<div class="ml-interactive" data-ml="sgd-basic">
  <div class="ml-controls">
    <label>
      Learning rate
      <input type="range" min="0.01" max="0.25" value="0.08" step="0.01" data-role="lr" />
      <span class="ml-readout" data-role="lr-val">0.08</span>
    </label>
    <label>
      Steps
      <input type="range" min="5" max="60" value="25" step="1" data-role="steps" />
      <span class="ml-readout" data-role="steps-val">25</span>
    </label>
    <label>
      Start x
      <input type="range" min="-3.5" max="3.5" value="2.5" step="0.1" data-role="x0" />
      <span class="ml-readout" data-role="x0-val">2.5</span>
    </label>
    <label>
      Start y
      <input type="range" min="-3.5" max="3.5" value="2.0" step="0.1" data-role="y0" />
      <span class="ml-readout" data-role="y0-val">2.0</span>
    </label>
    <button type="button" class="ml-button" data-role="reset">Reset start</button>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="SGD trajectory on quadratic bowl">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="contours"></g>
    <path data-role="path" fill="none" stroke="#1f77b4" stroke-width="3"></path>
    <circle data-role="start" r="5" fill="#2b8a3e"></circle>
    <circle data-role="end" r="5" fill="#d9480f"></circle>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#1f77b4"></span>SGD path</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>start</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>end</span>
  </div>
</div>

In most practical use-cases of deep learning, the Euclidean metric choice for steepest descent in SGD with a single global learning rate is often poorly matched to the induced geometry of the loss landscapes.

The loss landscapes are often ill-conditioned, with large disparities in curvature across parameter directions, where uneven curvature and gradient magnitudes across different parameter directions lead SGD to use conservatively small global step sizes, while the ill-conditioned space further amplifies the effect of gradient noise, as high curvature makes updates more sensitive to noise.

---

With these limitations around the usage of a global learning step and difficulties in navigating the vastly uneven loss landscape with varying scale of magnitudes for gradients under different directions of parameter space, it would likely serve us well to use some form of coordinate-adaptable step function that tunes to the nature of varying gradient scales and curvatures along each direction.

## Adam

One of the most commonly used optimizers today, Adam, tries to mitigate this via applying a diagonal, coordinate-wise scaling of updates in parameter space, tuning the effective step size on a per-parameter bases and reducing sensitivity to gradient scale differences. Adam maintains running estimates of the first moment and second raw moment of gradients, which helps normalize gradient scales across coordinates and stabilize updates in uneven gradient regimes, making it more robust to gradient scale changes.

To expand on the problem, consider a deep network with varying gradient scales s.t. $g_1 \approx 10^{-6}$ and $g_2 \approx 10^2$, a single learning rate $\alpha$ is painful to tune here since:

* Too large $\implies \theta_2$ explodes
* Too small $\implies \theta_1$ barely moves

<div style="margin-left: 2em; font-size: 0.85em;"><em>Let's take a look at a loss landscape with very different curvature along each axis (steep in one direction, flat in the other). The same global learning rate principle from SGD produces zig-zagging updates here, while a simple per-coordinate rescaling stabilizes the path.

The curvature ratio controls how anisotropic the loss landscape is (higher means one direction is much steeper), we can compare both plain SGD and coordinate-scaled learning here.</em></div>

<div class="ml-interactive" data-ml="sgd-scaling">
  <div class="ml-controls">
    <label>
      Curvature ratio (a:b)
      <input type="range" min="1" max="60" value="25" step="1" data-role="ratio" />
      <span class="ml-readout" data-role="ratio-val">25</span>
    </label>
    <label>
      Learning rate
      <input type="range" min="0.005" max="0.1" value="0.03" step="0.005" data-role="lr" />
      <span class="ml-readout" data-role="lr-val">0.12</span>
    </label>
    <label>
      Steps
      <input type="range" min="5" max="60" value="30" step="1" data-role="steps" />
      <span class="ml-readout" data-role="steps-val">30</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Gradient scaling comparison">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="contours"></g>
    <path data-role="path-sgd" fill="none" stroke="#d9480f" stroke-width="3"></path>
    <path data-role="path-pre" fill="none" stroke="#2b8a3e" stroke-width="3"></path>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>SGD (single lr)</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>coordinate-scaled</span>
  </div>
</div>

Now, we define "distance" **coordinate-wise** using a diagonal scaling.

Let $d_i > 0$ be per-coordinate scale factors, and define

$|\Delta\theta|_{d}^2 := \sum_i d_i,\Delta\theta_i^2.$

This is a valid norm (equivalently $|\Delta\theta|_d^2 = \Delta\theta^\top D,\Delta\theta$ with $D=\mathrm{diag}(d)$). The steepest descent solution under this metric will divide the gradient by $d_i$.

Now, again solving the constrained linearized improvement with this diagonal norm, we get

$\min_{\Delta\theta} \langle g, \Delta\theta \rangle \quad \text{subject to} \quad \sum_i d_i,\Delta\theta_i^2 \leq \eta^2$

Applying the Lagrangian multiplier, we get

$\mathcal{J}(\Delta\theta, \lambda) = \sum_i g_i \Delta\theta_i + \lambda \left( \sum_i d_i,\Delta\theta_i^2 - \eta^2 \right)$

and the stationary condition turns out to be

$\frac{\partial \mathcal{J}}{\partial \Delta\theta_i} = g_i + 2\lambda, d_i,\Delta\theta_i = 0$

Solving for $\Delta\theta_i$ we get

$\Delta\theta_i = -\frac{1}{2\lambda},\frac{g_i}{d_i}$

The update direction is

$\Delta\theta \propto -\mathrm{diag}(d)^{-1} g$

In Adam, $d_i$ is taken to be roughly $\sqrt{\hat{s}_{t,i}} + \epsilon$ (a smoothed estimate of the RMS gradient at coordinate $i$), which yields the classic division by $\sqrt{\hat{s}_t}+\epsilon$.

### Adam algorithm

Now, that we have established Adam as steepest descent under a diagonal coordinate-wise scaled metric.

<div style="margin-left: 2em; font-size: 0.85em;"><em>Let's take a look at how momentum averages gradients over time and can reduce zig-zagging on ill-conditioned loss landscapes. Increasing beta smooths the update direction; at a fixed (stable) learning rate, the path straightens along the shallow direction instead of bouncing across the steep axis.</em></div>

<div class="ml-interactive" data-ml="sgd-momentum">
  <div class="ml-controls">
    <label>
      Momentum (beta)
      <input type="range" min="0" max="0.95" value="0.8" step="0.01" data-role="beta" />
      <span class="ml-readout" data-role="beta-val">0.80</span>
    </label>
    <label>
      Learning rate
      <input type="range" min="0.005" max="0.08" value="0.03" step="0.005" data-role="lr" />
      <span class="ml-readout" data-role="lr-val">0.12</span>
    </label>
    <label>
      Steps
      <input type="range" min="5" max="70" value="35" step="1" data-role="steps" />
      <span class="ml-readout" data-role="steps-val">35</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Momentum path on ill-conditioned bowl">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="contours"></g>
    <path data-role="path-sgd" fill="none" stroke="#adb5bd" stroke-width="2"></path>
    <path data-role="path-mom" fill="none" stroke="#1f77b4" stroke-width="3"></path>
    <g data-role="axes"></g>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#1f77b4"></span>momentum</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#adb5bd"></span>plain SGD</span>
  </div>
</div>

The complete Adam algorithm is

**First moment (momentum):**
$v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t$

**Second moment (gradient variance):**
$s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(elementwise)}$

**Bias correction:**
$\hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$

**Update:**
$\Delta\theta_t^{\text{Adam}} = -\eta \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$

```python
def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam optimizer update step. Return (param_new, m_new, v_new)."""
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))
    return param_new, m_new, v_new
```

---

However, diagonals are scaled independently with no cross-coordinate coupling and Adam still does not account for cross-parameter curvature.

If we consider a linear layer with weight matrix $W \in \mathbb{R}^{m \times n}$.

For the transformation $y = Wx$.

Under, Adam's coordinate-wise scaling, $W_{ij}$ and $W_{kl}$ are treated as independent.

Adam treats $W$ as a flattened vector of $m \times n$ independent parameters. But, in practical realities, $W$ is a **linear operator** often with cross-coordinate coupling, and low effective rank.

## Inuition for Muon Optimization

Returning back to the core issue around *cross-coordinate coupling* and *direction imbalance*. Let's try to understand the nature of a linear layer, a linear layer computes $y = Wx$.

When we update the weights $W \to W + \Delta W$, the output changes to
$y_{new} = (W + \Delta W)x = Wx + \Delta W \cdot x = y + \Delta y$
where $\Delta y = \Delta W \cdot x$.

Under the geometrical intuition of Muon, we don't directly care how much $W$ changed entry-by-entry. We care how much $\Delta y$ can be for typical inputs x, i.e., how much the layer's behavior changed.

Thus, the constrain in Muon for $\Delta W$ should rather be, "how much can it change outputs?" instead of "how big are the changes in its individual entries".

Muon optimization is typically used for dense linear layers, where activations (after normalization layers) tend to have entries of order 1, i.e., not too big, not too small.

With the **RMS (Root Mean Square) norm**:
$\lVert v \rVert_{RMS} := \sqrt{\frac{1}{d} \sum_{i=1}^dv_i^2}$
If all entries are $\pm 1$, then $\lVert v \rVert_{\mathrm{RMS}} = 1 \implies \lVert v \rVert_{\mathrm{RMS}} = \frac{\lVert v \rVert_2}{\sqrt{d}}$. As inferred from above, dense neural network activations (post normalization) typically have $\lVert v \rVert_{\mathrm{RMS}} \approx 1$.

### The Operator Norm

For a matrix $M$ acting on a vector $x$: $y = Mx$

The condition number $\kappa(M) = \sigma_{\max}(M) / \sigma_{\min}(M)$ captures the relative difficulty of optimization by quantifying how differently the loss responds to parameter updates along its steepest and flattest directions.

To measure "how much a matrix can stretch vectors", we use an **operator norm**:
$\lVert M \rVert_{op} := \max_{ x \neq 0} \frac{\lVert Mx \rVert}{\lVert x \rVert}$
i.e., the maximum factor by which $M$ can stretch a vector's norm.

e.g., for the standard Euclidean norm, this equals the **spectral norm**:
$\lVert M \rVert_2 = \sigma_{max}(M)$
i.e., the largest singular value of $M$.

Since, we're measuring activations with RMS norm, we define the **RMS-to-RMS operator norm**:
$ \lVert M \rVert_{RMS \rightarrow RMS} := \max_{x \neq 0} \frac{\lVert Mx \rVert_{RMS}}{\lVert x \rVert_{RMS}}$

Consider, a matrix $M \in \mathbb{R}^{m \times n}$ with

$\lVert Mx \rVert_{\text{RMS}} = \frac{\lVert Mx \rVert_2}{\sqrt{m}}$ > $\text{(RMS over Mx averages over m components)}, \quad \lVert x \rVert_{\text{RMS}} = \frac{\lVert x \rVert_2}{\sqrt{n}} > \text{(RMS over x averages over n components)} $

Therefore:

$$\lVert M \rVert_{RMS \to RMS} = \max_{x \neq 0} \frac{\lVert Mx \rVert_2 / \sqrt{m}}{\lVert x \rVert_2 / \sqrt{n}} = \sqrt{\frac{n}{m}} \cdot \max_{x \neq 0} \frac{\lVert Mx \rVert_2}{\lVert x \rVert_2} = \sqrt{\frac{n}{m}} \cdot \sigma_{\max}(M)$$

Or in terms of $\text{fan-in} \ (n)$ and $\text{fan-out} \ (m)$:

$$\lVert M \rVert_{RMS \to RMS} = \sqrt{\frac{\text{fan-in}}{\text{fan-out}}} \cdot \lVert M \rVert_*$$

where $\lVert M \rVert_* = \sigma_{\max}(M)$ is the spectral norm.

### Formulating Muon's constrained Optimization

Now, we precisely bound the affect of weight change on outputs.

Since $ \Delta y = \Delta W \cdot x$:

$$\lVert \Delta y\rVert_{RMS} = \lVert \Delta W \cdot x \rVert_{RMS} \leq \lVert \Delta W \rVert_{RMS \to RMS} \cdot \lVert x \rVert_{RMS}$$

If inputs satisfy $\lVert x \rVert_{\text{RMS}} \leq 1$ (typical for normalized activations), then:

$$\lVert \Delta y \rVert_{RMS} \leq \lVert \Delta W \rVert_{RMS \to RMS}$$

Thus, the RMS-to-RMS operator norm of $\Delta W$ directly bounds how much the layer output can change.

Applying the template of **constrained linearized improvement**, we get 

$$\min_{\Delta W} \langle \nabla_W \mathcal{L}, \Delta W \rangle \quad \text{subject to} \quad |\Delta W|_{\text{RMS}\to\text{RMS}} \leq \eta \tag{✧}$$

Instead of constraint over parameter update, we find the weight update $\Delta W$ that maximizes descent along the gradient, subject to bounding how much the layer's output can change.

### Orthogonalization as a Scalable Mechanism for Parameter Updates in Muon

The constraint $\lVert \Delta W \rVert_{RMS \rightarrow RMS} \leq \eta $ involves the spectral norm (largest singular value).

Consider the SVD of $ \Delta W$:
$\Delta W = U \Sigma V^\top = \sum_{i=1}^r \sigma_i u_i v_i^\top$
where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ are singular values, $u_i$ are left singular vectors, and $v_i$ are right singular vectors.

The spectral norm for such $\Delta W$ is $\lVert \Delta W \rVert_* = \sigma_1$, the largest singular value.

Matrices with $\lVert \Delta W \rVert_* \leq \eta$ are exactly those where *no singular value exceeds $\eta$*.

If we consider matrices where *all* singular values equal some constant $c$:
$\sigma_1 = \sigma_2 = \cdots = \sigma_r = c$.

Such matrices have the form $\Delta W = c \cdot Q$ where $Q = UV^\top$ satisfies
$Q^\top Q = VU^\top UV^\top = VV^\top = I $ (when $r=\min(m,n)$)

A matrix $Q$ with $Q^\top Q = I$ is called **semi-orthogonal** (or **orthogonal** if square matrix)

Now, given any matrix $M$, what's the "closest" semi-orthogonal matrix?
Formally, we solve $\min_{Q:Q^\top Q=I} |M - Q|_F$

The Frobenius norm $|M - Q|_F^2 = |U\Sigma V^\top - Q|_F^2$.

Since $U$ and $V$ are orthogonal, we can write $Q = U\tilde{Q}V^\top$ for some orthogonal $\tilde{Q}$.

The problem becomes minimizing $|\Sigma - \tilde{Q}|_F^2$. For diagonal $\Sigma$ with non-negative entries, the closest orthogonal matrix is $\tilde{Q} = I$,

giving $Q^* = UIV^\top = UV^\top$.

$ Q^* = UV^\top$ is called the **polar factor** of $M$.

We can identify any matrix M as:
$$M = (UV^\top)(V\Sigma V^\top) = QP$$

where:

* $Q = UV^\top$ is orthogonal (or semi-orthogonal) $\rightarrow$ the **rotation/reflection** part
* $P = V\Sigma V^\top$ is symmetric positive semi-definite $\rightarrow$ the **stretch** part

An intuition to orthogonalization is *discarding the stretch, retaining the rotation*.

<div style="margin-left: 2em; font-size: 0.85em;"><em>Let's visualize a $ 2 \times 2$ weight matrix acting on the unit circle. The spectral norm is the maximum stretch. The polar factor removes stretching while preserving rotation, echoing Muon's orthogonalized update.

The sliders labeled a, b, c, d are the entries of the weight matrix
`W = [[a, b], [c, d]]`. The orange curve shows how W stretches the unit circle; the green curve shows the closest orthogonal (polar) factor that preserves rotation but removes stretching. The dashed ring is an RMS gain guide that scales with fan-in/out via `sigma_max * sqrt(n/m)`; as we change n and m, the ring (and scale) adjusts even if the matrix entries stay fixed.</em></div>

<div class="ml-interactive" data-ml="operator-norm">
  <div class="ml-controls">
    <label>
      a
      <input type="range" min="-2" max="2" value="1.4" step="0.05" data-role="a" />
      <span class="ml-readout" data-role="a-val">1.40</span>
    </label>
    <label>
      b
      <input type="range" min="-2" max="2" value="0.4" step="0.05" data-role="b" />
      <span class="ml-readout" data-role="b-val">0.40</span>
    </label>
    <label>
      c
      <input type="range" min="-2" max="2" value="-0.6" step="0.05" data-role="c" />
      <span class="ml-readout" data-role="c-val">-0.60</span>
    </label>
    <label>
      d
      <input type="range" min="-2" max="2" value="1.1" step="0.05" data-role="d" />
      <span class="ml-readout" data-role="d-val">1.10</span>
    </label>
    <label>
      fan-in (n)
      <input type="range" min="8" max="256" value="64" step="8" data-role="n" />
      <span class="ml-readout" data-role="n-val">64</span>
    </label>
    <label>
      fan-out (m)
      <input type="range" min="8" max="256" value="64" step="8" data-role="m" />
      <span class="ml-readout" data-role="m-val">64</span>
    </label>
  </div>
  <svg class="ml-plot" viewBox="0 0 640 360" role="img" aria-label="Operator norm and polar factor visualization">
    <rect x="0" y="0" width="640" height="360" fill="white"></rect>
    <g data-role="grid"></g>
    <g data-role="axes"></g>
    <path data-role="unit" fill="none" stroke="#adb5bd" stroke-width="2"></path>
    <path data-role="mapped" fill="none" stroke="#d9480f" stroke-width="3"></path>
    <path data-role="polar" fill="none" stroke="#2b8a3e" stroke-width="3"></path>
    <path data-role="rms" fill="none" stroke="#845ef7" stroke-width="2" stroke-dasharray="6 6"></path>
  </svg>
  <div class="ml-legend">
    <span class="ml-chip"><span class="ml-dot" style="background:#adb5bd"></span>unit circle</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#d9480f"></span>W circle</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#2b8a3e"></span>polar(W) circle</span>
    <span class="ml-chip"><span class="ml-dot" style="background:#845ef7"></span>RMS gain ring</span>
    <span class="ml-chip">s_max: <span class="ml-readout" data-role="sigma">0.00</span></span>
    <span class="ml-chip">RMS gain: <span class="ml-readout" data-role="rms">0.00</span></span>
  </div>
</div>

### Solving Muon's Constrained Optimization

Now we solve (✧):

$$\min_{\Delta W} \langle G, \Delta W \rangle \quad \text{s.t.} \quad \lVert \Delta W \rVert_{\text{RMS} \to \text{RMS}} \leq \eta$$

where $G = \nabla_W \mathcal{L}$ is the gradient and $W \in \mathbb{R}^{m \times n}$ (fan-out $m$, fan-in $n$).

**Which direction decreases the objective fastest?**

Consider the change in $\Delta W$ for a small step $\varepsilon H$:

$\langle G, \Delta W + \varepsilon H \rangle = \langle G, \Delta W \rangle + \varepsilon \langle G, H \rangle$

Here, the rate of change depends entirely on $\langle G, H \rangle$.

To decrease the objective, we need $\langle G, H \rangle < 0$, and the most negative value occurs when $H$ points opposite to $G$. So the steepest descent direction is $-G$.

With our operator norm constraint of $\lVert \Delta W \rVert_{RMS \rightarrow RMS} \leq \eta$. Since the objective is linear in $\Delta W$, the minimum lies on the **boundary** $\lVert \Delta W \rVert_{\text{RMS} \to \text{RMS}} = \eta$. Now, we want the direction on this boundary that achieves maximal alignment with $-G$.

**Finding the optimal direction.**

If we want to maximize $\langle -g, v \rangle$ subject to $|v| = 1$, the answer is $v = -g / |g|$. We normalize $g$ to get a unit vector pointing in the same direction.

For matrices, we want the same thing: find the matrix $Q$ with $|Q|_{\text{op}} = 1$ that maximizes $\langle -G, Q \rangle$. The matrix analogue of "normalize to unit length" is to project onto the unit operator-norm boundary. This replaces $G$ by its polar factor $U_G V_G^\top$, which preserves the singular directions of $G$ while discarding their relative magnitudes.

Why does this work? If we express $G$ as $U_G \Sigma_G V_G^\top$ (SVD), then:

$$\langle G, U_G V_G^\top \rangle = \text{tr}(G^\top U_G V_G^\top) = \text{tr}(V_G \Sigma_G U_G^\top U_G V_G^\top) = \text{tr}(\Sigma_G) = \sum_i \sigma_i$$

This is the sum of all singular values, i.e., the maximum possible inner product with any matrix of spectral norm 1. So $-U_G V_G^\top$ is the unit-norm matrix most aligned with $-G$, just as $-g/|g|$ is the unit vector most aligned with $-g$.

**Scaling to saturate the constraint.**

Now, we want $\Delta W = -c \cdot U_G V_G^\top$ for some $c > 0$. Since $U_G V_G^\top$ has spectral norm = 1:

$\lVert \Delta W \rVert_{\text{RMS} \to \text{RMS}} = \sqrt{\frac{n}{m}} \cdot |{-c \cdot U_G V_G^\top}|*{\text{op}} = c \sqrt{\frac{n}{m}}$

Setting this equal to $\eta$:

$c \sqrt{\frac{n}{m}} = \eta \quad \Rightarrow \quad c = \eta \sqrt{\frac{m}{n}} = \eta \sqrt{\frac{\text{fan-out}}{\text{fan-in}}}$

$$\Delta W^* = -\eta \sqrt{\frac{\text{fan-out}}{\text{fan-in}}} \cdot U_G V_G^\top \tag{✧}$$

$[$ The $\sqrt{\text{fan-out}/\text{fan-in}}$ factor can be handled explicitly (shape scaling) or implicitly (learning-rate adjustment / update-RMS calibration) $]$

$ \implies$ Muon's optimal update is the **orthogonalized gradient**, i.e, the polar factor $U_G V_G^\top$ scaled by a shape-dependent factor.

---

**What do we orthogonalize?**

So far, we solved a *per-step* constrained problem and got a closed-form step:
$\Delta W^* \propto -,U_G V_G^\top.$

> In SGD/Adam, we step using some processed version of the mini-batch gradient. In Muon, what matrix should we feed into the "orthogonalizer" to get $UV^\top$?

The most naive choice would be $ \Delta W_t \propto \text{-polar}(G_t)$ where $G_t = \nabla_W \mathcal{L}(W_t)$.

However, **mini-batch gradient matrices are typically low rank**. If we force a low-rank matrix to be orthogonal (full rank) via a full SVD/polar computation, we're effectively *inventing directions* in the null space where the gradient was actually zero. Thus, **amplifying noise!**

Hence, Muon introduces a *momentum buffer* first, it accumulates gradients acorss steps so the matrix we orthogonalize becomes representative (and closer to full rank).

**Momentum Buffer** $B_t$:
$ B_t = \mu B_{t-1} + G_t$

Because each step's gradient lies in a slightly differnt subspace, the sum rapidly becomes **higher rank**.

## Muon: The Core Algorithm

Now, we know that Muon optimization wants matrix updates on the direction of polar factor $UV^\top$ obtained from **momentum buffer** $B_t$.

At a high level, Muon is:

$G_t \xrightarrow{\text{accumulate}} B_t \xrightarrow{\text{orthogonalize}} O_t \xrightarrow{\text{step}} W_{t+1}$

Let $W_t \in \mathbb{R}^{m \times n}$ be a weight matrix and $G_t = \nabla_W \mathcal{L}(W_t)$.

With hyperparameters:

* learning rate $ \rightarrow \gamma$
* momentum $ \rightarrow \mu$
* weight decay $ \rightarrow \lambda$

We initialize $B_0 = 0$.

Iterating for t=1,2,...

1. Accumulate momentum buffer:
   $B_t = \mu B_{t-1} + G_t$

2. Orthogonalize (over a unit norm):
   $O_t = \text{Ortho}(B_t) \approx \text{polar}(B_t) = U_t V_t^\top$

3. Scale (shape/update-RMS calibration): choose a scalar so the update RMS is comparable across shapes and matches a desired target.

   Two widely used choices in practice are:

   * **Shape-based LR adjustment** (Keller / “original” style): adjust the effective step based on matrix dimensions to reduce RMS drift across rectangular shapes.

   * **AdamW RMS-matching** (Moonshot / “match_rms_adamw” style): target an update RMS similar to AdamW (often around 0.2–0.4), implemented via a shape-dependent multiplier (e.g., proportional to $\sqrt{\max(m,n)}$) or by explicitly normalizing the update RMS.

4. Update with decoupled weight decay:
   $W_{t+1} = W_t - \gamma, O_t - \gamma,\lambda, W_t$

### Newton-Schulz as an approximation to SVD

SVD calculation is computationally expensive, $ O(\min(m,n) \cdot mn)$ for an $ m \times n$ weight matrix. Also, while batched SVD exists, it is still heavy compared to GEMMs, and can become a bottleneck at LLM scale.

So, we use the following identity as the conceptual bridge:

For any matrix $A$ with SVD $A = U\Sigma V^\top$:

$A^\top A = V\Sigma^2 V^\top$

Now consider the matrix inverse square root $(A^\top A)^{-1/2}$:

$(A^\top A)^{-1/2} = V\Sigma^{-1} V^\top$

What happens when we multiply $A$ by this?

$A (A^\top A)^{-1/2} = U\Sigma V^\top \cdot V\Sigma^{-1} V^\top = U\Sigma \Sigma^{-1} V^\top = UV^\top$

**The polar factor equals $A$ times the inverse square root of $A^\top A$!**

$\text{polar}(A) = UV^\top = A(A^\top A)^{-1/2}$

At first, computing $(A^\top A)^{-1/2}$ seems no easier than SVD. Both involve the spectrum.

But Muon does *not* compute an explicit inverse square root. Instead, it uses a Newton–Schulz *polar iteration* that applies a low-degree polynomial to $X_k$ so that singular values are pushed toward 1, using only matrix multiplications.

Muon uses a polynomial variant with optimized coefficients:
$(a, b, c) = (3.4445, -4.7750, 2.0315)$

Each iteration applies (with Gram matrix feedback):

* Gram matrix: $A_k = X_k X_k^\top \quad (m \times m)$
* Polynomials: $A_k^2 = A_k A_k$
* update:

$X_{k+1} = aX_k + (bA_k + cA_k^2)X_k$

If $A_k \approx I$, then the rows of $X_k$ are approximately orthonormal. Newton–Schulz uses $A_k$ (and powers of $A_k$) as feedback to push $A_k$ toward $I$.

So, for a momentum buffer matrix $B \in \mathbb{R}^{m \times n}$, algorithm for Newton-Schulz iteration are:

1. Frobenius normalization

$$ X_0 = \frac{B}{\lVert B \rVert_F + \epsilon} $$

Given $K$ steps, for iterations $k = 0, 1, \cdots, K-1$:

* Gram matrix: $A_k = X_k X_k^\top \quad (m \times m)$
* Polynomials: $A_k^2 = A_k A_k$
* finally, update: $X_{k+1} = aX_k + (bA_k + cA_k^2)X_k$

and, finally output: $O = X_K$.

**Why this shapes singular values?**

If $X_k = U\Sigma V^\top$, then:
$$ A_k = X_k X_k^\top = U \Sigma^2 U^\top \implies A_k^2 = U\Sigma^4U^\top$$

$$ \implies (bA_k + cA_k^2)X_k = U(b\Sigma^2 + c\Sigma^4)U^\top \cdot U\Sigma V^\top = U(b\Sigma^3 + c\Sigma^5)V^\top$$

and $aX_k = U(a\Sigma)V^\top$

Which shapes our expression of $X_{k+1} = aX_k + (bA_k + cA_k^2)X_k$ to be
$$X_{k+1} = U(a\Sigma + b\Sigma^3 + c\Sigma^5)V^\top$$

Here, each iteration applies the scalar polynomial $\varphi(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ to each singular value.

The coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ are chosen so that $\varphi(\sigma)$ pulls singular values toward 1 in the stable region after the Frobenius normalization.

## Muon: The Complete Algorithm

Now, with complete understanding of Newton-Schulz usage as an approximate replacement to computationally expensive SVD, we can complete the Muon Optimization algorithm with a detailed look inside the *orthogonalization* mechanism as well.

Let $W_t \in \mathbb{R}^{m \times n}$ be a matrix parameter and $G_t = \nabla_W \mathcal{L}(W_t)$.

We define momentum buffer $B_t$ and hyperparameters:

* learning rate $\rightarrow \gamma$
* momentum $\rightarrow \mu$
* weight decay $\rightarrow \lambda$
* coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$

We initialize $B_0 = 0$

For each step t:

1. **Momentum accumulation (restore rank)**:
   $$ B_t = \mu B_{t-1} + G_t$$

2. **Orthogonalize via Newton-Schulz (approximate polar factor)**:

   * Frobenius normalize $X_0 = \frac{B_t}{|B_t|_F + \epsilon}$

   * Iterate $K$ times:
     $$ A_k = X_k X_k^\top \implies X_{k+1} = aX_k + (bA_k + cA_k^2)X_k $$

   * Orthogonalized matrix output $O_t = X_K$

   $[$ If $ m > n$, we generally work with $X_0^\top$ and transpose back at the end. $]$

3. **Update RMS / LR calibration**:

   A common modern choice (Moonshot-style) is to match AdamW-like update RMS by scaling the orthogonalized update by

   $$ O_t \leftarrow 0.2, O_t,\sqrt{\max(m,n)} $$

   (equivalently: keep $O_t$ fixed and scale the effective learning rate by $0.2\sqrt{\max(m,n)}$).

   Other implementations use a different shape-based adjustment rule, but the core goal is the same: keep update RMS consistent across matrix shapes.

4. **Update with decoupled weight decay**:
   $$ W_{t+1} = W_t - \gamma,(O_t + \lambda W_t) $$

Now we recall from  $(✧)$ that in a pure constrained-optimization derivation, the optimal solution includes
$$\Delta W^* = -\eta \sqrt{\frac{\text{fan-out}}{\text{fan-in}}}, UV^\top$$

In code, this “shape scaling” is typically implemented either:

* Explicitly (a shape-dependent multiplier on the update), or
* Implicitly via the learning-rate adjustment / update-RMS calibration.

Different implementations expose this as an `adjust_lr_fn` / scaling mode knob (e.g., “original” vs “match_rms_adamw”).

## Practical caveat at scale: MuonClip / QK-Clip (Moonshot)

While Muon’s core update is derived cleanly from the operator-norm constrained problem, large-scale LLM training surfaced a specific failure mode: **exploding attention logits** (max pre-softmax scores can shoot to 1e3+ early), which correlates with uncontrolled growth of the query/key projection operators.

Moonshot’s Kimi K2 work addresses this by wrapping Muon into **MuonClip**, i.e. Muon + weight decay + consistent update-RMS matching + a targeted clipping mechanism called **QK-Clip**.

The key idea is not to clamp logits inside the forward pass (which distorts the attention distribution), but to **rescale the query/key projection weights *after* the optimizer step** whenever an already-computed per-head max-logit statistic exceeds a threshold. This keeps training stable without altering the forward/backward computation of the current step, and the clipping can be applied only to the heads that actually exhibit the runaway behavior.

## A wrap up

Now, with all derivations at our hand and a neat algorithm for Muon's optimization, we look at the overall picture once again with respect to the common optimization methods used and how Muon differs in its utility.

Consider, how a 2D matrix (say $M$) transforms the unit circle:

* Maps the unit circle to an ellipse
* Major axis along the dominant singular direction
* Aspect ratio = condition number $\kappa$

However, the **polar factor** $U V^\top$:

* Maps the unit circle to a *circle*
* All directions are treated equally
* condition number $\kappa = 1$

Gradient matrices are often highly ill-conditioned with a few directions with a relatively large gradient magnitude.

**Adam**'s responds to this by scaling each coordinate indepently to help with coordinate-wise variance, but this still doesn't fix the *directional imbalance*.

**Muon**'s responds by *orthogonalizing* the (momentum) update. This sets singular values near 1, making the update well-conditioned.

We can now complete our understanding of three fundamentally different optimizers:

| Optimizer | Geometry            | Constraint                                       | Update Form                                   |
| --------- | ------------------- | ------------------------------------------------ | --------------------------------------------- |
| SGD       | Euclidean           | $\lVert \Delta\theta \rVert_2 \leq \eta$                    | $-\alpha g$                                   |
| Adam      | Diagonal            | $\sum_i d_i,\Delta\theta_i^2 \leq \eta^2$        | $-\alpha \cdot \mathrm{diag}(d)^{-1} \cdot g$ |
| **Muon**  | Operator (spectral) | $\lVert \Delta W \rVert_{\text{RMS}\to\text{RMS}} \leq \eta$ | $-\alpha \cdot UV^\top$                       |

**The unifying view** is that each optimizer is still steepest descent under a different notion of distance as we discussed under **constrained linearized improvement**. Muon's notion of operator norm matches closely to what linear layers actually do.

## References

- Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. "Muon: An optimizer for hidden layers in neural networks." Blog post (2024). [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)
- Jeremy Bernstein. "Deriving Muon." Blog post (Mar 7, 2025). [https://jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon)
- Jingyuan Liu et al. "Muon is Scalable for LLM Training." arXiv:2502.16982 (2025). (Practical considerations at scale) [https://doi.org/10.48550/arXiv.2502.16982](https://doi.org/10.48550/arXiv.2502.16982)
- Kimi Team et al. "Kimi K2: Open Agentic Intelligence." arXiv:2507.20534 (2025). (MuonClip / QK-Clip) [https://doi.org/10.48550/arXiv.2507.20534](https://doi.org/10.48550/arXiv.2507.20534)
- Diederik P. Kingma and Jimmy Ba. "Adam: A Method for Stochastic Optimization." arXiv:1412.6980 (2014). [https://doi.org/10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
- Herbert Robbins and Sutton Monro. "A Stochastic Approximation Method." The Annals of Mathematical Statistics 22(3):400-407 (1951). [https://doi.org/10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)


<style>
.ml-interactive {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  background: var(--entry);
  margin: 1rem 0 2rem;
}

.ml-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 18px;
  align-items: center;
  margin-bottom: 12px;
}

.ml-controls label {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  font-size: 0.95rem;
}

.ml-controls input[type="range"] {
  width: 180px;
}

.ml-readout {
  font-variant-numeric: tabular-nums;
  color: var(--secondary);
}

.ml-plot {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
  border: 1px solid var(--border);
}

.ml-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 8px;
  font-size: 0.9rem;
  color: var(--secondary);
}

.ml-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.ml-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}

.ml-button {
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--code-bg);
  color: var(--primary);
}

.ml-button:hover {
  background: var(--tertiary);
}
</style>

<script>
(function () {
  function makeScaler(width, height, pad, xMin, xMax, yMin, yMax) {
    function sx(x) {
      return pad + ((x - xMin) / (xMax - xMin)) * (width - 2 * pad);
    }
    function sy(y) {
      return height - pad - ((y - yMin) / (yMax - yMin)) * (height - 2 * pad);
    }
    return { sx: sx, sy: sy };
  }

  function drawGrid(grid, width, height, pad) {
    var lines = [];
    for (var i = 0; i <= 10; i += 1) {
      var gx = pad + ((width - 2 * pad) * i) / 10;
      var gy = pad + ((height - 2 * pad) * i) / 10;
      lines.push('<line x1="' + gx + '" y1="' + pad + '" x2="' + gx + '" y2="' + (height - pad) + '" stroke="#eee" />');
      lines.push('<line x1="' + pad + '" y1="' + gy + '" x2="' + (width - pad) + '" y2="' + gy + '" stroke="#eee" />');
    }
    grid.innerHTML = lines.join("");
  }

  function drawAxes(axes, width, height, pad, sx, sy) {
    var axis = [];
    axis.push('<line x1="' + pad + '" y1="' + sy(0) + '" x2="' + (width - pad) + '" y2="' + sy(0) + '" stroke="#333" />');
    axis.push('<line x1="' + sx(0) + '" y1="' + pad + '" x2="' + sx(0) + '" y2="' + (height - pad) + '" stroke="#333" />');
    axes.innerHTML = axis.join("");
  }

  function contourPath(a, b, sx, sy, levels) {
    var paths = [];
    for (var i = 0; i < levels.length; i += 1) {
      var c = levels[i];
      var rx = Math.sqrt(c / a);
      var ry = Math.sqrt(c / b);
      if (!isFinite(rx) || !isFinite(ry)) continue;
      var steps = 120;
      var d = "";
      for (var t = 0; t <= steps; t += 1) {
        var ang = (Math.PI * 2 * t) / steps;
        var x = rx * Math.cos(ang);
        var y = ry * Math.sin(ang);
        var px = sx(x);
        var py = sy(y);
        d += (t === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
      }
      paths.push('<path d="' + d.trim() + '" fill="none" stroke="#ced4da" stroke-width="1.2" />');
    }
    return paths.join("");
  }

  function pathFromPoints(points, sx, sy) {
    var d = "";
    for (var i = 0; i < points.length; i += 1) {
      var px = sx(points[i].x);
      var py = sy(points[i].y);
      d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
    }
    return d.trim();
  }

  function computeSGDPath(x0, y0, a, b, lr, steps) {
    var pts = [{ x: x0, y: y0 }];
    var x = x0;
    var y = y0;
    for (var i = 0; i < steps; i += 1) {
      var gx = a * x;
      var gy = b * y;
      x -= lr * gx;
      y -= lr * gy;
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  function computeScaledPath(x0, y0, a, b, lr, steps) {
    var pts = [{ x: x0, y: y0 }];
    var x = x0;
    var y = y0;
    for (var i = 0; i < steps; i += 1) {
      var gx = a * x;
      var gy = b * y;
      x -= lr * (gx / a);
      y -= lr * (gy / b);
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  function computeMomentumPath(x0, y0, a, b, lr, steps, beta) {
    var pts = [{ x: x0, y: y0 }];
    var x = x0;
    var y = y0;
    var vx = 0;
    var vy = 0;
    for (var i = 0; i < steps; i += 1) {
      var gx = a * x;
      var gy = b * y;
      vx = beta * vx + (1 - beta) * gx;
      vy = beta * vy + (1 - beta) * gy;
      x -= lr * vx;
      y -= lr * vy;
      pts.push({ x: x, y: y });
    }
    return pts;
  }

  function setupSgdBasic(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var contours = root.querySelector("[data-role='contours']");
    var path = root.querySelector("[data-role='path']");
    var startDot = root.querySelector("[data-role='start']");
    var endDot = root.querySelector("[data-role='end']");
    var lrInput = root.querySelector("[data-role='lr']");
    var lrVal = root.querySelector("[data-role='lr-val']");
    var stepsInput = root.querySelector("[data-role='steps']");
    var stepsVal = root.querySelector("[data-role='steps-val']");
    var x0Input = root.querySelector("[data-role='x0']");
    var x0Val = root.querySelector("[data-role='x0-val']");
    var y0Input = root.querySelector("[data-role='y0']");
    var y0Val = root.querySelector("[data-role='y0-val']");
    var resetBtn = root.querySelector("[data-role='reset']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -4;
    var xMax = 4;
    var yMin = -3.2;
    var yMax = 3.2;
    var scale = makeScaler(width, height, pad, xMin, xMax, yMin, yMax);

    var a = 1.0;
    var b = 4.0;

    function render() {
      var lr = parseFloat(lrInput.value);
      var steps = parseInt(stepsInput.value, 10);
      var x0 = parseFloat(x0Input.value);
      var y0 = parseFloat(y0Input.value);

      lrVal.textContent = lr.toFixed(2);
      stepsVal.textContent = String(steps);
      x0Val.textContent = x0.toFixed(1);
      y0Val.textContent = y0.toFixed(1);

      var pts = computeSGDPath(x0, y0, a, b, lr, steps);
      path.setAttribute("d", pathFromPoints(pts, scale.sx, scale.sy));
      startDot.setAttribute("cx", scale.sx(pts[0].x));
      startDot.setAttribute("cy", scale.sy(pts[0].y));
      var last = pts[pts.length - 1];
      endDot.setAttribute("cx", scale.sx(last.x));
      endDot.setAttribute("cy", scale.sy(last.y));

      contours.innerHTML = contourPath(a, b, scale.sx, scale.sy, [0.5, 1, 2, 3, 4, 5]);
    }

    resetBtn.addEventListener("click", function () {
      x0Input.value = "2.5";
      y0Input.value = "2.0";
      render();
    });

    drawGrid(grid, width, height, pad);
    drawAxes(axes, width, height, pad, scale.sx, scale.sy);
    render();

    lrInput.addEventListener("input", render);
    stepsInput.addEventListener("input", render);
    x0Input.addEventListener("input", render);
    y0Input.addEventListener("input", render);
  }

  function setupScaling(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var contours = root.querySelector("[data-role='contours']");
    var pathSgd = root.querySelector("[data-role='path-sgd']");
    var pathPre = root.querySelector("[data-role='path-pre']");
    var ratioInput = root.querySelector("[data-role='ratio']");
    var ratioVal = root.querySelector("[data-role='ratio-val']");
    var lrInput = root.querySelector("[data-role='lr']");
    var lrVal = root.querySelector("[data-role='lr-val']");
    var stepsInput = root.querySelector("[data-role='steps']");
    var stepsVal = root.querySelector("[data-role='steps-val']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -4;
    var xMax = 4;
    var yMin = -3.2;
    var yMax = 3.2;
    var scale = makeScaler(width, height, pad, xMin, xMax, yMin, yMax);

    function render() {
      var ratio = parseFloat(ratioInput.value);
      var lr = parseFloat(lrInput.value);
      var steps = parseInt(stepsInput.value, 10);

      ratioVal.textContent = ratio.toFixed(0);
      lrVal.textContent = lr.toFixed(2);
      stepsVal.textContent = String(steps);

      var a = ratio;
      var b = 1.0;
      var x0 = 2.5;
      var y0 = 2.0;

      var ptsSgd = computeSGDPath(x0, y0, a, b, lr, steps);
      var ptsPre = computeScaledPath(x0, y0, a, b, lr, steps);

      pathSgd.setAttribute("d", pathFromPoints(ptsSgd, scale.sx, scale.sy));
      pathPre.setAttribute("d", pathFromPoints(ptsPre, scale.sx, scale.sy));

      contours.innerHTML = contourPath(a, b, scale.sx, scale.sy, [0.5, 1, 2, 3, 4, 6, 8]);
    }

    drawGrid(grid, width, height, pad);
    drawAxes(axes, width, height, pad, scale.sx, scale.sy);
    render();

    ratioInput.addEventListener("input", render);
    lrInput.addEventListener("input", render);
    stepsInput.addEventListener("input", render);
  }

  function setupMomentum(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var contours = root.querySelector("[data-role='contours']");
    var pathSgd = root.querySelector("[data-role='path-sgd']");
    var pathMom = root.querySelector("[data-role='path-mom']");
    var betaInput = root.querySelector("[data-role='beta']");
    var betaVal = root.querySelector("[data-role='beta-val']");
    var lrInput = root.querySelector("[data-role='lr']");
    var lrVal = root.querySelector("[data-role='lr-val']");
    var stepsInput = root.querySelector("[data-role='steps']");
    var stepsVal = root.querySelector("[data-role='steps-val']");

    var width = 640;
    var height = 360;
    var pad = 45;
    var xMin = -4;
    var xMax = 4;
    var yMin = -3.2;
    var yMax = 3.2;
    var scale = makeScaler(width, height, pad, xMin, xMax, yMin, yMax);

    function render() {
      var beta = parseFloat(betaInput.value);
      var lr = parseFloat(lrInput.value);
      var steps = parseInt(stepsInput.value, 10);

      betaVal.textContent = beta.toFixed(2);
      lrVal.textContent = lr.toFixed(2);
      stepsVal.textContent = String(steps);

      var a = 20.0;
      var b = 1.0;
      var x0 = 2.5;
      var y0 = 2.0;

      var ptsSgd = computeSGDPath(x0, y0, a, b, lr, steps);
      var ptsMom = computeMomentumPath(x0, y0, a, b, lr, steps, beta);

      pathSgd.setAttribute("d", pathFromPoints(ptsSgd, scale.sx, scale.sy));
      pathMom.setAttribute("d", pathFromPoints(ptsMom, scale.sx, scale.sy));

      contours.innerHTML = contourPath(a, b, scale.sx, scale.sy, [0.5, 1, 2, 3, 4, 6, 8]);
    }

    drawGrid(grid, width, height, pad);
    drawAxes(axes, width, height, pad, scale.sx, scale.sy);
    render();

    betaInput.addEventListener("input", render);
    lrInput.addEventListener("input", render);
    stepsInput.addEventListener("input", render);
  }

  function mulMatVec(M, v) {
    return {
      x: M[0][0] * v.x + M[0][1] * v.y,
      y: M[1][0] * v.x + M[1][1] * v.y
    };
  }

  function invSqrt2x2(p, q, r) {
    var trace = p + r;
    var det = p * r - q * q;
    var disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));
    var l1 = 0.5 * (trace + disc);
    var l2 = 0.5 * (trace - disc);

    l1 = Math.max(l1, 1e-12);
    l2 = Math.max(l2, 1e-12);

    var v1x = q;
    var v1y = l1 - p;
    if (Math.abs(v1x) + Math.abs(v1y) < 1e-8) {
      v1x = 1;
      v1y = 0;
    }
    var n1 = Math.sqrt(v1x * v1x + v1y * v1y);
    v1x /= n1;
    v1y /= n1;
    var v2x = -v1y;
    var v2y = v1x;

    var inv1 = 1 / Math.sqrt(l1);
    var inv2 = 1 / Math.sqrt(l2);

    var m00 = inv1 * v1x * v1x + inv2 * v2x * v2x;
    var m01 = inv1 * v1x * v1y + inv2 * v2x * v2y;
    var m11 = inv1 * v1y * v1y + inv2 * v2y * v2y;

    return [
      [m00, m01],
      [m01, m11]
    ];
  }

  function computeSigmaMax(a, b, c, d) {
    var s1 = a * a + b * b + c * c + d * d;
    var det = (a * d - b * c);
    var disc = Math.sqrt(Math.max(0, s1 * s1 - 4 * det * det));
    var sigma2 = 0.5 * (s1 + disc);
    return Math.sqrt(Math.max(0, sigma2));
  }

  function setupOperatorNorm(root) {
    var grid = root.querySelector("[data-role='grid']");
    var axes = root.querySelector("[data-role='axes']");
    var unitPath = root.querySelector("[data-role='unit']");
    var mappedPath = root.querySelector("[data-role='mapped']");
    var polarPath = root.querySelector("[data-role='polar']");
    var rmsPath = root.querySelector("[data-role='rms']");
    var sigmaEl = root.querySelector("[data-role='sigma']");
    var rmsEl = root.querySelector("[data-role='rms']");

    var inputs = {
      a: root.querySelector("[data-role='a']"),
      b: root.querySelector("[data-role='b']"),
      c: root.querySelector("[data-role='c']"),
      d: root.querySelector("[data-role='d']"),
      n: root.querySelector("[data-role='n']"),
      m: root.querySelector("[data-role='m']")
    };
    var vals = {
      a: root.querySelector("[data-role='a-val']"),
      b: root.querySelector("[data-role='b-val']"),
      c: root.querySelector("[data-role='c-val']"),
      d: root.querySelector("[data-role='d-val']"),
      n: root.querySelector("[data-role='n-val']"),
      m: root.querySelector("[data-role='m-val']")
    };

    var width = 640;
    var height = 360;
    var pad = 45;
    var baseX = 3.5;
    var baseY = 2.6;
    var scale = makeScaler(width, height, pad, -baseX, baseX, -baseY, baseY);

    function circlePath(transform, colorScale) {
      var steps = 140;
      var d = "";
      for (var i = 0; i <= steps; i += 1) {
        var ang = (Math.PI * 2 * i) / steps;
        var v = { x: Math.cos(ang), y: Math.sin(ang) };
        if (transform) {
          v = transform(v);
        }
        var px = scale.sx(v.x * colorScale);
        var py = scale.sy(v.y * colorScale);
        d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
      }
      return d.trim();
    }

    function render() {
      var a = parseFloat(inputs.a.value);
      var b = parseFloat(inputs.b.value);
      var c = parseFloat(inputs.c.value);
      var d = parseFloat(inputs.d.value);
      var n = parseInt(inputs.n.value, 10);
      var m = parseInt(inputs.m.value, 10);

      vals.a.textContent = a.toFixed(2);
      vals.b.textContent = b.toFixed(2);
      vals.c.textContent = c.toFixed(2);
      vals.d.textContent = d.toFixed(2);
      vals.n.textContent = String(n);
      vals.m.textContent = String(m);

      var sigma = computeSigmaMax(a, b, c, d);
      var rms = sigma * Math.sqrt(n / m);
      sigmaEl.textContent = sigma.toFixed(3);
      rmsEl.textContent = rms.toFixed(3);

      var maxRadius = Math.max(baseX, sigma * 1.15, rms * 1.15);
      var yRadius = Math.max(baseY, maxRadius * (baseY / baseX));
      scale = makeScaler(width, height, pad, -maxRadius, maxRadius, -yRadius, yRadius);
      drawGrid(grid, width, height, pad);
      drawAxes(axes, width, height, pad, scale.sx, scale.sy);

      var A = [
        [a, b],
        [c, d]
      ];
      var p = a * a + c * c;
      var q = a * b + c * d;
      var r = b * b + d * d;
      var invSqrt = invSqrt2x2(p, q, r);
      var polar = [
        [A[0][0] * invSqrt[0][0] + A[0][1] * invSqrt[1][0], A[0][0] * invSqrt[0][1] + A[0][1] * invSqrt[1][1]],
        [A[1][0] * invSqrt[0][0] + A[1][1] * invSqrt[1][0], A[1][0] * invSqrt[0][1] + A[1][1] * invSqrt[1][1]]
      ];

      unitPath.setAttribute("d", circlePath(null, 1.0));
      mappedPath.setAttribute("d", circlePath(function (v) { return mulMatVec(A, v); }, 1.0));
      polarPath.setAttribute("d", circlePath(function (v) { return mulMatVec(polar, v); }, 1.0));
      rmsPath.setAttribute("d", circlePath(null, rms));
    }

    render();

    Object.keys(inputs).forEach(function (key) {
      inputs[key].addEventListener("input", render);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var sgd = document.querySelector("[data-ml='sgd-basic']");
    if (sgd) setupSgdBasic(sgd);

    var scaling = document.querySelector("[data-ml='sgd-scaling']");
    if (scaling) setupScaling(scaling);

    var momentum = document.querySelector("[data-ml='sgd-momentum']");
    if (momentum) setupMomentum(momentum);

    var opnorm = document.querySelector("[data-ml='operator-norm']");
    if (opnorm) setupOperatorNorm(opnorm);
  });
})();
</script>