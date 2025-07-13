---
title: "A deep-dive into RoPE, and why it matters?"
date: 2025-07-13
draft: false
math: true
toc: true
---
## What is Positional Encoding and why it matters?

When training any large language model based on **Transformers** architecture, our input token sequences tend to form a $\text{seq\\_len} \times \text{seq\\_len}$ dimension **Attention** network where, the positional information between tokens aren't preserved natively, it's simply the representation of **attention** scores between each token (normalized by the $\sqrt{\text{dim\\_len}}$).

In order to preserve positional information such that the network learns differently about *"Dog attacks the Cat"* and *"Cat attacks the Dog"*, we add a **deterministic** (remains the same throughout the network) encoding for each position across the dimensional embedding at each position.

The positional encoding recommended in [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) was **Sinusioidal Positional Encoding** represented as:
$$
PE_{pos, 2i} = \sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

$$
PE_{pos, 2i+1} = \cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

where we apply pair-wise *sine* and *cosine* embedding to each of the consecutive embedding dimensions pair.

The term $\frac{2i}{d_{model}}$ forms an arithmetic progression from $0$ to $\left( \frac{d}{2} - 1 \right)$ where $d_{model} \rightarrow$ embedding dimension of the model.

$$
\text{positional embedding} \rightarrow \vec{p_t}: 
\begin{bmatrix}
    \sin(w_1.t) \\\\
    \cos(w_1.t) \\\\
    \sin(w_2.t) \\\\
    \cos(w_2.t) \\\\
    \dots \\\\
    \sin(w_{\frac{d}{2}}.t) \\\\
    \cos(w_{\frac{d}{2}}.t)
\end{bmatrix}_{d \times 1}
$$

Compared to using Binary Position Encoding and then utilizing the cycle of least significant bit (LSB) and most significant bit (MSB) [[1](https://huggingface.co/blog/designing-positional-encoding#binary-position-encoding)], which switches abruptly between positional encoding values at different dimensional pairs, **Sinusoidal Positional Embedding** maintains a smooth flow between the positional encoding values for different dimensional pairs.

## Why Rotary Positional Embedding (RoPE)?

There are a few limitations even to *Sinusoidal Positional Embedding* that we wish to mitigate away.

1. Sinusoidal Positional Encoding allows relative position inference implicitly, but doesn't preserve relative distance in dot-product space, making it suboptimal for models relying on attention similarity. [[2](https://arxiv.org/abs/1803.02155)]
2. The similarity (dot-product) between two token embeddings varies with absolute positions, even if their relative distance stays the same.

**RoPE** helps us mitigate through both of these limitations.

Consider, the example from above, *"Dog attacks the Cat"*, in order to reliably encode the relative position between tokens, we need a positional embedding method such that the dot product to the applied embeddings for both *"Dog"* and *"Cat"* remains the same in both the examples, *"Dog attacks the Cat"* and *"Once upon a time, a Dog attacks the Cat"*, i.e., the dot product between two embeddings for tokens $x_m$ and $x_n$ depends only on their relative position to each other $\Delta = (m - n)$.

RoPE encodes relative positional information in the *attention dot product* between entire query and key vectors, even though the operation is defined per 2D pair of dimensions.

For a **d-dimensional embedding** (say d = 768) **RoPE** partitions the input token vector $\vec{x}$ into $\frac{d}{2}$ disjoint 2D subspaces, and applies a position dependent relation to each pair. <br>

$$
R_{\theta_{m,i}} =
\left(
\begin{array}{cccc}
\mathbf{R_{\theta_0}} & & & \\\\
& \mathbf{R_{\theta_1}} & & \\\\
& & \ddots & \\\\
& & & \mathbf{R_{\theta_{d/2}}}
\end{array}
\right)
\left(
\begin{array}{c}
x_1 \\\\
x_2 \\\\
\vdots \\\\
x_d
\end{array}
\right)
$$

where, each $R_{\theta_i}$ refers to a rotation matrix

$$
\begin{pmatrix}
\cos(\theta_i) & -\sin(\theta_i) \\\\
\sin(\theta_i) &  \cos(\theta_i)
\end{pmatrix}
$$

for any given specific token position $m$.

Each 2D operation operates independently, rotating a 2D subvector using a fixed frequency $\theta_i$.

For any attention network:
- Query vector: $q = W_q \cdot x$
- Key vector: $k = W_k \cdot x$
- Value vector: $v = W_v \cdot x$

these are task-specific learned transformations, which adapt raw token embeddings to more useful subspaces for querying and reasoning.

We, then apply RoPE to these learned transformations,
$$
R(m)q = R(m) \cdot (W_q \cdot x_m)
$$
$$
R(n)k  = R(m) \cdot (W_k \cdot x_n)
$$

the attention mechanism computes,

$$
attn_{m,n} = q_m^T k_n = \left\\{R(m) W_q x_m \right\\}^T \left\\{ R(n) W_k x_n \right\\}
$$

where, **RoPE** ensures that $R(m)^T R(n) = R(n - m)$, i.e., the inner product depends only on $\Delta = (n - m)$ due to the properties of rotation matrices.

Hence, $\langle R(m)q, R(n)k \rangle = \langle R(n-m)q,k \rangle$

In practice, we don't use a matrix multiplication to compute RoPE to avoid computational inefficiency due to sparsity present in the matrix. Instead, we directly apply the rotations to pairs of elements independently, taking advantage of the regular pattern in the computation:

$$
R_{\boldsymbol{\Theta}, p}^d \mathbf{q} =
\begin{pmatrix}
q_1 \\\\
q_2 \\\\
q_3 \\\\
q_4 \\\\
\vdots \\\\
q_{d-1} \\\\
q_d
\end{pmatrix}
\otimes
\begin{pmatrix}
\cos(p\theta_1) \\\\
\cos(p\theta_1) \\\\
\cos(p\theta_2) \\\\
\cos(p\theta_2) \\\\
\vdots \\\\
\cos(p\theta_{d/2}) \\\\
\cos(p\theta_{d/2})
\end{pmatrix}
+
\begin{pmatrix}
-q_2 \\\\
q_1 \\\\
-q_4 \\\\
q_3 \\\\
\vdots \\\\
-q_d \\\\
q_{d-1}
\end{pmatrix}
\otimes
\begin{pmatrix}
\sin(p\theta_1) \\\\
\sin(p\theta_1) \\\\
\sin(p\theta_2) \\\\
\sin(p\theta_2) \\\\
\vdots \\\\
\sin(p\theta_{d/2}) \\\\
\sin(p\theta_{d/2})
\end{pmatrix}
$$

### PyTorch implementation

```python
import torch

def apply_rope(token):
    """
    Apply RoPE to a token tensor.
    token dimension: (batch_size, seq_len, d_model)
    """
    dim = token.shape[-1]
    assert dim % 2 == 0, "RoPE requires even number of dimensions"

    half = dim // 2
    freq = 1 / (10000 ** (torch.arange(0, half, device = token.device, dtype = token.dtype) / half))

    position = torch.arange(token.size(1), device=token.device, dtype=token.dtype)

    theta = position.unsqueeze(-1) * freq

    sin = torch.sin(theta)
    cos = torch.cos(theta)

    # split the token into even and odd indices
    t_even = token[..., ::2]
    t_odd = token[..., 1::2]

    # Apply rotation (element-wise)
    rot_even = t_even * cos - t_odd * sin
    rot_odd = t_even * sin + t_odd * cos

    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2,-1)
```


### Intuition behind RoPE:

1. Each token's vector (after linear projection) is sliced into pairs of dimensions: $[x_0, x_1], [x_2, x_3], \dots$
2. Each pair is rotated deterministically by a position-dependent angle $\theta_{m,i}$ where, $\theta_{m,i}$ is frequency-dependent for each token's position $m$.
3. The rotation is mathematically equivalent to placing the pair on the unit circle and rotating it counter clockwise.
$$
\begin{bmatrix}
\cos(\theta_{m,i}) & -\sin(\theta_{m,i}) \\\\
\sin(\theta_{m,i}) &  \cos(\theta_{m,i})
\end{bmatrix}
\cdot
\begin{bmatrix}
x \\\\
y 
\end{bmatrix}
$$

For each plane *i* $(0 ≤ i ≤ \frac{d}{2})$ the rotation angle for a token at position *m* is:
$$
\theta_{m,i} = m \cdot \frac{1}{10000^{\frac{2i}{d}}}
$$
- 10000 is a base **b** (derived originally through multiple experimentations)
- **Exponent decay:** the denominator grows exponentially with *i*, so large *i* -> tiny angle increments.
- **Linear growth**: the numerator grows linearly with *m*,  so further tokens always rotate more.

Now, if we consider each 2-D plane as a little clock hand: <br>
To complete a full $2\pi$ rotation, we need:
$$
\theta_{m,i} = 2\pi \Rightarrow m\frac{1}{10000^{\frac{2i}{d}}} = 2\pi
$$
$$
m_i^{period} = 2\pi \cdot 10000^{\frac{2i}{d}}
$$

| plane index *i* | period-length (how many tokens for 2π)             |
| --------------- | -------------------------------------------------- |
| 0               | 2π·10000⁰ = **≈ 6 tokens**                         |
| *d/2*−1         | 2π·10000^(1−2/d) ≈ **tens-of-thousands of tokens** |

#### Short-Range Structure
- **High-frequency planes** (small *i*) $ \rightarrow$ changes phase quickly with small changes in m.
- Attention scores uses dot-products of rotated vectors; small positional shifts produce large cosine variations.
- Therefore, neighboring tokens produces very different activations in these planes $ \rightarrow $ the model can resolve fine-grained local order.

#### Long-Range Structure
- **Low-frequency planes** (large *i*) $ \rightarrow$ changes phase extremely slowly.
- Tokens separated by large positions still have the almost same representation in these planes.
- Consequently, their dot-product (attention weight) is almost $\cos(0) = 1$, giving the model a **smooth, gradually decaying bias towards distant tokens**.

**RoPE**'s attention scores satisfy
$$
Attn(m, n) \propto \cos \left( \frac{\left|m-n \right|}{10000^{\frac{2i}{d}}} \right)
$$
so the influence of a token at distance |m-n| decays gracefully without any hand-tuned masking.

## References

1. Attention is All You Need: [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
2. Self-Attention with Relative Position Representations: [Shaw et al. [2018]](https://arxiv.org/abs/1803.02155)
3. RoFormer: [Su et al. [2021]](https://arxiv.org/abs/2104.09864)
4. [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
5. [Positional Embeddings in Transformer Models: Evolution from Text to Vision Domains](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-positional-embedding-19/blog/positional-embedding/)
