# Why Denoising (and Sometimes Diffusion) for Copula Densities?

## Summary

Copula densities live on a constrained set (non-negative, unit mass, uniform marginals). This repository uses denoising-style training plus IPFP/Sinkhorn projection to produce valid copula densities from empirical bivariate data.

The codebase supports **two inference variants**:
- **One-shot**: a single forward pass from a histogram (fast).
- **Iterative diffusion**: DDIM sampling in log-density space, optionally with classifier-free guidance (CFG) on histogram conditioning (more compute, potentially higher fidelity).

---

## The Challenge: Copulas Are Not Images

### Copula Constraints

A valid copula density \( c(u,v) \) must satisfy:
\[
\int_0^1 c(u,v) \, dv = 1 \quad \forall u \in [0,1]
\]
\[
\int_0^1 c(u,v) \, du = 1 \quad \forall v \in [0,1]
\]

**These are HARD constraints.** Even small violations make the result invalid.

### Why This Is Hard for Standard NNs

**Problem 1: Gradient Conflict**
- Density loss: "Match the shape"
- Constraint loss: "Make marginals uniform"
- These compete → unstable training

**Problem 2: Projection Breaks Gradients**
- Need IPFP to enforce constraints
- But projection is non-differentiable (iterative algorithm)
- Can't backprop through it effectively

**Problem 3: Extreme Dynamic Range**
- Copula densities range from \( 10^{-9} \) to \( 10^6 \)
- Especially at boundaries \( u,v \in \{0,1\} \)
- Standard activations (ReLU, sigmoid) struggle

---

## How Diffusion Solves These

### 1. Iterative Refinement (optional)

```
Step 0:   Noisy density (random)
Step 500: Rough shape emerges, marginals ~uniform
Step 999: Fine details, constraints nearly perfect
```

**Advantage**:
- Progressive constraint satisfaction
- Early steps learn global structure
- Late steps refine details
- Natural curriculum built-in

### 2. Projection as Part of Generation

In diffusion:
```python
for t in reversed(range(1000)):
    x_{t-1} = denoise(x_t)
    x_{t-1} = project_to_copula(x_{t-1})  # Part of the process!
```

**Advantage**:
- Projection is EXPECTED in generation
- Model learns to output "projectable" densities
- Gradients flow through denoising, not projection

### 3. Log-Space Parameterization

Diffusion operates on **log-density**:
- \( \log c(u,v) \in (-\infty, \infty) \)
- No constraints on range
- Handles extreme values naturally

**Advantage**:
- \( 10^{-9} \to 10^6 \) becomes \( -20 \to 14 \) in log-space
- Much easier for neural networks

### 4. Multi-Scale by Design

U-Net architecture + 1000 denoising steps:
- Learns features at ALL scales simultaneously
- Down-sampling path: Global correlation structure
- Up-sampling path: Local tail dependencies
- Skip connections: Preserve fine details

**Advantage over CNN**:
- CNN sees one receptive field size
- Diffusion sees from pixel-level to image-level (naturally multi-scale)

---

## Empirical Validation
This repository is set up to evaluate the one-shot and diffusion variants under a fixed benchmark suite (bivariate metrics and vine-level PIT/NLL). Any claims about accuracy and runtime should be supported by those results.

---

## Key Insight (and what we do \*not\* claim)

Diffusion can be used as a **stochastic generator** (sample multiple plausible densities by changing the diffusion randomness), but this project’s primary goal is **accurate amortized density estimation** for vines. We therefore evaluate diffusion mainly as an \emph{estimator} (one-shot vs iterative refinement), and treat uncertainty-style benefits as optional.

---

## Recommended Experiments for Paper

1. **Ablation: Diffusion vs. CNN**
   - Same architecture (UNet)
   - Different inference: one-shot (single forward) vs. iterative DDIM (S steps)
   - Metric: ISE, marginal error, tail dependence

2. **Ablation: Number of Timesteps**
   - T = 1, 10, 100, 500, 1000
   - Show: more steps → better fits

3. **Comparison: vs. Kernel Density Estimation**
   - KDE with boundary correction
   - Diffusion
   - Metric: MSE, marginal uniformity

4. **Application: Real Data**
   - Financial returns, weather data, etc.
   - Show: learned copulas enable better risk estimation

---

## Conclusion

In this project, diffusion is primarily used as a **denoising mechanism** for copula density estimation under constraints. The recommended workflow is to keep both one-shot and diffusion variants, compare them with a consistent evaluation protocol, and choose the default based on measured accuracy and runtime.
