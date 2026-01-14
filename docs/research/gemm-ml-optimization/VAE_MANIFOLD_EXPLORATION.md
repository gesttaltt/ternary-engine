# VAE-Based Exploration of GEMM Solution Manifolds

**Doc-Type:** Research Proposal · Version 1.0 · Updated 2025-12-29

A generative approach to discovering the full family of valid matrix multiplication decompositions through emergent structure, not explicit computation.

---

## The Core Idea

**Traditional approach**: Compute/enumerate the manifold, then learn it.
**Proposed approach**: Let the manifold *emerge* from the invariant constraint.

Instead of training a VAE to recall a pre-designed manifold, we train it to satisfy a **correctness predicate**. The manifold of valid solutions emerges as the set of points where the predicate holds.

```
Traditional:  Data → VAE → Approximate Data Distribution
Proposed:     Constraint → VAE → Emergent Solution Manifold
```

---

## The Bilinear Invariant as Loss

### The Correctness Predicate

A tensor decomposition D = {(u₁,v₁,w₁), ..., (uᵣ,vᵣ,wᵣ)} is valid iff:

```python
def is_valid_matmul_decomposition(D, n):
    """
    Check if decomposition D correctly computes n×n matrix multiplication.

    The bilinear invariant: For ALL input matrices A, B:
        C[i,j] = Σₖ A[i,k] × B[k,j]  must equal
        C[i,j] = Σₜ (uₜ·vec(A)) × (vₜ·vec(B)) × wₜ[i,j]
    """
    T_reconstructed = sum(outer3(u, v, w) for u, v, w in D)
    T_true = matmul_tensor(n)  # The ground truth tensor
    return torch.allclose(T_reconstructed, T_true)
```

### The Key Insight

We don't need labeled data of "valid decompositions". We only need:
1. A way to generate candidate decompositions (the decoder)
2. A way to check correctness (the bilinear invariant)

The VAE learns to generate points that satisfy the invariant. The latent space organizes around *other* structure (sparsity, symmetry, rank distribution) because the invariant is already enforced.

---

## Architecture: Invariant-Constrained VAE

### Representation

A rank-r decomposition for n×n matmul:
```
D = {(u₁,v₁,w₁), ..., (uᵣ,vᵣ,wᵣ)}

where:
  uᵢ ∈ ℝⁿ² (or {-1,0,+1}ⁿ² for ternary)
  vᵢ ∈ ℝⁿ²
  wᵢ ∈ ℝⁿ²
```

Flatten to vector: x ∈ ℝ^(3 × r × n²)

### Encoder

```python
class DecompositionEncoder(nn.Module):
    """
    Encode a tensor decomposition into latent space.

    Key: Must be invariant to factor ordering (summation is commutative).
    Solution: Use set-based encoder (DeepSets / Transformer with no positional encoding).
    """
    def __init__(self, n, r, latent_dim):
        self.factor_embed = nn.Linear(3 * n * n, hidden_dim)
        self.set_encoder = SetTransformer(hidden_dim, latent_dim)  # Order-invariant

    def forward(self, D):
        # D: (batch, r, 3, n, n) - r factors, each is (u, v, w)
        factors = D.view(batch, r, -1)  # (batch, r, 3n²)
        embedded = self.factor_embed(factors)  # (batch, r, hidden)
        z_mean, z_logvar = self.set_encoder(embedded)  # Order-invariant encoding
        return z_mean, z_logvar
```

### Decoder

```python
class DecompositionDecoder(nn.Module):
    """
    Decode latent vector to tensor decomposition.

    Key: Output must be a SET of factors (order doesn't matter).
    """
    def __init__(self, latent_dim, n, r):
        self.latent_to_factors = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r * 3 * n * n)
        )

    def forward(self, z):
        flat = self.latent_to_factors(z)
        D = flat.view(-1, r, 3, n, n)  # (batch, r, 3, n, n)
        return D
```

### Loss Function: Emergence Through Constraint

```python
def vae_loss(D_input, D_reconstructed, z_mean, z_logvar, T_true):
    """
    The magic: Correctness loss encodes the bilinear invariant.
    The VAE learns the manifold of valid solutions.
    """

    # 1. Reconstruction loss (standard VAE)
    recon_loss = mse(D_input, D_reconstructed)

    # 2. KL divergence (regularization)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    # 3. CORRECTNESS LOSS - The Bilinear Invariant
    T_generated = decomposition_to_tensor(D_reconstructed)
    correctness_loss = mse(T_generated, T_true)

    # 4. Optional: Sparsity/structure regularization
    sparsity_loss = torch.mean(torch.abs(D_reconstructed))

    return recon_loss + β * kl_loss + λ * correctness_loss + γ * sparsity_loss
```

---

## The Emergence Mechanism

### Phase 1: Random Exploration

Initially, the decoder outputs random decompositions. Most are invalid (don't compute matmul correctly). The correctness loss is high.

### Phase 2: Constraint Satisfaction

The VAE learns to generate decompositions that satisfy the bilinear invariant. The correctness loss decreases. The latent space starts to organize.

### Phase 3: Manifold Discovery

Once correctness is achieved, the VAE has freedom in HOW to achieve it. The latent space organizes around secondary structure:
- One direction might correspond to sparsity
- Another to symmetry type
- Another to coefficient magnitudes

**The manifold emerges because all valid solutions share the invariant, but differ in other ways.**

### Phase 4: Interpolation and Extrapolation

Sampling from the latent space generates NEW valid decompositions. Interpolating between two known solutions traces a path through the solution manifold.

---

## For Ternary GEMM

### Discrete Constraint

```python
def ternary_quantize(D):
    """Force decomposition to ternary coefficients."""
    return torch.sign(D) * (torch.abs(D) > 0.5).float()

def ternary_loss(D):
    """Encourage ternary values."""
    # Distance to nearest {-1, 0, +1}
    return torch.mean(torch.min(
        torch.abs(D - 1),
        torch.min(torch.abs(D), torch.abs(D + 1))
    ))
```

### The Ternary Solution Manifold

For 4×4 ternary matmul with rank-47 decompositions:
- AlphaTensor found 14,236 non-equivalent solutions
- These form a discrete manifold embedded in continuous space
- VAE can learn to generate points near this discrete set

### Continuous Relaxation → Discrete Solutions

Train with soft ternary constraint, then snap to discrete:

```python
# During training: soft constraint
D_soft = decoder(z)
loss = correctness_loss(D_soft) + ternary_loss(D_soft)

# During inference: hard snap
D_hard = ternary_quantize(D_soft)
assert is_valid_matmul_decomposition(D_hard)  # Verify correctness preserved
```

---

## Connection to Your Ternary VAE Work

### The p-adic Hierarchy Analogy

In your ternary VAE work:
- VAE-B learns radial structure (valuation → radius mapping)
- Hierarchy emerges from the constraint (p-adic distance)
- The manifold organizes around mathematical structure

For GEMM VAE:
- VAE learns decomposition structure
- Validity emerges from the bilinear constraint
- The manifold organizes around algebraic invariants

### Shared Pattern: Constraint-Driven Emergence

```
Ternary VAE:   p-adic constraint → hierarchical manifold emerges
GEMM VAE:     bilinear constraint → decomposition manifold emerges
```

The principle is the same: **Don't design the manifold. Let it emerge from the invariant.**

---

## Implementation Sketch

### Training Data: Self-Generated

```python
def generate_training_data(n, r, num_samples):
    """
    We don't need pre-computed valid decompositions.
    Start with random, let correctness loss guide learning.

    Optionally: seed with known decompositions (Strassen, Laderman)
    to accelerate convergence.
    """
    # Random initialization
    data = torch.randn(num_samples, r, 3, n, n)

    # Optional: include known valid decompositions
    data[0] = strassen_decomposition()
    data[1] = laderman_decomposition()
    # ...

    return data
```

### Training Loop

```python
def train_gemm_vae(vae, n, epochs):
    T_true = matmul_tensor(n)  # Ground truth tensor

    for epoch in range(epochs):
        # Generate candidates (can be random or from replay buffer)
        D_input = generate_candidates(batch_size)

        # VAE forward pass
        z_mean, z_logvar = vae.encode(D_input)
        z = reparameterize(z_mean, z_logvar)
        D_reconstructed = vae.decode(z)

        # Loss with bilinear invariant
        loss = vae_loss(D_input, D_reconstructed, z_mean, z_logvar, T_true)

        # Track how many generated decompositions are actually valid
        T_gen = decomposition_to_tensor(D_reconstructed)
        validity_rate = (mse(T_gen, T_true, reduction='none').mean(dim=-1) < 1e-6).float().mean()

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Validity={validity_rate:.2%}")
```

### Manifold Exploration

```python
def explore_manifold(vae, num_samples=1000):
    """Sample from latent space to discover new valid decompositions."""
    z = torch.randn(num_samples, latent_dim)
    D_generated = vae.decode(z)

    # Check which are valid
    valid_decompositions = []
    for d in D_generated:
        if is_valid_matmul_decomposition(d):
            valid_decompositions.append(d)

    # Compute invariants to understand structure
    for d in valid_decompositions:
        print(f"Sparsity: {sparsity(d):.2f}, Symmetry: {symmetry_type(d)}")

    return valid_decompositions
```

---

## Expected Outcomes

### 1. Latent Space Interpretability

After training, different latent dimensions should correspond to:
- Sparsity level
- Symmetry type (cyclic, reflection, none)
- Coefficient distribution
- Hardware affinity (memory access patterns)

### 2. Novel Algorithm Discovery

Sampling from unexplored regions of latent space may yield:
- New decompositions not found by AlphaTensor
- Decompositions optimized for specific constraints (ternary, sparse)
- Interpolations between known algorithms

### 3. Understanding the Manifold

Visualizing the latent space reveals:
- How many disconnected components exist
- Which regions are dense vs sparse
- The "shape" of the solution manifold

---

## Research Questions

1. **Does the manifold have natural coordinates?**
   - Can we find a parameterization where movement corresponds to interpretable changes?

2. **Is the ternary manifold a subset or a different object?**
   - Does discretization create a disconnected set, or does it trace out a sub-manifold?

3. **Can we learn the manifold dimension?**
   - β-VAE techniques to disentangle and count true degrees of freedom

4. **Transfer across matrix sizes?**
   - Does structure learned on 2×2 transfer to 4×4?

---

## Next Steps

1. **Prototype for 2×2**: Strassen's decomposition is known, test if VAE can rediscover it
2. **Scale to 3×3**: Laderman's decomposition, more complex manifold
3. **Ternary constraint**: Add quantization, explore discrete solution set
4. **Connect to existing TritNet infrastructure**: Reuse ternary layer code

---

**The Philosophy**:

> We don't compute the manifold. We don't restrict to the manifold. We let the manifold emerge from the constraint. The VAE becomes a lens through which we observe structure that was always there but invisible to enumeration.

---

**Version**: 1.0 · **Updated**: 2025-12-29 · **Status**: Research Proposal
