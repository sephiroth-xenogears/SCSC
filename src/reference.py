"""
SCSC Active Inference Reference Implementation
================================================
Python/NumPy reference for the fused CUDA kernel.
Implements the 3-stage pipeline exactly as designed in Phase 7-8 handoff.

Kernel structure (from handoff document Section 1.2.1):
  active_inference_step(policy_offset, policy_count, ...)
  ├── Prologue: Load data (DRAM → working memory)
  ├── Stage 1: VFE minimization (N iterations)
  ├── Stage 2: G(π) evaluation (K policies)
  ├── Stage 3: Action selection (softmax → action)
  └── Epilogue: Return results (action, μ, F)

This implementation prioritizes clarity and correctness over performance.
Every intermediate value is stored for comparison with CUDA output.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from gen_model import (
    GenerativeModel, create_default_model,
    STATE_DIM, NUM_POLICIES, VFE_ITERATIONS, VFE_LEARNING_RATE,
)


@dataclass
class VFETrace:
    """Records intermediate values from VFE minimization for CUDA verification."""
    mu_history: List[np.ndarray] = field(default_factory=list)
    F_history: List[float] = field(default_factory=list)
    gradient_history: List[np.ndarray] = field(default_factory=list)
    prediction_error_history: List[np.ndarray] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Complete output of one active inference step."""
    # Stage 1 outputs
    mu: np.ndarray              # Updated beliefs [state_dim]
    F: float                    # Final free energy (scalar)
    vfe_trace: VFETrace         # Intermediate values for debugging

    # Stage 2 outputs
    G: np.ndarray               # Expected free energy per policy [num_policies]

    # Stage 3 outputs
    policy_probs: np.ndarray    # Softmax probabilities [num_policies]
    action: int                 # Selected action (argmax policy index)
    action_vector: np.ndarray   # Control vector for selected policy [state_dim]


# ============================================================
# Stage 1: Variational Free Energy Minimization
# ============================================================

def compute_vfe(
    mu: np.ndarray,
    o: np.ndarray,
    model: GenerativeModel,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Variational Free Energy and its gradient.

    VFE (simplified, diagonal precision):
      F(μ) = 0.5 * ε_o^T Π_o ε_o  +  0.5 * ε_x^T Π_x ε_x

    Where:
      ε_o = o - C @ μ          (sensory prediction error)
      ε_x = μ - d              (prior prediction error)
      Π_o = diag(Pi_o)         (observation precision)
      Π_x = diag(Pi_x)         (state precision)

    Gradient:
      ∂F/∂μ = -C^T @ (Π_o * ε_o)  -  Π_x * ε_x

    Note: The A matrix is used in Stage 2 for state prediction.
    In the CUDA kernel, the A @ μ MatVec in Stage 1 computes
    the predicted next state for use in both VFE and G(π).

    Args:
        mu: Current belief state [state_dim]
        o: Current observation [obs_dim]
        model: Generative model

    Returns:
        F: Free energy (scalar)
        gradient: ∂F/∂μ [state_dim]
        eps_o: Sensory prediction error [obs_dim]
    """
    # Sensory prediction error
    o_predicted = model.C @ mu
    eps_o = o - o_predicted                        # [obs_dim]

    # Prior prediction error
    eps_x = mu - model.D                           # [state_dim]

    # Free energy (scalar)
    F_sensory = 0.5 * np.sum(model.Pi_o * eps_o * eps_o)
    F_prior = 0.5 * np.sum(model.Pi_x * eps_x * eps_x)
    F = F_sensory + F_prior

    # Gradient ∂F/∂μ
    # -C^T @ diag(Pi_o) @ eps_o  =  -C^T @ (Pi_o * eps_o)
    grad_sensory = -model.C.T @ (model.Pi_o * eps_o)   # [state_dim]
    grad_prior = model.Pi_x * eps_x                      # [state_dim]
    gradient = grad_sensory + grad_prior

    return F, gradient, eps_o


def stage1_vfe_minimization(
    mu_init: np.ndarray,
    o: np.ndarray,
    model: GenerativeModel,
    n_iterations: int = VFE_ITERATIONS,
    learning_rate: float = VFE_LEARNING_RATE,
) -> Tuple[np.ndarray, float, VFETrace]:
    """
    Stage 1: VFE minimization via gradient descent.

    Corresponds to CUDA kernel Stage 1:
      - μ is in Shared Memory (visible to all threads)
      - Gradients are in Registers (per-thread)
      - A matrix is in Shared Memory (for MatVec)
      - __syncthreads() after each μ update

    Args:
        mu_init: Initial belief state [state_dim]
        o: Current observation [obs_dim]
        model: Generative model
        n_iterations: Number of gradient descent iterations
        learning_rate: Step size

    Returns:
        mu: Optimized belief state [state_dim]
        F: Final free energy
        trace: Intermediate values for verification
    """
    trace = VFETrace()
    mu = mu_init.copy()

    for iteration in range(n_iterations):
        # Record state before update
        trace.mu_history.append(mu.copy())

        # Compute VFE and gradient
        F, gradient, eps_o = compute_vfe(mu, o, model)

        trace.F_history.append(F)
        trace.gradient_history.append(gradient.copy())
        trace.prediction_error_history.append(eps_o.copy())

        # Gradient descent update
        # In CUDA: each thread i updates mu[i] -= lr * gradient[i]
        # Then __syncthreads() before next iteration
        mu = mu - learning_rate * gradient

    # Final F after last update
    F_final, _, _ = compute_vfe(mu, o, model)
    trace.F_history.append(F_final)
    trace.mu_history.append(mu.copy())

    return mu, F_final, trace


# ============================================================
# Stage 2: Expected Free Energy G(π) Evaluation
# ============================================================

def compute_G_policy(
    mu: np.ndarray,
    policy_idx: int,
    model: GenerativeModel,
) -> float:
    """
    Compute Expected Free Energy G(π) for a single policy.

    G(π_k) = ambiguity + risk

    Ambiguity: Expected uncertainty of observations under policy k
      = -E[ln p(o|x)] under predicted state

    Risk: KL divergence from predicted observations to preferred observations
      = E[ln q(o|π) - ln p(o)]  (simplified)

    Simplified computation for Phase 7 (Plan A):
      1. Predict next state:  μ_pred = A @ μ + B[k] @ μ
      2. Predict observation: o_pred = C @ μ_pred
      3. G(k) = 0.5 * Σ_i (1/Pi_o[i])           (ambiguity: inverse precision)
              + 0.5 * (o_pred - o_pref)^T Π_o (o_pred - o_pref)  (risk)

    Where o_pref = C @ D (preferred observation = observation of prior mean)

    In CUDA kernel:
      - B[k] is loaded from Constant Memory
      - MatVec: μ_pred = A @ μ + B[k] @ μ
      - Reduction via atomicAdd(&G_shared[k], g_local)

    Args:
        mu: Current optimized belief [state_dim]
        policy_idx: Policy index k
        model: Generative model

    Returns:
        G_k: Expected free energy for policy k (scalar)
    """
    # Predict next state under policy k
    # μ_pred = A @ μ + B[k] @ μ
    mu_pred = model.A @ mu + model.B[policy_idx] @ mu    # [state_dim]

    # Predict observation under predicted state
    o_pred = model.C @ mu_pred                             # [obs_dim]

    # Preferred observation (observation of prior/goal state)
    o_pref = model.C @ model.D                             # [obs_dim]

    # Ambiguity term: expected sensory uncertainty
    # Sum of inverse precisions (entropy of observation likelihood)
    ambiguity = 0.5 * np.sum(1.0 / model.Pi_o)

    # Risk term: divergence from preferred observations
    o_diff = o_pred - o_pref
    risk = 0.5 * np.sum(model.Pi_o * o_diff * o_diff)

    G_k = ambiguity + risk
    return G_k


def stage2_evaluate_policies(
    mu: np.ndarray,
    model: GenerativeModel,
    policy_offset: int = 0,
    policy_count: int = NUM_POLICIES,
) -> np.ndarray:
    """
    Stage 2: Evaluate G(π) for all policies.

    Supports Plan B interface: policy_offset and policy_count
    allow evaluating a slice of policies without changing kernel code.

    Plan A (Phase 7): policy_offset=0, policy_count=10
    Plan B (Phase 8+): Tier 1 offset=0 count=10, Tier 2 variable

    In CUDA kernel:
      - B matrices in Constant Memory (56.4 KB / 64 KB = 88%)
      - G[K] results in Shared Memory (40 bytes)
      - atomicAdd for reduction within each policy

    Args:
        mu: Optimized belief state from Stage 1 [state_dim]
        model: Generative model
        policy_offset: Starting policy index (Plan B support)
        policy_count: Number of policies to evaluate (Plan B support)

    Returns:
        G: Expected free energy for each policy [policy_count]
    """
    G = np.zeros(policy_count, dtype=np.float32)

    for k in range(policy_count):
        actual_k = policy_offset + k
        G[k] = compute_G_policy(mu, actual_k, model)

    return G


# ============================================================
# Stage 3: Action Selection (Softmax)
# ============================================================

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax.

    P(k) = exp(-G[k] / temperature) / Σ_j exp(-G[j] / temperature)

    Note the negative sign: lower G(π) = better policy = higher probability.
    """
    # Shift for numerical stability
    scaled = -x / temperature
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def stage3_action_selection(
    G: np.ndarray,
    mu: np.ndarray,
    model: GenerativeModel,
    policy_offset: int = 0,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Stage 3: Select action via softmax over G(π).

    In CUDA kernel:
      - Softmax computed in Shared Memory
      - Action index written to DRAM (Epilogue)

    Args:
        G: Expected free energy per policy [num_policies]
        mu: Current belief state [state_dim]
        model: Generative model
        policy_offset: For Plan B compatibility
        temperature: Softmax temperature (lower = more deterministic)

    Returns:
        probs: Policy probabilities [num_policies]
        action_idx: Selected policy index
        action_vector: State change for selected policy [state_dim]
    """
    probs = softmax(G, temperature)

    # Select best policy (argmax = most probable)
    action_idx = int(np.argmax(probs))

    # Compute action vector: what the selected policy does to the state
    actual_k = policy_offset + action_idx
    action_vector = model.B[actual_k] @ mu

    return probs, action_idx, action_vector


# ============================================================
# Complete Active Inference Step (Fused Kernel)
# ============================================================

def active_inference_step(
    mu_init: np.ndarray,
    o: np.ndarray,
    model: GenerativeModel,
    policy_offset: int = 0,
    policy_count: int = NUM_POLICIES,
    n_iterations: int = VFE_ITERATIONS,
    learning_rate: float = VFE_LEARNING_RATE,
    temperature: float = 1.0,
) -> InferenceResult:
    """
    Complete active inference step: the Python equivalent of the fused CUDA kernel.

    active_inference_step(policy_offset, policy_count, ...)
    ├── [Prologue]: Data already in function arguments
    ├── Stage 1: VFE minimization
    ├── Stage 2: G(π) evaluation
    ├── Stage 3: Action selection
    └── [Epilogue]: Results returned in InferenceResult

    This function signature matches the CUDA kernel interface,
    including Plan B support via policy_offset/policy_count.

    Args:
        mu_init: Initial belief state [state_dim]
        o: Current observation [obs_dim]
        model: Generative model
        policy_offset: Starting policy index (Plan B)
        policy_count: Number of policies to evaluate (Plan B)
        n_iterations: VFE iterations
        learning_rate: VFE learning rate
        temperature: Softmax temperature

    Returns:
        InferenceResult with all outputs and traces
    """
    # ---- Stage 1: VFE Minimization ----
    mu, F, vfe_trace = stage1_vfe_minimization(
        mu_init, o, model, n_iterations, learning_rate
    )

    # ---- Stage 2: G(π) Evaluation ----
    G = stage2_evaluate_policies(mu, model, policy_offset, policy_count)

    # ---- Stage 3: Action Selection ----
    probs, action_idx, action_vector = stage3_action_selection(
        G, mu, model, policy_offset, temperature
    )

    return InferenceResult(
        mu=mu,
        F=F,
        vfe_trace=vfe_trace,
        G=G,
        policy_probs=probs,
        action=action_idx,
        action_vector=action_vector,
    )


# ============================================================
# Utility: MatVec verification helpers
# ============================================================

def matvec_reference(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Explicit row-parallel MatVec matching CUDA thread mapping.

    In CUDA: thread i computes s[i] = Σ_j A[i][j] * x[j]
    This function computes the same thing element-by-element
    so we can verify individual thread outputs.

    Returns:
        result[i] for each i (equivalent to A @ x)
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        acc = np.float32(0.0)
        for j in range(n):
            acc += A[i, j] * x[j]
        result[i] = acc
    return result


def verify_bank_conflict_free(state_dim: int = STATE_DIM) -> bool:
    """
    Verify A matrix padding eliminates bank conflicts.

    Shared Memory has 32 banks.
    stride = padded_cols = state_dim + 1 = 39
    gcd(39, 32) should be 1 for conflict-free access.
    """
    padded_cols = state_dim + 1  # 39
    from math import gcd
    g = gcd(padded_cols, 32)
    conflict_free = (g == 1)
    print(f"Padded columns: {padded_cols}")
    print(f"gcd({padded_cols}, 32) = {g}")
    print(f"Bank conflict free: {conflict_free}")

    # Compare with unpadded
    g_unpadded = gcd(state_dim, 32)
    print(f"\nWithout padding: gcd({state_dim}, 32) = {g_unpadded}")
    print(f"Would have {g_unpadded}-way bank conflicts")

    return conflict_free


if __name__ == '__main__':
    print("=== SCSC Active Inference Reference ===\n")

    # Create model and synthetic observation
    model = create_default_model(seed=42)
    rng = np.random.RandomState(123)

    mu_init = model.D + rng.normal(0, 0.1, STATE_DIM).astype(np.float32)
    o = model.C @ (model.D + rng.normal(0, 0.05, STATE_DIM).astype(np.float32))

    # Run complete inference step
    result = active_inference_step(mu_init, o, model)

    print(f"Stage 1 - VFE Minimization:")
    print(f"  Initial F: {result.vfe_trace.F_history[0]:.6f}")
    print(f"  Final F:   {result.F:.6f}")
    print(f"  F reduced: {result.vfe_trace.F_history[0] - result.F:.6f}")
    print(f"  μ norm:    {np.linalg.norm(result.mu):.6f}")

    print(f"\nStage 2 - G(π) Evaluation:")
    for k in range(NUM_POLICIES):
        marker = " ← best" if k == result.action else ""
        print(f"  G[{k}] = {result.G[k]:.6f}  P = {result.policy_probs[k]:.4f}{marker}")

    print(f"\nStage 3 - Action Selection:")
    print(f"  Selected policy: {result.action}")
    print(f"  Action vector norm: {np.linalg.norm(result.action_vector):.6f}")

    print(f"\n=== Bank Conflict Verification ===")
    verify_bank_conflict_free()
