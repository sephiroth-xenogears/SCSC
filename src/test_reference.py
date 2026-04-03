"""
SCSC Test Harness
==================
Known-answer test generation and verification for CUDA kernel validation.

Usage:
  python test_reference.py              # Run all tests
  python test_reference.py --export     # Export test vectors for CUDA

Validation criteria (from handoff M1-c):
  FP32 tolerance: ~1e-5
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from gen_model import (
    GenerativeModel, create_default_model, memory_report,
    STATE_DIM, NUM_POLICIES, VFE_ITERATIONS,
)
from reference import (
    active_inference_step, compute_vfe, stage1_vfe_minimization,
    stage2_evaluate_policies, stage3_action_selection,
    matvec_reference, verify_bank_conflict_free, softmax,
    InferenceResult,
)


FP32_TOLERANCE = 1e-5
RELAXED_TOLERANCE = 1e-4  # For accumulated floating point operations


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []

    def check(self, condition: bool, msg: str):
        if not condition:
            self.passed = False
            self.messages.append(f"  FAIL: {msg}")
        return condition

    def check_close(self, actual, expected, tol, label: str):
        if np.isscalar(actual):
            diff = abs(actual - expected)
            ok = diff <= tol
        else:
            diff = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
            ok = diff <= tol
        if not ok:
            self.check(False, f"{label}: max diff = {diff:.2e}, tolerance = {tol:.2e}")
        return ok

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}"]
        for m in self.messages:
            lines.append(m)
        return "\n".join(lines)


# ============================================================
# Test Cases
# ============================================================

def test_model_dimensions() -> TestResult:
    """Verify model matrix dimensions match design spec."""
    t = TestResult("Model Dimensions")
    model = create_default_model()

    t.check(model.A.shape == (38, 38), f"A shape: {model.A.shape} != (38, 38)")
    t.check(model.B.shape == (10, 38, 38), f"B shape: {model.B.shape} != (10, 38, 38)")
    t.check(model.C.shape == (38, 38), f"C shape: {model.C.shape} != (38, 38)")
    t.check(model.D.shape == (38,), f"D shape: {model.D.shape} != (38,)")
    t.check(model.Pi_o.shape == (38,), f"Pi_o shape: {model.Pi_o.shape}")
    t.check(model.Pi_x.shape == (38,), f"Pi_x shape: {model.Pi_x.shape}")

    # Verify FP32
    t.check(model.A.dtype == np.float32, f"A dtype: {model.A.dtype}")
    t.check(model.B.dtype == np.float32, f"B dtype: {model.B.dtype}")

    return t


def test_memory_layout() -> TestResult:
    """Verify memory usage matches handoff document Section 1.2.5."""
    t = TestResult("Memory Layout (Section 1.2.5)")
    model = create_default_model()
    report = memory_report(model)

    sm = report['shared_memory']
    # A[38][39] = 5,928 bytes (document says 5,928)
    t.check(sm['A_padded'] == 5928,
            f"A padded bytes: {sm['A_padded']} != 5928")

    # μ[38] = 152 bytes
    t.check(sm['mu'] == 152, f"μ bytes: {sm['mu']} != 152")

    # σ[38] = 152 bytes
    t.check(sm['sigma'] == 152, f"σ bytes: {sm['sigma']} != 152")

    # G[10] = 40 bytes
    t.check(sm['G'] == 40, f"G bytes: {sm['G']} != 40")

    # Total shared < 48KB
    t.check(sm['total'] < 48 * 1024,
            f"Shared total {sm['total']} exceeds 48KB")

    # Constant Memory: B[10][38][38] = 57,760 bytes (document says 57,760)
    cm = report['constant_memory']
    t.check(cm['B_matrices'] == 57760,
            f"B bytes: {cm['B_matrices']} != 57760")

    # Constant Memory utilization ~88%
    t.check(85.0 < cm['utilization_pct'] < 92.0,
            f"Constant mem utilization: {cm['utilization_pct']:.1f}%")

    return t


def test_bank_conflict_free() -> TestResult:
    """Verify A matrix padding eliminates bank conflicts."""
    t = TestResult("Bank Conflict Avoidance")
    from math import gcd

    padded = STATE_DIM + 1  # 39
    g = gcd(padded, 32)
    t.check(g == 1, f"gcd({padded}, 32) = {g}, should be 1")

    unpadded = gcd(STATE_DIM, 32)
    t.check(unpadded > 1,
            f"Without padding: gcd({STATE_DIM}, 32) = {unpadded} (confirms conflict)")

    return t


def test_matvec_consistency() -> TestResult:
    """Verify row-parallel MatVec matches numpy."""
    t = TestResult("MatVec Row-Parallel Consistency")
    model = create_default_model()
    rng = np.random.RandomState(42)
    x = rng.randn(STATE_DIM).astype(np.float32)

    np_result = model.A @ x
    row_result = matvec_reference(model.A, x)

    t.check_close(row_result, np_result, FP32_TOLERANCE, "MatVec A@x")
    return t


def test_vfe_decreases() -> TestResult:
    """Verify VFE monotonically decreases during minimization."""
    t = TestResult("VFE Monotonic Decrease")
    model = create_default_model()
    rng = np.random.RandomState(123)

    mu_init = model.D + rng.normal(0, 0.1, STATE_DIM).astype(np.float32)
    o = model.C @ (model.D + rng.normal(0, 0.05, STATE_DIM).astype(np.float32))

    _, _, trace = stage1_vfe_minimization(mu_init, o, model)

    for i in range(1, len(trace.F_history)):
        t.check(
            trace.F_history[i] <= trace.F_history[i-1] + RELAXED_TOLERANCE,
            f"F[{i}]={trace.F_history[i]:.6f} > F[{i-1}]={trace.F_history[i-1]:.6f}"
        )

    # F should decrease overall
    t.check(
        trace.F_history[-1] < trace.F_history[0],
        f"F did not decrease: {trace.F_history[0]:.6f} → {trace.F_history[-1]:.6f}"
    )

    return t


def test_vfe_gradient() -> TestResult:
    """Verify VFE gradient via finite differences."""
    t = TestResult("VFE Gradient (Finite Difference Check)")
    model = create_default_model()
    rng = np.random.RandomState(42)

    mu = rng.randn(STATE_DIM).astype(np.float32) * 0.1
    o = model.C @ (model.D + rng.normal(0, 0.05, STATE_DIM).astype(np.float32))

    F0, grad, _ = compute_vfe(mu, o, model)

    # Finite difference for each dimension
    eps = 1e-3
    grad_fd = np.zeros(STATE_DIM, dtype=np.float32)
    for i in range(STATE_DIM):
        mu_plus = mu.copy()
        mu_plus[i] += eps
        F_plus, _, _ = compute_vfe(mu_plus, o, model)
        grad_fd[i] = (F_plus - F0) / eps

    # Relaxed tolerance due to FP32 and finite difference approximation
    t.check_close(grad, grad_fd, 1e-2, "Gradient vs finite diff")
    return t


def test_softmax_properties() -> TestResult:
    """Verify softmax normalization and ordering."""
    t = TestResult("Softmax Properties")

    G = np.array([1.0, 2.0, 0.5, 3.0, 1.5], dtype=np.float32)
    probs = softmax(G)

    # Sum to 1
    t.check_close(np.sum(probs), 1.0, FP32_TOLERANCE, "Softmax sum")

    # All positive
    t.check(np.all(probs > 0), "Softmax all positive")

    # Lower G → higher probability (since softmax uses -G)
    best_idx = np.argmin(G)
    t.check(np.argmax(probs) == best_idx,
            f"Best policy: argmax(P)={np.argmax(probs)} != argmin(G)={best_idx}")

    return t


def test_policy_offset() -> TestResult:
    """Verify Plan B interface (policy_offset/policy_count)."""
    t = TestResult("Plan B Interface (policy_offset)")
    model = create_default_model()
    rng = np.random.RandomState(42)
    mu = rng.randn(STATE_DIM).astype(np.float32) * 0.1

    # Full evaluation
    G_full = stage2_evaluate_policies(mu, model, policy_offset=0, policy_count=10)

    # Partial evaluation (first 5)
    G_first = stage2_evaluate_policies(mu, model, policy_offset=0, policy_count=5)

    # Partial evaluation (last 5)
    G_last = stage2_evaluate_policies(mu, model, policy_offset=5, policy_count=5)

    # First 5 should match
    t.check_close(G_first, G_full[:5], FP32_TOLERANCE, "G[0:5] match")

    # Last 5 should match
    t.check_close(G_last, G_full[5:], FP32_TOLERANCE, "G[5:10] match")

    return t


def test_full_pipeline_deterministic() -> TestResult:
    """Verify complete pipeline is deterministic with same seed."""
    t = TestResult("Full Pipeline Determinism")
    model = create_default_model(seed=42)
    rng = np.random.RandomState(123)

    mu_init = model.D + rng.normal(0, 0.1, STATE_DIM).astype(np.float32)
    o = model.C @ (model.D + rng.normal(0, 0.05, STATE_DIM).astype(np.float32))

    r1 = active_inference_step(mu_init, o, model)

    # Re-run with identical inputs
    r2 = active_inference_step(mu_init, o, model)

    t.check_close(r1.mu, r2.mu, 0.0, "μ identical")
    t.check_close(r1.F, r2.F, 0.0, "F identical")
    t.check_close(r1.G, r2.G, 0.0, "G identical")
    t.check(r1.action == r2.action, f"Action: {r1.action} != {r2.action}")

    return t


def test_full_pipeline_values() -> TestResult:
    """
    Run full pipeline and record reference values.
    These exact values will be compared against CUDA output.
    """
    t = TestResult("Full Pipeline Reference Values")
    model = create_default_model(seed=42)
    rng = np.random.RandomState(123)

    mu_init = model.D + rng.normal(0, 0.1, STATE_DIM).astype(np.float32)
    o = model.C @ (model.D + rng.normal(0, 0.05, STATE_DIM).astype(np.float32))

    result = active_inference_step(mu_init, o, model)

    # Basic sanity checks
    t.check(np.all(np.isfinite(result.mu)), "μ is finite")
    t.check(np.isfinite(result.F), "F is finite")
    t.check(np.all(np.isfinite(result.G)), "G is finite")
    t.check(np.all(np.isfinite(result.policy_probs)), "Probs are finite")
    t.check(0 <= result.action < NUM_POLICIES, f"Action in range: {result.action}")
    t.check_close(np.sum(result.policy_probs), 1.0, FP32_TOLERANCE, "Prob sum")

    # Print reference values for visual verification
    print(f"\n  Reference values (for CUDA comparison):")
    print(f"    F_initial = {result.vfe_trace.F_history[0]:.8f}")
    print(f"    F_final   = {result.F:.8f}")
    print(f"    μ[0:5]    = {result.mu[:5]}")
    print(f"    G[0:5]    = {result.G[:5]}")
    print(f"    P[0:5]    = {result.policy_probs[:5]}")
    print(f"    action    = {result.action}")

    return t


# ============================================================
# Test Vector Export (for CUDA verification)
# ============================================================

def export_test_vectors(output_dir: str = "test_vectors"):
    """
    Export deterministic test vectors as binary files for CUDA verification.

    File format: raw FP32 little-endian binary
    This allows CUDA test harness to load with a simple fread().
    """
    os.makedirs(output_dir, exist_ok=True)

    model = create_default_model(seed=42)
    rng = np.random.RandomState(123)

    mu_init = model.D + rng.normal(0, 0.1, STATE_DIM).astype(np.float32)
    o = model.C @ (model.D + rng.normal(0, 0.05, STATE_DIM).astype(np.float32))

    result = active_inference_step(mu_init, o, model)

    # Save inputs
    model.A.tofile(os.path.join(output_dir, "A.bin"))
    model.B.tofile(os.path.join(output_dir, "B.bin"))
    model.C.tofile(os.path.join(output_dir, "C.bin"))
    model.D.tofile(os.path.join(output_dir, "D.bin"))
    model.Pi_o.tofile(os.path.join(output_dir, "Pi_o.bin"))
    model.Pi_x.tofile(os.path.join(output_dir, "Pi_x.bin"))
    mu_init.tofile(os.path.join(output_dir, "mu_init.bin"))
    o.tofile(os.path.join(output_dir, "o.bin"))

    # Save expected outputs
    result.mu.tofile(os.path.join(output_dir, "mu_expected.bin"))
    np.array([result.F], dtype=np.float32).tofile(
        os.path.join(output_dir, "F_expected.bin"))
    result.G.tofile(os.path.join(output_dir, "G_expected.bin"))
    result.policy_probs.tofile(os.path.join(output_dir, "probs_expected.bin"))
    np.array([result.action], dtype=np.int32).tofile(
        os.path.join(output_dir, "action_expected.bin"))

    # Save VFE trace (F values at each iteration)
    np.array(result.vfe_trace.F_history, dtype=np.float32).tofile(
        os.path.join(output_dir, "F_trace.bin"))

    # Save metadata as JSON
    metadata = {
        "state_dim": STATE_DIM,
        "num_policies": NUM_POLICIES,
        "vfe_iterations": VFE_ITERATIONS,
        "fp32_tolerance": FP32_TOLERANCE,
        "relaxed_tolerance": RELAXED_TOLERANCE,
        "seed_model": 42,
        "seed_data": 123,
        "files": {
            "A.bin": {"shape": list(model.A.shape), "dtype": "float32"},
            "B.bin": {"shape": list(model.B.shape), "dtype": "float32"},
            "C.bin": {"shape": list(model.C.shape), "dtype": "float32"},
            "D.bin": {"shape": list(model.D.shape), "dtype": "float32"},
            "Pi_o.bin": {"shape": list(model.Pi_o.shape), "dtype": "float32"},
            "Pi_x.bin": {"shape": list(model.Pi_x.shape), "dtype": "float32"},
            "mu_init.bin": {"shape": [STATE_DIM], "dtype": "float32"},
            "o.bin": {"shape": [STATE_DIM], "dtype": "float32"},
            "mu_expected.bin": {"shape": [STATE_DIM], "dtype": "float32"},
            "F_expected.bin": {"shape": [1], "dtype": "float32"},
            "G_expected.bin": {"shape": [NUM_POLICIES], "dtype": "float32"},
            "probs_expected.bin": {"shape": [NUM_POLICIES], "dtype": "float32"},
            "action_expected.bin": {"shape": [1], "dtype": "int32"},
            "F_trace.bin": {"shape": [VFE_ITERATIONS + 1], "dtype": "float32"},
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Test vectors exported to {output_dir}/")
    print(f"  {len(metadata['files'])} files, metadata.json included")

    # Verify round-trip
    A_loaded = np.fromfile(
        os.path.join(output_dir, "A.bin"), dtype=np.float32
    ).reshape(STATE_DIM, STATE_DIM)
    assert np.array_equal(A_loaded, model.A), "Round-trip verification failed!"
    print("  Round-trip verification: OK")


# ============================================================
# Main: Run all tests
# ============================================================

def run_all_tests() -> bool:
    """Run all tests and print report."""
    tests = [
        test_model_dimensions,
        test_memory_layout,
        test_bank_conflict_free,
        test_matvec_consistency,
        test_vfe_decreases,
        test_vfe_gradient,
        test_softmax_properties,
        test_policy_offset,
        test_full_pipeline_deterministic,
        test_full_pipeline_values,
    ]

    print("=" * 60)
    print("SCSC Reference Implementation Test Suite")
    print("=" * 60)

    all_passed = True
    results = []
    for test_fn in tests:
        result = test_fn()
        results.append(result)
        print(result.report())
        if not result.passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print(f"ALL {len(tests)} TESTS PASSED")
    else:
        failed = sum(1 for r in results if not r.passed)
        print(f"FAILURES: {failed}/{len(tests)} tests failed")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    if '--export' in sys.argv:
        export_dir = "test_vectors"
        if len(sys.argv) > 2 and sys.argv[-1] != '--export':
            export_dir = sys.argv[-1]
        export_test_vectors(export_dir)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
