"""
SCSC Generative Model Definition
=================================
Defines the generative model matrices and parameters for active inference.
This module serves as the Python counterpart of gen_model.cuh.

State space (38 dimensions):
  - 17 servo positions  [0:17]
  - 17 servo velocities [17:34]
  - 2 camera positions  [34:36]
  - 2 camera velocities [36:38]

Future extensions (Phase 8+):
  - IMU/FSR sensors: 38 → 52 dimensions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Constants matching CUDA kernel design
# ============================================================
STATE_DIM = 38          # 17 servo pos + 17 servo vel + 2 cam pos + 2 cam vel
NUM_SERVO = 17
NUM_CAMERA = 2
NUM_POLICIES = 10       # K=10 (Plan A)
VFE_ITERATIONS = 10     # N iterations for VFE minimization
VFE_LEARNING_RATE = 0.01

# Memory layout constants (for CUDA verification)
A_PADDED_COLS = 39      # A[38][39] with +1 column pad for bank conflict avoidance
SHARED_MEM_A_BYTES = STATE_DIM * A_PADDED_COLS * 4  # 5,928 bytes
CONST_MEM_B_BYTES = NUM_POLICIES * STATE_DIM * STATE_DIM * 4  # 57,760 bytes


@dataclass
class GenerativeModel:
    """
    Active inference generative model for Manoi PF01 robot control.

    Generative model structure:
      - Dynamics:    x_t = A @ x_{t-1} + B[k] @ u_t + noise_x
      - Observation: o_t = C @ x_t + noise_o
      - Prior:       x_0 ~ N(d, Sigma_x)

    Where:
      A: State transition matrix [STATE_DIM x STATE_DIM]
      B: Policy-dependent control matrices [NUM_POLICIES x STATE_DIM x STATE_DIM]
      C: Observation matrix [obs_dim x STATE_DIM]
      D: Prior expectations [STATE_DIM]
      Pi_o: Observation precision (diagonal) [obs_dim]
      Pi_x: State precision (diagonal) [STATE_DIM]
    """
    state_dim: int = STATE_DIM
    num_policies: int = NUM_POLICIES
    obs_dim: int = STATE_DIM  # Default: full state observation

    # Model matrices (initialized by create_* functions)
    A: np.ndarray = field(default=None)      # [state_dim, state_dim]
    B: np.ndarray = field(default=None)      # [num_policies, state_dim, state_dim]
    C: np.ndarray = field(default=None)      # [obs_dim, state_dim]
    D: np.ndarray = field(default=None)      # [state_dim] prior mean
    Pi_o: np.ndarray = field(default=None)   # [obs_dim] observation precision
    Pi_x: np.ndarray = field(default=None)   # [state_dim] state precision

    def __post_init__(self):
        if self.A is None:
            self.A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        if self.B is None:
            self.B = np.zeros((self.num_policies, self.state_dim, self.state_dim),
                              dtype=np.float32)
        if self.C is None:
            self.C = np.zeros((self.obs_dim, self.state_dim), dtype=np.float32)
        if self.D is None:
            self.D = np.zeros(self.state_dim, dtype=np.float32)
        if self.Pi_o is None:
            self.Pi_o = np.ones(self.obs_dim, dtype=np.float32)
        if self.Pi_x is None:
            self.Pi_x = np.ones(self.state_dim, dtype=np.float32)


def create_default_model(seed: int = 42) -> GenerativeModel:
    """
    Create a default generative model with physically plausible parameters.

    A matrix structure:
      Position-velocity coupling (x_pos += dt * x_vel)
      Damped velocity dynamics (x_vel *= decay)

    B matrices:
      Each policy applies different control gains to velocities.

    C matrix:
      Block diagonal (servo block + camera block).
      This is the initial design choice; dense matrix is an alternative
      to be evaluated during implementation.
    """
    rng = np.random.RandomState(seed)
    dt = 0.01   # 100 Hz control loop
    decay = 0.95  # velocity damping

    model = GenerativeModel()

    # ----------------------------------------------------------
    # A matrix: State transition (position-velocity dynamics)
    # ----------------------------------------------------------
    # Structure:
    #   [ I    dt*I   0    0   ]  <- servo positions
    #   [ 0  decay*I  0    0   ]  <- servo velocities
    #   [ 0    0      I  dt*I  ]  <- camera positions
    #   [ 0    0      0  decay*I] <- camera velocities
    A = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float32)

    # Servo position block: x_pos[i] += dt * x_vel[i]
    for i in range(NUM_SERVO):
        A[i, i] = 1.0                        # position persistence
        A[i, NUM_SERVO + i] = dt              # velocity integration

    # Servo velocity block: x_vel[i] *= decay
    for i in range(NUM_SERVO):
        A[NUM_SERVO + i, NUM_SERVO + i] = decay

    # Camera position block
    cam_offset = 2 * NUM_SERVO  # = 34
    for i in range(NUM_CAMERA):
        A[cam_offset + i, cam_offset + i] = 1.0
        A[cam_offset + i, cam_offset + NUM_CAMERA + i] = dt

    # Camera velocity block
    for i in range(NUM_CAMERA):
        A[cam_offset + NUM_CAMERA + i, cam_offset + NUM_CAMERA + i] = decay

    model.A = A

    # ----------------------------------------------------------
    # B matrices: Policy-dependent control gains
    # ----------------------------------------------------------
    # Each policy applies different gains to velocity dimensions.
    # B[k] acts on the state to produce policy-specific transitions.
    B = np.zeros((NUM_POLICIES, STATE_DIM, STATE_DIM), dtype=np.float32)

    for k in range(NUM_POLICIES):
        # Each policy has a different control gain pattern
        gains = rng.uniform(-0.1, 0.1, size=NUM_SERVO).astype(np.float32)
        for i in range(NUM_SERVO):
            # Policy k modifies velocity of servo i
            B[k, NUM_SERVO + i, i] = gains[i]

        # Camera control gains (smaller)
        cam_gains = rng.uniform(-0.05, 0.05, size=NUM_CAMERA).astype(np.float32)
        for i in range(NUM_CAMERA):
            B[k, cam_offset + NUM_CAMERA + i, cam_offset + i] = cam_gains[i]

    model.B = B

    # ----------------------------------------------------------
    # C matrix: Observation model (block diagonal)
    # ----------------------------------------------------------
    # Design choice: block diagonal (servo/camera independent)
    # Alternative: dense matrix (to be evaluated)
    #
    # Block diagonal structure:
    #   [ C_servo    0      ]
    #   [   0     C_camera  ]
    C = create_block_diagonal_C(rng)
    model.C = C

    # ----------------------------------------------------------
    # D: Prior state expectations (neutral pose)
    # ----------------------------------------------------------
    D = np.zeros(STATE_DIM, dtype=np.float32)
    # Servo neutral positions (centered)
    D[:NUM_SERVO] = 0.0
    # Velocities start at zero
    D[NUM_SERVO:2*NUM_SERVO] = 0.0
    # Camera centered
    D[cam_offset:cam_offset + NUM_CAMERA] = 0.0
    D[cam_offset + NUM_CAMERA:] = 0.0
    model.D = D

    # ----------------------------------------------------------
    # Precision matrices (diagonal, stored as vectors)
    # ----------------------------------------------------------
    # Observation precision: higher for positions, lower for velocities
    Pi_o = np.ones(STATE_DIM, dtype=np.float32)
    Pi_o[:NUM_SERVO] = 10.0           # servo position precision
    Pi_o[NUM_SERVO:2*NUM_SERVO] = 1.0  # servo velocity precision
    Pi_o[cam_offset:cam_offset + NUM_CAMERA] = 10.0       # camera pos
    Pi_o[cam_offset + NUM_CAMERA:] = 1.0                   # camera vel
    model.Pi_o = Pi_o

    # State precision (prior)
    Pi_x = np.ones(STATE_DIM, dtype=np.float32) * 0.1
    model.Pi_x = Pi_x

    return model


def create_block_diagonal_C(rng: np.random.RandomState) -> np.ndarray:
    """
    Create block-diagonal observation matrix.

    Structure:
      C = [ C_servo    0      ]   servo block: 34x34
          [   0     C_camera  ]   camera block: 4x4

    Each block is near-identity with small perturbations,
    representing slightly noisy/coupled observations.
    """
    C = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float32)

    # Servo block (positions + velocities)
    servo_dim = 2 * NUM_SERVO  # 34
    C[:servo_dim, :servo_dim] = (
        np.eye(servo_dim, dtype=np.float32)
        + rng.normal(0, 0.01, (servo_dim, servo_dim)).astype(np.float32)
    )

    # Camera block (positions + velocities)
    cam_dim = 2 * NUM_CAMERA  # 4
    cam_offset = servo_dim    # 34
    C[cam_offset:, cam_offset:] = (
        np.eye(cam_dim, dtype=np.float32)
        + rng.normal(0, 0.01, (cam_dim, cam_dim)).astype(np.float32)
    )

    return C


def create_dense_C(rng: np.random.RandomState) -> np.ndarray:
    """
    Alternative: Dense observation matrix.
    Servo and camera observations are coupled.
    Use this to compare with block-diagonal and decide which to adopt.
    """
    C = (
        np.eye(STATE_DIM, dtype=np.float32)
        + rng.normal(0, 0.01, (STATE_DIM, STATE_DIM)).astype(np.float32)
    )
    return C


def memory_report(model: GenerativeModel) -> dict:
    """
    Report memory usage matching CUDA kernel memory hierarchy design.

    Returns dict with byte counts for each memory tier.
    Compare with handoff document Section 1.2.5.
    """
    report = {}

    # Shared Memory tier
    a_bytes = model.state_dim * A_PADDED_COLS * 4  # A[38][39] padded
    mu_bytes = model.state_dim * 4
    sigma_bytes = model.state_dim * 4
    g_bytes = model.num_policies * 4
    c_d_o_bytes = (
        model.obs_dim * model.state_dim * 4  # C matrix
        + model.state_dim * 4                  # D vector
        + model.obs_dim * 4                    # o vector
    )
    shared_total = a_bytes + mu_bytes + sigma_bytes + g_bytes + c_d_o_bytes

    report['shared_memory'] = {
        'A_padded': a_bytes,
        'mu': mu_bytes,
        'sigma': sigma_bytes,
        'G': g_bytes,
        'C_D_o': c_d_o_bytes,
        'total': shared_total,
        'capacity': 48 * 1024,
        'utilization_pct': 100.0 * shared_total / (48 * 1024),
    }

    # Constant Memory tier
    b_bytes = model.num_policies * model.state_dim * model.state_dim * 4
    report['constant_memory'] = {
        'B_matrices': b_bytes,
        'capacity': 64 * 1024,
        'utilization_pct': 100.0 * b_bytes / (64 * 1024),
    }

    # Register tier (per thread)
    grad_bytes = 4       # dF/dmu[tid]
    eps_bytes = 4         # epsilon[tid]
    work_vars = 120       # ~30 working variables
    reg_total = grad_bytes + eps_bytes + work_vars
    report['registers_per_thread'] = {
        'gradient': grad_bytes,
        'epsilon': eps_bytes,
        'work_vars': work_vars,
        'total': reg_total,
        'capacity': 255 * 4,  # 255 registers * 4 bytes
        'utilization_pct': 100.0 * reg_total / (255 * 4),
    }

    return report


if __name__ == '__main__':
    model = create_default_model()
    print("=== SCSC Generative Model ===")
    print(f"State dimension: {model.state_dim}")
    print(f"Observation dimension: {model.obs_dim}")
    print(f"Number of policies: {model.num_policies}")
    print(f"\nA matrix shape: {model.A.shape}")
    print(f"B matrices shape: {model.B.shape}")
    print(f"C matrix shape: {model.C.shape}")
    print(f"D vector shape: {model.D.shape}")

    print("\n=== Memory Layout Report ===")
    report = memory_report(model)
    for tier, info in report.items():
        print(f"\n{tier}:")
        for k, v in info.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}%")
            else:
                print(f"  {k}: {v:,} bytes" if 'byte' in k or k in ['total', 'capacity',
                      'A_padded', 'mu', 'sigma', 'G', 'C_D_o', 'B_matrices',
                      'gradient', 'epsilon', 'work_vars'] else f"  {k}: {v}")
