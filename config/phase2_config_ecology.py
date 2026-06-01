"""
phase2_config_ecology.py
------------------------

Configuration file for CDR Phase II.3 — Ecological systems
(predator–prey dynamics).

Aligned with:
- loader V3 (log-returns only)
- reduced state space (no year_norm / no SOI)
- improved kernel stability
"""

import os


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "ecology",
    "lynxhare.csv"
)


# ---------------------------------------------------------
# SYSTEM VARIABLES (FINAL)
# ---------------------------------------------------------

# MUST match loader V3 output exactly
STATE_VARIABLES = [
    "hare_log_return",
    "lynx_log_return",
]


# ---------------------------------------------------------
# DISCRETIZATION
# ---------------------------------------------------------

BINS_PER_VARIABLE = 3

TOTAL_STATES = BINS_PER_VARIABLE ** len(STATE_VARIABLES)  # = 9


# ---------------------------------------------------------
# KERNEL (NEW — CRITICAL)
# ---------------------------------------------------------

# Improves stability with small datasets
KERNEL_DIRICHLET_ALPHA = 0.3


# ---------------------------------------------------------
# EPSILON GRID
# ---------------------------------------------------------

EPS_GRID_SIZE = 21

EPS_MIN = 0.0
EPS_MAX = 0.5


# ---------------------------------------------------------
# INJECTION TEST
# ---------------------------------------------------------

INJECTION_EPS = 0.25

INJECTION_TOL = 0.05

# Mais agressivo para garantir recuperação
INJECTION_LENGTH_MULTIPLIER = 20


# ---------------------------------------------------------
# CONTROLS (LEAN SET — IMPORTANT)
# ---------------------------------------------------------

N_CONTROLS = 4

CONTROL_TYPES = [
    "shuffle_time",
    "block_shuffle",
    "species_swap",
    "transition_randomization",
]


# ---------------------------------------------------------
# HOLDOUT TEST
# ---------------------------------------------------------

# Mantido apenas por compatibilidade
TRAIN_FRACTION = 0.7

MAX_GENERALIZATION_DELTA = 0.12


# ---------------------------------------------------------
# DISCRETIZATION SENSITIVITY
# ---------------------------------------------------------

ALT_BINS = 3

ALT_QUANTILES = (0.25, 0.75)

MAX_BIN_DELTA = 0.15


# ---------------------------------------------------------
# RESULTS PATH
# ---------------------------------------------------------

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "phase2_ecology"
)


# ---------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------

RANDOM_SEED = 42