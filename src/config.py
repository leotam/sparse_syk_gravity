import os

def set_thread_safety():
    """Sets environment variables to disable implicit multithreading."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==========================================
# PHYSICS CONSTANTS
# ==========================================
BETA = 4.0
COUPLING_J = 1.0
WORMHOLE_STR = 2.5

# ==========================================
# DEFAULT SIMULATION PARAMETERS
# ==========================================
# Used by Benchmark (Fig 1), Phase Diagram (Fig 2), and Linear (Fig 3)
TIME_STEPS = 80       
TIME_RANGE = 5.0
SFF_TIME_RANGE = 20.0