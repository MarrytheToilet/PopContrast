"""PopContrast: training-free popularity-debiased decoding for generative recommendation.

Built on top of the `genrec` reproduction of TIGER. Modules here are intentionally
thin wrappers that *reuse* genrec's dataset / model code rather than reimplementing
it, so we stay aligned with the trained checkpoints.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# All experiment scripts read/write here; override with POPCONTRAST_RESULTS if needed.
RESULTS_DIR = os.environ.get("POPCONTRAST_RESULTS", os.path.join(REPO_ROOT, "results"))
