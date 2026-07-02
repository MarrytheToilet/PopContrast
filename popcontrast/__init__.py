"""PopSteer: training-free popularity-debiased decoding for generative recommendation.

Built on top of the `genrec` reproduction of TIGER. Modules here are intentionally
thin wrappers that *reuse* genrec's dataset / model code rather than reimplementing
it, so we stay aligned with the trained checkpoints.
"""
