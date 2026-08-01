"""Evaluation: metrics, comparisons, and visualization.

This package evaluates the effective-map regressor baselines and the 3-method
comparison (from-scratch MLP, learned conv, oracle conv).

Modules:
- ``evaluate``: the 3-method comparison + Toeplitz-ness + kernel-recovery
  metrics on train and held-out val configs.
- ``regressor_eval``: metric suite for the learned-target regressor
  (sanity check).
- ``oracle_regressor_eval``: full metric suite for the oracle-target regressor
  baseline.
- ``visualize``: figure generation (including the shared plot helpers used by
  the regressor eval modules).
- ``_shared``: internal helpers (MSE, config conversion, split sampling,
  aggregation, summary CSV) shared by the eval modules.
"""
