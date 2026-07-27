"""Smoke tests for Phase 1.

These tests verify the core hypothesis precondition: a 1D convolution solves
the generated data better than a from-scratch over-parameterized MLP, because
the convolution has the right inductive bias (weight sharing + locality) while
the MLP only has capacity.

Run with::

    python -m src.smoke.test_conv_vs_mlp
"""
