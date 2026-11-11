"""orderbias: answer-order-bias study for learned response evaluators.

Portable core package. Nothing in here may hardcode a Longleaf / cluster path
or import CUDA-only training libraries; cluster-specific pieces live under the
top-level ``cluster/`` directory (see TODO #1b).
"""

__version__ = "0.1.0"
