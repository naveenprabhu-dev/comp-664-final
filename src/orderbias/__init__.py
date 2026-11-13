"""orderbias: answer-order-bias study for learned response evaluators.

Portable core package. No hardcoded cluster paths. Data, configuration, and
interface imports stay lightweight; future model/training implementations may
use the GPU stack explicitly. Cluster launchers and environments live under the
top-level ``cluster/`` directory.
"""

__version__ = "0.1.0"
