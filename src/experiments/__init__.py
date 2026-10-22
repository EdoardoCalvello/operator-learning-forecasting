"""
Neural operator experiments on dynamical systems.

Experiments on chaotic dynamical systems with Transformer Neural Operators.

Main experiment types:
- smoothing: data assimilation (partial -> full state reconstruction)
- forecasting: temporal prediction (past -> future state)

Structure:
- base.py: BaseExperiment class
- smoothing.py: smoothing experiments (L63, L96, KSE)
- forecasting.py: forecasting experiments (L63, L96, KSE)
- evaluation.py: evaluation of a trained checkpoint over one horizon
- evaluation_statistics.py: long-rollout evaluation by composing forecasts
"""

from .base import BaseExperiment
from .smoothing import SmoothingExperiment
from .forecasting import ForecastingExperiment

__all__ = [
    'BaseExperiment',
    'SmoothingExperiment',
    'ForecastingExperiment'
]
