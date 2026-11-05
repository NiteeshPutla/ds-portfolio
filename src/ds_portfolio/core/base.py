"""
DS Portfolio Core - Demonstrating SOLID, DRY, highly testable code design for Data Science.
"""

from typing import Protocol, TypeVar, Generic, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class ScalerProtocol(Protocol):
    """SOLID: Interface Segregation Principle - minimal scaler contract."""
    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...
    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...


class EstimatorProtocol(Protocol):
    """SOLID: Open/Closed Principle - contract for ML models."""
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "EstimatorProtocol": ...
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...


@dataclass
class EvaluationMetrics:
    """Dataclass holding standard ML evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None = None
    mae: float | None = None
    rmse: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "mae": self.mae,
            "rmse": self.rmse,
            "metadata": self.metadata
        }


class AbstractModelPipeline(ABC):
    """
    SOLID: Single Responsibility & Open/Closed Principle.
    Abstract template for production model pipelines.
    """

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform raw input into model features."""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train internal models."""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
        """Evaluate performance strictly on test set."""
        pass

    @abstractmethod
    def predict(self, raw_input: pd.DataFrame) -> np.ndarray:
        """Inference pipeline for raw un-preprocessed data."""
        pass
