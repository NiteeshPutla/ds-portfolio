"""
Production Machine Learning Engine using PyTorch and Scikit-Learn.
Demonstrates:
- Clean modular architectures
- PyTorch Custom Datasets & Training loops
- Scikit-Learn Model evaluation and cross-validation
- End-to-end maintainable code adhering to SOLID/DRY principles
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ds_portfolio.core.base import AbstractModelPipeline, EvaluationMetrics


class TabularPyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for clean batch loading."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DeepTabularClassifier(nn.Module):
    """Multi-layer Perceptron (MLP) for Tabular Binary Classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ECommerceConversionPipeline(AbstractModelPipeline):
    """
    Complete ML Pipeline predicting high-value purchase conversion.
    Exemplifies Scikit-Learn preprocessors and PyTorch/Sklearn dual execution.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["session_duration_sec", "pages_visited"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category", "device"])
            ]
        )
        self.model = None

    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
        """Preprocesses numerical and categorical columns cleanly."""
        df = data.dropna(subset=["session_duration_sec", "pages_visited", "category", "device"]).copy()

        # Target: Is High Purchase (purchase_amount > 50)
        y = (df["purchase_amount"] > 50.0).astype(int).values if "purchase_amount" in df.columns else None

        if not hasattr(self.preprocessor, "transformers_"):
            X = self.preprocessor.fit_transform(df)
        else:
            X = self.preprocessor.transform(df)

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains Scikit-learn or PyTorch model based on config."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        elif self.model_type == "pytorch":
            self._train_pytorch(X, y)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _train_pytorch(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, batch_size: int = 32) -> None:
        """Internal PyTorch Training Loop."""
        dataset = TabularPyTorchDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = DeepTabularClassifier(input_dim=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

        self.model = model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
        """Generates comprehensive Evaluation Metrics."""
        if self.model_type == "random_forest":
            preds = self.model.predict(X_test)
            probs = self.model.predict_proba(X_test)[:, 1]
        elif self.model_type == "pytorch":
            self.model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X_test, dtype=torch.float32)
                probs = self.model(test_tensor).numpy().flatten()
                preds = (probs > 0.5).astype(int)

        return EvaluationMetrics(
            accuracy=accuracy_score(y_test, preds),
            precision=precision_score(y_test, preds, zero_division=0),
            recall=recall_score(y_test, preds, zero_division=0),
            f1_score=f1_score(y_test, preds, zero_division=0),
            roc_auc=roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None
        )

    def predict(self, raw_input: pd.DataFrame) -> np.ndarray:
        """Inference wrapper for raw unseen DataFrames."""
        X, _ = self.preprocess(raw_input)
        if self.model_type == "random_forest":
            return self.model.predict(X)
        elif self.model_type == "pytorch":
            self.model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X, dtype=torch.float32)
                probs = self.model(test_tensor).numpy().flatten()
                return (probs > 0.5).astype(int)
