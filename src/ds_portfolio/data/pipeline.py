"""
Data processing and generation pipelines covering:
1. Large-scale data processing (Pandas, Polars, PyArrow)
2. Synthetic data generation & annotation strategies
3. Feature engineering & transformations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class PriorityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SyntheticDataConfig:
    n_samples: int = 10000
    random_seed: int = 42
    include_anomalies: bool = True
    missing_rate: float = 0.05


class DataPipeline:
    """
    Demonstrates:
    - Large-scale data transformation
    - High-performance Pandas / PyArrow optimizations
    - Automated annotation strategy generation
    """

    def __init__(self, config: SyntheticDataConfig = SyntheticDataConfig()):
        self.config = config
        np.random.seed(config.random_seed)

    def generate_e_commerce_dataset(self) -> pd.DataFrame:
        """
        Generates realistic e-commerce customer interaction logs for scale analysis.
        Demonstrates schema design, mixed data types, and temporal attributes.
        """
        n = self.config.n_samples

        customer_ids = [f"CUST-{i:06d}" for i in np.random.randint(1000, 9999, size=n)]
        session_duration = np.random.exponential(scale=300, size=n)  # seconds
        pages_visited = np.random.poisson(lam=5, size=n) + 1
        purchase_amount = np.where(
            np.random.rand(n) > 0.7,
            np.random.lognormal(mean=4.0, sigma=1.0, size=n),
            0.0
        )
        category = np.random.choice(
            ["Electronics", "Fashion", "Home", "Books", "Beauty"],
            size=n,
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
        device = np.random.choice(["Mobile", "Desktop", "Tablet"], size=n, p=[0.6, 0.35, 0.05])

        # User Feedback (Text for NLP/Prompt Engineering)
        feedback_templates = [
            "Great product! Fast shipping and excellent quality.",
            "Average experience. Product was ok, but delivery took longer than expected.",
            "Terrible customer service. Item arrived damaged and returns were painful.",
            "Loved the design! Highly recommend to everyone.",
            "Not worth the price. Quality is disappointing.",
            "Prompt response from support. Resolved my query quickly."
        ]
        user_feedback = np.random.choice(feedback_templates, size=n)

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "session_duration_sec": np.round(session_duration, 2),
            "pages_visited": pages_visited,
            "purchase_amount": np.round(purchase_amount, 2),
            "category": category,
            "device": device,
            "user_feedback": user_feedback,
            "created_at": pd.date_range(start="2024-01-01", periods=n, freq="min")
        })

        # Introduce synthetic missing values for robustness
        if self.config.missing_rate > 0:
            mask = np.random.rand(*df.shape) < self.config.missing_rate
            # Keep customer_id non-null
            mask[:, 0] = False
            df = df.mask(mask)

        return df

    def create_annotation_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data Annotation Strategy Design:
        Rule-based heuristic sampling to prioritize items for human-in-the-loop annotation.
        High ambiguity or extreme edge-cases get assigned HIGH or CRITICAL priority.
        """
        df_stratified = df.copy()

        # Flag high value customers with negative feedback for urgent annotation
        negative_words = ["terrible", "damaged", "disappointing", "painful", "not worth"]
        feedback_str = df_stratified["user_feedback"].astype(str).str.lower()

        has_negative = feedback_str.apply(lambda text: any(w in text for w in negative_words))
        is_high_spend = df_stratified["purchase_amount"] > 100.0

        conditions = [
            (has_negative & is_high_spend),
            (has_negative & ~is_high_spend),
            (~has_negative & is_high_spend),
        ]
        choices = [PriorityLevel.CRITICAL.value, PriorityLevel.HIGH.value, PriorityLevel.MEDIUM.value]

        df_stratified["annotation_priority"] = np.select(conditions, choices, default=PriorityLevel.LOW.value)
        return df_stratified
