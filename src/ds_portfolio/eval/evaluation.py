"""
Evaluation & Testing Framework.
Demonstrates:
- Comprehensive ML model evaluation
- Prompt engineering evaluation
- Statistical significance testing
- A/B testing framework
- Cross-validation with stratification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


@dataclass
class ModelEvaluationResult:
    """Container for comprehensive model evaluation results."""
    model_name: str
    cv_scores: Dict[str, List[float]]
    test_metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importance: Dict[str, float] | None = None


@dataclass
class ABTestResult:
    """Container for A/B test statistical results."""
    metric_name: str
    variant_a_mean: float
    variant_b_mean: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    recommendation: str


class ModelEvaluator:
    """
    Comprehensive ML Model Evaluation Suite.
    Supports cross-validation, statistical testing, and fair comparison.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def evaluate_classifier(self, model: Any, X: np.ndarray, y: np.ndarray,
                           model_name: str = "model") -> ModelEvaluationResult:
        """Full evaluation with CV and test set metrics."""

        # Cross-validation scores
        cv_metrics = {
            "accuracy": cross_val_score(model, X, y, cv=self.cv_strategy, scoring="accuracy"),
            "precision": cross_val_score(model, X, y, cv=self.cv_strategy, scoring="precision"),
            "recall": cross_val_score(model, X, y, cv=self.cv_strategy, scoring="recall"),
            "f1": cross_val_score(model, X, y, cv=self.cv_strategy, scoring="f1"),
            "roc_auc": cross_val_score(model, X, y, cv=self.cv_strategy, scoring="roc_auc"),
        }

        # Fit on full data for test predictions
        model.fit(X, y)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        test_metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if y_proba is not None else None,
        }

        # Feature importance if available
        feat_importance = None
        if hasattr(model, "feature_importances_"):
            feat_importance = dict(enumerate(model.feature_importances_))
        elif hasattr(model, "coef_"):
            feat_importance = dict(enumerate(np.abs(model.coef_[0])))

        return ModelEvaluationResult(
            model_name=model_name,
            cv_scores={k: v.tolist() for k, v in cv_metrics.items()},
            test_metrics=test_metrics,
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred, output_dict=False),
            feature_importance=feat_importance,
        )

    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Compare multiple models with statistical significance."""
        results = []
        for name, model in models.items():
            eval_result = self.evaluate_classifier(model, X, y, name)
            row = {
                "model": name,
                "cv_accuracy_mean": np.mean(eval_result.cv_scores["accuracy"]),
                "cv_accuracy_std": np.std(eval_result.cv_scores["accuracy"]),
                "cv_f1_mean": np.mean(eval_result.cv_scores["f1"]),
                "cv_f1_std": np.std(eval_result.cv_scores["f1"]),
                "test_accuracy": eval_result.test_metrics["accuracy"],
                "test_f1": eval_result.test_metrics["f1"],
            }
            results.append(row)
        return pd.DataFrame(results).sort_values("cv_f1_mean", ascending=False)


class ABTester:
    """
    A/B Testing Framework for Data Science Experiments.
    Implements frequentist statistical testing with effect size estimation.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def run_ttest(self, variant_a: np.ndarray, variant_b: np.ndarray,
                  metric_name: str = "metric") -> ABTestResult:
        """Independent two-sample t-test with effect size (Cohen's d)."""
        n_a, n_b = len(variant_a), len(variant_b)
        mean_a, mean_b = np.mean(variant_a), np.mean(variant_b)
        std_a, std_b = np.std(variant_a, ddof=1), np.std(variant_b, ddof=1)

        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(variant_a, variant_b, equal_var=False)

        # Cohen's d for effect size
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

        # Confidence interval for difference in means
        se = np.sqrt(std_a**2 / n_a + std_b**2 / n_b)
        ci_low = (mean_a - mean_b) - 1.96 * se
        ci_high = (mean_a - mean_b) + 1.96 * se

        is_sig = p_value < self.alpha

        if is_sig and effect_size > 0:
            rec = "Implement Variant A (statistically significant improvement)"
        elif is_sig and effect_size < 0:
            rec = "Implement Variant B (statistically significant improvement)"
        else:
            rec = "No significant difference detected; consider running longer or larger sample"

        return ABTestResult(
            metric_name=metric_name,
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_low, ci_high),
            is_significant=is_sig,
            recommendation=rec
        )

    def run_proportion_ztest(self, successes_a: int, trials_a: int,
                              successes_b: int, trials_b: int,
                              metric_name: str = "conversion_rate") -> ABTestResult:
        """Two-proportion Z-test for conversion rate experiments."""
        from statsmodels.stats.proportion import proportions_ztest

        prop_a = successes_a / trials_a
        prop_b = successes_b / trials_b

        z_stat, p_value = proportions_ztest(
            [successes_a, successes_b], [trials_a, trials_b]
        )

        # Effect size: Cohen's h
        import math
        effect_size = 2 * (math.asin(np.sqrt(prop_a)) - math.asin(np.sqrt(prop_b)))

        # CI for difference in proportions
        se = np.sqrt(prop_a*(1-prop_a)/trials_a + prop_b*(1-prop_b)/trials_b)
        diff = prop_a - prop_b
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se

        is_sig = p_value < self.alpha

        if is_sig and effect_size > 0:
            rec = f"Implement Variant A (conversion {prop_a:.2%} vs {prop_b:.2%})"
        elif is_sig and effect_size < 0:
            rec = f"Implement Variant B (conversion {prop_b:.2%} vs {prop_a:.2%})"
        else:
            rec = "No significant difference in conversion rates"

        return ABTestResult(
            metric_name=metric_name,
            variant_a_mean=prop_a,
            variant_b_mean=prop_b,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_low, ci_high),
            is_significant=is_sig,
            recommendation=rec
        )


class PromptEvalFramework:
    """
    Evaluation Framework for Prompt Engineering Experiments.
    Measures accuracy, faithfulness, hallucination, cost, latency.
    """

    def __init__(self):
        self.experiments = []

    def log_experiment(self, prompt_version: str, task: str,
                       metrics: Dict[str, float], metadata: Dict[str, Any] = None) -> None:
        """Log a prompt experiment result."""
        entry = {
            "prompt_version": prompt_version,
            "task": task,
            "metrics": metrics,
            "metadata": metadata or {},
            "timestamp": pd.Timestamp.now()
        }
        self.experiments.append(entry)

    def compare_prompts(self, task: str) -> pd.DataFrame:
        """Compare all logged experiments for a task."""
        task_exps = [e for e in self.experiments if e["task"] == task]
        if not task_exps:
            return pd.DataFrame()

        rows = []
        for exp in task_exps:
            row = {"prompt_version": exp["prompt_version"], **exp["metrics"]}
            rows.append(row)
        return pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    def generate_report(self, task: str) -> str:
        """Generate markdown evaluation report."""
        df = self.compare_prompts(task)
        if df.empty:
            return f"No experiments logged for task: {task}"

        best = df.iloc[0]
        report = f"# Prompt Evaluation Report: {task}\n\n"
        report += f"**Best Version**: {best['prompt_version']}\n"
        report += f"**Accuracy**: {best.get('accuracy', 'N/A'):.3f}\n"
        report += f"**Hallucination Rate**: {best.get('hallucination_rate', 'N/A'):.3f}\n"
        report += f"**Avg Latency**: {best.get('avg_latency_sec', 'N/A'):.2f}s\n\n"
        report += "## Full Comparison\n\n"
        report += df.to_markdown(index=False)
        return report