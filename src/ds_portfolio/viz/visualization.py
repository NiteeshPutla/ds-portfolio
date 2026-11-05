"""
Data Visualization & Exploration Module.
Demonstrates:
- Matplotlib & Seaborn for statistical visualizations
- Plotly for interactive dashboards
- Bokeh for web-based visualizations
- Professional chart styling & accessibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ChartConfig:
    """Configuration for consistent chart styling."""
    theme: str = "whitegrid"  # darkgrid, whitegrid, dark, white, ticks
    palette: str = "viridis"
    figsize: tuple = (10, 6)
    dpi: int = 100
    font_scale: float = 1.1


class EDAVisualizer:
    """
    Comprehensive Exploratory Data Analysis Visualization Suite.
    Produces publication-quality charts for data science portfolios.
    """

    def __init__(self, config: ChartConfig = ChartConfig()):
        self.config = config
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure global matplotlib/seaborn aesthetics."""
        sns.set_theme(
            style=self.config.theme,
            palette=self.config.palette,
            font_scale=self.config.font_scale,
            rc={
                "figure.figsize": self.config.figsize,
                "figure.dpi": self.config.dpi,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )

    def plot_target_distribution(self, df: pd.DataFrame, target_col: str) -> plt.Figure:
        """Visualize target variable distribution with statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        sns.countplot(data=df, x=target_col, ax=axes[0], palette="Set2")
        axes[0].set_title(f"Distribution of {target_col}")
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel("Count")

        # Add count labels
        for p in axes[0].patches:
            axes[0].annotate(f'{int(p.get_height())}',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=10)

        # Percentage plot
        counts = df[target_col].value_counts(normalize=True).sort_index()
        counts.plot(kind='bar', ax=axes[1], color=sns.color_palette("Set2"))
        axes[1].set_title(f"Percentage Distribution of {target_col}")
        axes[1].set_ylabel("Proportion")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.tight_layout()
        return fig

    def plot_numerical_distributions(self, df: pd.DataFrame,
                                      numerical_cols: List[str],
                                      target_col: Optional[str] = None) -> plt.Figure:
        """Grid of histograms with KDE for numerical features."""
        n_cols = len(numerical_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            if target_col and target_col in df.columns:
                for target_val in df[target_col].unique():
                    subset = df[df[target_col] == target_val][col].dropna()
                    sns.histplot(subset, kde=True, ax=ax, label=str(target_val),
                                alpha=0.5, stat="density")
                ax.legend()
            else:
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                  numerical_cols: List[str],
                                  method: str = "pearson") -> plt.Figure:
        """Annotated correlation heatmap for numerical features."""
        corr_matrix = df[numerical_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title(f"Correlation Matrix ({method.capitalize()})", pad=20)
        plt.tight_layout()
        return fig

    def plot_categorical_vs_target(self, df: pd.DataFrame,
                                    cat_cols: List[str],
                                    target_col: str) -> plt.Figure:
        """Stacked bar charts for categorical vs target relationship."""
        n_cols = len(cat_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        axes = axes[0]

        for i, col in enumerate(cat_cols):
            ct = pd.crosstab(df[col], df[target_col], normalize='index')
            ct.plot(kind='bar', stacked=True, ax=axes[i], colormap="Set2")
            axes[i].set_title(f"{col} vs {target_col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Proportion")
            axes[i].legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def plot_feature_importance(self, feature_names: List[str],
                                 importances: np.ndarray,
                                 top_k: int = 15) -> plt.Figure:
        """Horizontal bar chart of feature importances."""
        indices = np.argsort(importances)[::-1][:top_k]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
        bars = ax.barh(range(len(top_features)), top_importances,
                       color=sns.color_palette("viridis", len(top_features)))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_k} Feature Importances")
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_importances)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        return fig


class InteractiveDashboard:
    """
    Plotly-based Interactive Dashboard Components.
    Demonstrates web-ready visualizations for stakeholder communication.
    """

    @staticmethod
    def create_kpi_cards(metrics: Dict[str, float]) -> go.Figure:
        """Create KPI metric cards using Plotly indicator traces."""
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=list(metrics.keys()),
            specs=[[{"type": "indicator"}] * len(metrics)]
        )

        for i, (name, value) in enumerate(metrics.items(), 1):
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=value,
                    number={"font": {"size": 32}},
                    title={"text": name, "font": {"size": 14}},
                    delta={"reference": value * 0.9, "relative": True}
                ),
                row=1, col=i
            )

        fig.update_layout(height=150, showlegend=False, margin=dict(t=50, b=20))
        return fig

    @staticmethod
    def create_funnel_chart(stages: List[str], values: List[int]) -> go.Figure:
        """Conversion funnel visualization."""
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker={"color": px.colors.sequential.Viridis},
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 2}}
        ))
        fig.update_layout(
            title="Conversion Funnel",
            height=400,
            margin=dict(l=100, r=20, t=50, b=20)
        )
        return fig

    @staticmethod
    def create_time_series_decomposition(df: pd.DataFrame,
                                          date_col: str,
                                          value_col: str,
                                          freq: str = "D") -> go.Figure:
        """Interactive time series with trend/seasonality decomposition."""
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.set_index(date_col).resample(freq)[value_col].mean().interpolate()

        # Simple rolling trend
        trend = df_ts.rolling(window=7, center=True).mean()
        residual = df_ts - trend

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=["Original", "Trend", "Residual"],
            vertical_spacing=0.05
        )

        fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts.values, mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=trend.index, y=trend.values, mode='lines', name='Trend', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=residual.index, y=residual.values, mode='lines', name='Residual', line=dict(color='red')), row=3, col=1)

        fig.update_layout(height=600, title="Time Series Decomposition", showlegend=False)
        return fig

    @staticmethod
    def create_segment_comparison(df: pd.DataFrame,
                                   segment_col: str,
                                   metrics: List[str]) -> go.Figure:
        """Grouped bar chart for segment comparison."""
        fig = go.Figure()

        for metric in metrics:
            agg = df.groupby(segment_col)[metric].mean().reset_index()
            fig.add_trace(go.Bar(
                x=agg[segment_col],
                y=agg[metric],
                name=metric,
                text=agg[metric].round(2),
                textposition='outside'
            ))

        fig.update_layout(
            barmode='group',
            title="Segment Performance Comparison",
            xaxis_title=segment_col,
            yaxis_title="Metric Value",
            height=450
        )
        return fig


def save_fig(fig: plt.Figure, path: str, dpi: int = 300) -> None:
    """Save matplotlib figure with high quality."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_interactive(fig: go.Figure, path: str) -> None:
    """Save plotly figure as interactive HTML."""
    fig.write_html(path, include_plotlyjs='cdn')